"""Budgeted, detached-friendly paired screening controller; not convergence proof."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/project_marn-detr')


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-hours',type=int,choices=[4,8],required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    epochs=4 if args.gpu_hours==4 else 8
    assert json.loads((HERE/'query_only_check/query_only_result.json').read_text())['passed']
    formal=[json.loads(p.read_text()) for p in (HERE/'check_outputs').glob('*/c4_result.json')]
    assert any(x.get('passed') and x.get('stage')=='complete' for x in formal)
    guard=json.loads((ROOT/'experiments/marn_c3e/source_guard_optimized.json').read_text())
    for name,digest in guard.items():
        assert hashlib.sha256((ROOT/'rf-detr/src/rfdetr'/name).read_bytes()).hexdigest()==digest,name
    # Refuse to contend with an existing GPU training process.
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not gpu, 'GPU already in use: '+gpu
    if args.check_only:
        print(json.dumps(dict(ready=True,training_started=False,gpu_hours_cap=args.gpu_hours,
                              epochs_per_arm=epochs,variants=['marn','query_only'],seed=42,
                              schedule_horizon_epochs=10)))
        return
    folder=ROOT/'runs'/('mechanism_pair_seed42_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(exist_ok=False)
    started=time.monotonic();deadline=started+args.gpu_hours*3600
    report=dict(passed=False,stage='starting',gpu_hours_cap=args.gpu_hours,epochs_per_arm=epochs,
                seed=42,schedule_horizon_epochs=10,order=['marn','query_only'],arms=[],
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                installed_source_sha256=guard,
                caution='one paired seed and a shorter screen; no multi-seed/convergence/SOTA claim')
    status=folder/'pair_result.json';atomic(status,report)
    print('PAIR_OUTPUT',folder,flush=True)
    for variant in report['order']:
        report['stage']=variant;atomic(status,report)
        path=folder/variant
        with (folder/(variant+'_launch.log')).open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(HERE/'run_arm.py'),'--variant',variant,
                                   '--epochs',str(epochs),'--output',str(path)],cwd=HERE,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=proc.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                report.update(stage='budget_exhausted',wall_seconds=time.monotonic()-started)
                atomic(status,report)
                raise RuntimeError('Budget reached; retain native last checkpoint, do not compare unequal updates')
        if code!=0:
            report.update(stage='arm_failed',failed_variant=variant,exit_code=code,wall_seconds=time.monotonic()-started)
            atomic(status,report);raise RuntimeError('Arm failed: '+variant)
        result=json.loads((path/'completed.json').read_text())
        report['arms'].append(result);atomic(status,report)
    left,right=[json.loads((folder/x/'protocol.json').read_text()) for x in report['order']]
    assert left['trainable_parameters']==right['trainable_parameters']
    assert left['adapter_parameters']==right['adapter_parameters']
    assert left['marn']==right['marn']
    for key in ['seed','initial_weights_md5','baseline_config_sha256','screening_epochs','schedule_horizon_epochs']:
        assert left[key]==right[key],key
    for epoch in range(1,epochs+1):
        traces=[json.loads((folder/x/('input_trace_epoch_%03d.json'%epoch)).read_text()) for x in report['order']]
        assert traces[0]==traces[1],('Paired data order/augmentation mismatch',epoch)
    report.update(passed=True,stage='complete',wall_seconds=time.monotonic()-started)
    atomic(status,report)
    print('PAIR_SCREEN_COMPLETED',folder,flush=True)


if __name__=='__main__':main()
