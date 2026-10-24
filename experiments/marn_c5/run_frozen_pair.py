"""Sequential three-arm bounded pilot; no restart on failure."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/project_marn-detr')
HERE=Path(__file__).resolve().parent
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    tag='frozen_smoke' if args.smoke else 'frozen_screen'
    out=HERE/tag;out.mkdir(exist_ok=False)
    started=time.time();cap=600 if args.smoke else 7200
    cutoff=min(started+cap,1792274400) # 2026-10-18 06:00 +08:00
    report={'passed':False,'started_unix':started,'hard_stop_unix':cutoff,'scope':'frozen official detector; adapter-only single seed three-arm pilot','arms':[]}
    save(out/'controller.json',report)
    for kind in ('query_only','marn','supervised'):
        cmd=[sys.executable,str(HERE/'run_frozen_arm.py'),'--kind',kind,'--output',str(out/kind),'--epochs','1']
        if args.smoke:cmd+=['--smoke']
        remaining=cutoff-time.time()
        if remaining<=60:raise RuntimeError('Budget expired')
        with (out/(kind+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
            row={'kind':kind,'pid':child.pid,'started_unix':time.time()};report['arms'].append(row);save(out/'controller.json',report)
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                row['timed_out']=True;code=124
            row.update(exit_code=code,seconds=time.time()-row['started_unix']);save(out/'controller.json',report)
            if code:raise RuntimeError(f'{kind} failed: {code}')
    protocols=[json.loads((out/k/'protocol.json').read_text()) for k in ('query_only','marn','supervised')]
    audits=[json.loads((out/k/'epoch_audit.json').read_text()) for k in ('query_only','marn','supervised')]
    assert len({p['frozen_detector_sha256'] for p in protocols})==1
    assert len({p['adapter_initial_sha256'] for p in protocols})==1
    traces=[json.loads((out/k/'input_trace_epoch_001.json').read_text()) for k in ('query_only','marn','supervised')]
    assert traces[0]==traces[1]==traces[2]
    assert all(a[0]['all_losses_finite'] for a in audits)
    report.update(passed=True,wall_seconds=time.time()-started,paired_inputs_verified=True)
    save(out/'controller.json',report)
    print('FROZEN_PAIR_COMPLETED',json.dumps(report),flush=True)
if __name__=='__main__':main()
