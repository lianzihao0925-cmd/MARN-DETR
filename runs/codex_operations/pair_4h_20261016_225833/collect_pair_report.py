"""Archive small evidence after this specific paired run completes; no weights."""
from pathlib import Path
import hashlib
import json
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

root=Path('/root/autodl-tmp/project_marn-detr')
pair=root/'runs/mechanism_pair_seed42_20261016_225833'
launch=root/'runs/codex_operations/pair_4h_20261016_225833'
result=json.loads((pair/'pair_result.json').read_text())
assert result['passed'] and result['stage']=='complete', 'Training not complete'
assert (launch/'exit_code').read_text().strip()=='0'
assert result['wall_seconds']<4*3600
protocols=[json.loads((pair/v/'protocol.json').read_text()) for v in ('marn','query_only')]
keys=['seed','initial_weights_md5','baseline_config_sha256','screening_epochs','schedule_horizon_epochs','trainable_parameters','adapter_parameters','marn']
checks={k:protocols[0][k]==protocols[1][k] for k in keys}
for epoch in range(1,5):
    traces=[json.loads((pair/v/f'input_trace_epoch_{epoch:03d}.json').read_text()) for v in ('marn','query_only')]
    checks[f'epoch_{epoch}_data']=traces[0]==traces[1]
assert all(checks.values()),checks
files={}
for folder,prefix in [(pair,'pair'),(launch,'launch'),(root/'runs/baseline_10ep_seed42','baseline'),(root/'experiments/marn_c4','scripts')]:
    for p in folder.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        relative=p.relative_to(folder)
        if prefix=='scripts' and (len(relative.parts)!=1 or p.suffix not in ('.py','.md')):continue
        if p.suffix.lower() not in ('.json','.log','.csv','.yaml','.yml','.py','.sh','.md') and p.name!='exit_code':continue
        assert p.stat().st_size<25*1024**2,('Unexpected large evidence',p)
        files[prefix+'/'+relative.as_posix()]=p
manifest={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in files.items()}
verification=dict(passed=True,checks=checks,wall_seconds=result['wall_seconds'],
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True),
                  GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
                  weight_files={v:[dict(name=p.name,bytes=p.stat().st_size) for p in (pair/v).iterdir() if p.suffix in ('.ckpt','.pth')] for v in ('marn','query_only')})
archive=launch/'pair_report_20261017.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for name,p in files.items():z.write(p,name)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
    z.writestr('verification.json',json.dumps(verification,indent=2))
with ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,verification=verification)))
