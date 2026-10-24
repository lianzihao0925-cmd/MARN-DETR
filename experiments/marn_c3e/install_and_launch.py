"""Guarded atomic commit-only install, then bounded lifecycle regression."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT=Path('/root/autodl-tmp/project_marn-detr')
PACKAGE=ROOT/'rf-detr/src/rfdetr'
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
guard=json.loads((HERE/'source_guard_original.json').read_text())
for name,expected in guard.items():
    assert sha(PACKAGE/name)==expected, name
result=json.loads((HERE/'commit_check_result.json').read_text())
assert result['passed'] is True
candidate=HERE/'integration_candidate.py'
assert sha(candidate)==result['candidate_sha256']
assert result['original_sha256']==guard['marn/integration.py']
data=candidate.read_bytes()
compile(data,str(candidate),'exec')
target=PACKAGE/'marn/integration.py'
backup=ROOT/'runs/source_backups'/('marn_c3e_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
backup.mkdir(parents=True,exist_ok=False)
(backup/'integration.py').write_bytes(target.read_bytes())
temp=target.with_suffix('.c3e.tmp')
with temp.open('xb') as f:f.write(data)
os.replace(temp,target)
guard['marn/integration.py']=sha(target)
(HERE/'source_guard_optimized.json').write_text(json.dumps(guard,indent=2))
(backup/'install_record.json').write_text(json.dumps(dict(before=result['original_sha256'],after=sha(target)),indent=2))
source=ROOT/'experiments/marn_c3b/check_c3b.py'
check=HERE/'check_lifecycle.py'
assert not check.exists()
check.write_bytes(source.read_bytes())
log=(HERE/'lifecycle_launch.log').open('xb')
proc=subprocess.Popen([sys.executable,'-u',str(check)],cwd=HERE,stdin=subprocess.DEVNULL,
                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close()
(HERE/'lifecycle.pid').write_text(str(proc.pid))
print(json.dumps(dict(installed=True,backup=str(backup),lifecycle_pid=proc.pid)))
