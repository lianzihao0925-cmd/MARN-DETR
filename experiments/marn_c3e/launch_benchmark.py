"""Reuse unchanged accepted three-pair timing with optimized source guard."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/project_marn-detr')
results=list((HERE/'check_outputs').glob('*/c3b_result.json'))
assert len(results)==1, results
result=json.loads(results[0].read_text())
assert result.get('passed') is True and result.get('stage')=='complete', result
guard=json.loads((HERE/'source_guard_optimized.json').read_text())
for name,expected in guard.items():
    assert hashlib.sha256((ROOT/'rf-detr/src/rfdetr'/name).read_bytes()).hexdigest()==expected, name
folder=HERE/'throughput'
folder.mkdir(exist_ok=False)
original=ROOT/'experiments/marn_c3c/benchmark_c3c.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='0a3513a5429ca74908131a5addcd9fc5512737951b0b40cddd88b298ee3b7c9e'
(folder/'benchmark_c3c.py').write_bytes(original.read_bytes())
(folder/'source_guard.json').write_text(json.dumps(guard,indent=2))
with (folder/'launch.log').open('xb') as log:
    proc=subprocess.Popen([sys.executable,'-u',str(folder/'benchmark_c3c.py')],cwd=folder,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(folder/'launch.pid').write_text(str(proc.pid))
print(json.dumps(dict(benchmark_pid=proc.pid,folder=str(folder),protocol='unchanged C3C three paired repetitions')))
