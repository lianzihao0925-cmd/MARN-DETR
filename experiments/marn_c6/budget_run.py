"""Unique sequential GPU job ledger; absolute deadline and shared remaining cap."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/project_marn-detr/experiments/marn_c6')
DEADLINE=1792274400
def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.command and 0<a.seconds<=10260 and a.job.replace('_','').isalnum()
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'gpu_budget.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'gpu_budget.json';ledger=json.loads(path.read_text()) if path.exists() else dict(deadline=DEADLINE,phase_cap_seconds=14400,previous_conservative_seconds=4140,jobs=[])
    assert not any(j['name']==a.job for j in ledger['jobs']),'Job already submitted'
    assert all('seconds' in j for j in ledger['jobs']),'Unresolved previous job; do not reset budget'
    remaining=14400-4140-sum(j['seconds'] for j in ledger['jobs']);limit=min(a.seconds,remaining,DEADLINE-time.time()-5)
    assert limit>0
    # Do not launch over a pre-existing GPU workload.
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not busy,'GPU busy: '+busy
    assert __import__('shutil').disk_usage(ROOT).free>4*1024**3
    job=dict(name=a.job,started_unix=time.time(),limit_seconds=limit,command=a.command);ledger['jobs'].append(job);save(path,ledger)
    with (ROOT/(a.job+'.log')).open('x') as log:
        proc=subprocess.Popen(a.command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);job['pid']=proc.pid;save(path,ledger)
        try:code=proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    job.update(exit_code=code,seconds=time.time()-job['started_unix'],finished_unix=time.time());save(path,ledger)
    print('BUDGET_JOB_DONE',json.dumps(job),flush=True)
if __name__=='__main__':main()
