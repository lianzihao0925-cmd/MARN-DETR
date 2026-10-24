"""One sequential screen, run only inside shared budget guard, no auto-retry."""
import sys,subprocess,json,time,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
def save(p,r):q=p.with_suffix('.tmp');q.write_text(json.dumps(r,indent=2));q.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--tag',required=True);a=p.parse_args()
    assert a.tag.replace('_','').isalnum();out=HERE/a.tag;out.mkdir(exist_ok=False);started=time.time();r=dict(passed=False,seed=a.seed,smoke=a.smoke,arms=[]);save(out/'controller.json',r)
    for kind in ('query_only','random','online'):
        cmd=[sys.executable,str(HERE/'run_online_arm.py'),'--kind',kind,'--output',str(out/kind),'--seed',str(a.seed)]+(['--smoke'] if a.smoke else [])
        with (out/(kind+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);row=dict(kind=kind,pid=child.pid,started=time.time());r['arms'].append(row);save(out/'controller.json',r)
            code=child.wait();row.update(exit_code=code,seconds=time.time()-row['started']);save(out/'controller.json',r)
            if code:raise RuntimeError(f'{kind} failed; no automatic rerun')
    protocols=[json.loads((out/k/'protocol.json').read_text()) for k in ('query_only','random','online')]
    assert len({p['initial_parameter_sha256'] for p in protocols})==1
    assert len({p['parameters'] for p in protocols})==1
    traces=[json.loads((out/k/'input_trace.json').read_text()) for k in ('query_only','random','online')];assert traces[0]==traces[1]==traces[2]
    summaries={k:json.loads((out/k/'completed.json').read_text()) for k in ('query_only','random','online')}
    assert len({v['global_step'] for v in summaries.values()})==1
    r.update(passed=True,paired_init_data_updates=True,seconds=time.time()-started,summaries=summaries);save(out/'controller.json',r);print('ONLINE_SCREEN_DONE',json.dumps(r),flush=True)
if __name__=='__main__':main()
