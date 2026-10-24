"""Collect small C7 evidence; never read or copy detector checkpoints."""
import argparse, hashlib, json, time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('tag',choices=['frozen_smoke','frozen_screen'])
    args=parser.parse_args()
    root=HERE/args.tag
    controller=json.loads((root/'controller.json').read_text())
    assert controller.get('passed'), 'Only archive a completed, verified pair'
    report={'collected_unix':time.time(),'tag':args.tag,'controller':controller,'arms':{},'source_sha256':{}}
    for name in ('frozen_module.py','run_frozen_arm.py','run_frozen_pair.py','collect_frozen_report.py'):
        report['source_sha256'][name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for kind in ('query_only','marn','supervised'):
        folder=root/kind
        files={p.name:json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))}
        assert files['frozen_guard.json']['passed'] and files['completed.json']['passed']
        assert files['completed.json']['global_step']==(4 if args.tag=='frozen_smoke' else 7393)
        report['arms'][kind]={'files':files,'log_tail':(root/(kind+'.log')).read_text()[-3500:]}
    destination=HERE/(args.tag+'_report.json')
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(destination, destination.stat().st_size, hashlib.sha256(destination.read_bytes()).hexdigest())

if __name__=='__main__':main()
