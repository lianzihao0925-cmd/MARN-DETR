"""Collect verified small evidence only; never read large detector weights."""
import argparse,json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();assert a.tag.replace('_','').isalnum()
    folder=HERE/a.tag;controller=json.loads((folder/'controller.json').read_text());assert controller['passed'] and controller['paired_init_data_updates']
    report=dict(collected_unix=time.time(),tag=a.tag,controller=controller,budget=json.loads((HERE/'gpu_budget.json').read_text()),arms={},source_sha256={})
    for kind in ('query_only','random','online'):
        arm=folder/kind;files={}
        for path in sorted(arm.glob('*.json')):files[path.name]=json.loads(path.read_text())
        assert files['completed.json']['passed'] and files['completed.json']['native_checkpoint_strict'] and files['completed.json']['inference_export_strict']
        report['arms'][kind]=dict(files=files,log_tail=(folder/(kind+'.log')).read_text(errors='replace')[-3000:])
    for path in sorted(HERE.glob('*.py')):report['source_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    import rfdetr.marn.core,rfdetr.marn.integration,rfdetr.marn.lifecycle
    for module in (rfdetr.marn.core,rfdetr.marn.integration,rfdetr.marn.lifecycle):
        path=Path(module.__file__);report['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    q=controller['summaries']['query_only'];random=controller['summaries']['random'];online=controller['summaries']['online']
    report['comparison']=dict(regular_random_minus_query_pp=100*(random['regular']['AP']-q['regular']['AP']),regular_online_minus_query_pp=100*(online['regular']['AP']-q['regular']['AP']),regular_online_minus_random_pp=100*(online['regular']['AP']-random['regular']['AP']),official_initial_AP_pp=52.92240381267155,ema_separate=True,interpretation='development single paired seed and one epoch; no convergence, significant gain, or SOTA claim; previous full/frozen protocols separate')
    path=HERE/(a.tag+'_report.json');data=json.dumps(report,indent=2,ensure_ascii=False).encode();path.write_bytes(data)
    print('REPORT_READY',str(path),len(data),hashlib.sha256(data).hexdigest())
    print('COMPARISON',json.dumps(report['comparison']))
if __name__=='__main__':main()
