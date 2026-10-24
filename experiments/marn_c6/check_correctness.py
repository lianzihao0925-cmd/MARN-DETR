"""Meaningful gradient and exact prefix-replay checks before any long run."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',RF_HOME='/root/autodl-tmp/models/roboflow')
import sys,json,time
from pathlib import Path
import torch
ROOT=Path('/root/autodl-tmp/project_marn-detr');sys.path[:0]=[str(ROOT/'experiments/marn_c4'),str(ROOT/'experiments/marn_c5')]
from retrieval_module import RetrievalAdapter
from tail_probe import TailProbe
from query_only import load_experiment
from probe_router import TrainImages,collate
from formal_tools import atomic_json

def main():
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
    a=RetrievalAdapter(256,[1,2],300).cuda().train();raw=torch.randn(1,300,256,device='cuda',requires_grad=True)
    with torch.no_grad():
        a.bank.features.normal_();a.bank.geometry[:]=torch.tensor([.5,.5,.2,.2],device='cuda');a.bank.valid.fill_(True);a.bank.quality.fill_(1)
    a.snapshot=a.bank.snapshot(1);a.retrieval_query=raw;a.capture=raw.detach();a.window_reads=1
    target={'labels':torch.tensor([1,2],device='cuda'),'boxes':torch.tensor([[.5,.5,.2,.2]]*2,device='cuda'),'marn_object_id':torch.tensor([100,101],device='cuda'),'image_id':torch.tensor([42],device='cuda')}
    outputs=[{'pred_logits':torch.full((1,300,3),3.,device='cuda'),'pred_boxes':torch.tensor([.5,.5,.2,.2],device='cuda').expand(1,300,4)}]
    idx=[[(torch.tensor([0,1]),torch.tensor([0,1]))]]
    a.observe(outputs,idx,['_0'],[target]);assert a.retrieval_count==2 and a.retrieval_query is None and a.capture is None
    a.retrieval_loss.backward();assert raw.grad is not None and bool(raw.grad.abs().max()>0);assert a.reader.key.weight.grad is not None and bool(a.reader.key.weight.grad.abs().max()>0)
    assert a.bank.features.grad is None
    report={'query_gradient_max':float(raw.grad.abs().max()),'key_gradient_max':float(a.reader.key.weight.grad.abs().max()),'bank_stopgrad':True,'retrieval_count':a.retrieval_count,'replay':{}}
    del a,raw;torch.cuda.empty_cache()
    model=load_experiment(ROOT/'experiments/marn_c5/frozen_screen/supervised/last_regular.pth','cuda').eval().requires_grad_(False)
    probe=TailProbe(model);images,targets=collate([TrainImages(8)[i] for i in range(2)]);images=images.cuda()
    state={k:v.cpu().clone() for k,v in model.adapter.state_dict().items()}
    for name in ('fp32','bf16'):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=name=='bf16'):
            full=model.model(images);replay=probe.replay()
            row={k:float((full[k].float()-replay[k].float()).abs().max()) for k in full if k in replay}
            for k in replay:torch.testing.assert_close(full[k],replay[k],rtol=0,atol=0)
            selected=probe.cache['reader'][3];changed=probe.replay(selected);null=probe.replay(selected)
            for k in replay:torch.testing.assert_close(changed[k],null[k],rtol=0,atol=0);torch.testing.assert_close(full[k],changed[k],rtol=0,atol=0)
            report['replay'][name]=dict(exact=True,max_error=row)
    for k,v in model.adapter.state_dict().items():torch.testing.assert_close(v.cpu(),state[k],rtol=0,atol=0)
    probe.close();report.update(passed=True,seconds=time.monotonic()-started)
    atomic_json(ROOT/'experiments/marn_c6/correctness.json',report);print('CORRECTNESS_DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()
