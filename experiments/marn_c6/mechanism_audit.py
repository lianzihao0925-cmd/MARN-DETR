"""Train-only readout audit of saved pilots. No training or checkpoint changes."""
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', RF_HOME='/root/autodl-tmp/models/roboflow')
import sys, json, time, hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path('/root/autodl-tmp/project_marn-detr')
sys.path[:0]=[str(ROOT/'experiments/marn_c4'),str(ROOT/'experiments/marn_c5')]
from query_only import load_experiment
from probe_router import TrainImages,collate
from formal_tools import atomic_json
from torch.utils.data import DataLoader

def quantiles(x):
    x=x.float().flatten()
    return dict(n=x.numel(),mean=float(x.mean()),std=float(x.std()),quantiles=torch.quantile(x,torch.tensor([0.,.01,.1,.5,.9,.99,1.])).tolist())

def main():
    started=time.monotonic();torch.set_num_threads(4)
    out=ROOT/'experiments/marn_c6/audit';out.mkdir(parents=True,exist_ok=False)
    data=torch.load(ROOT/'experiments/marn_c5/router_pilot/train_probe_examples.pt',map_location='cpu',weights_only=True)
    y=data['y'];test=torch.isin(data['image_id'],data['heldout_image_ids']);train=~test
    report={'purpose':'train-only audit; no COCO validation model selection','cached_label':{'all':quantiles(y),'train':quantiles(y[train]),'selection_holdout':quantiles(y[test]),'zeros':int((y==0).sum()),'positive':float((y>0).float().mean()),'top_1pct_squared_energy_fraction':float(y.square().topk(max(1,len(y)//100)).values.sum()/y.square().sum().clamp_min(1e-20))},'models':{}}
    loader=DataLoader(TrainImages(128),batch_size=8,num_workers=4,shuffle=False,collate_fn=collate)
    for kind,weight in [('full_marn',ROOT/'runs/mechanism_pair_seed42_20261016_225833/marn/last_regular.pth'),('frozen_marn',ROOT/'experiments/marn_c5/frozen_screen/marn/last_regular.pth'),('frozen_supervised',ROOT/'experiments/marn_c5/frozen_screen/supervised/last_regular.pth')]:
        model=load_experiment(weight,'cuda').eval().requires_grad_(False);a=model.adapter
        state={k:v.cpu().clone() for k,v in a.state_dict().items()};rows={}
        for precision in ('fp32','bf16'):
            stats={'matches':0,'head_class_correct':0,'key_class_correct':0,'full_cosine_class_correct':0,'mass_true':0.,'cosine':[],'active_components':[],'delta_ratio':[],'raw_dtypes':{},'null_max_logits':0.}
            captured={}
            def reader_hook(module,inputs,outputs):captured['reader']=inputs
            def refine_hook(module,inputs,output):
                q,m,e,mask=inputs
                if mask.any():
                    diff=output.float()-q.float();stats['active_components'].append((diff[mask]!=0).float().mean().cpu());stats['delta_ratio'].append((diff.norm(dim=-1)[mask]/q.float().norm(dim=-1)[mask].clamp_min(1e-6)).cpu())
                stats['raw_dtypes'][str(q.dtype)]=stats['raw_dtypes'].get(str(q.dtype),0)+1
            h=a.reader.register_forward_hook(reader_hook);rh=a.refiner.register_forward_hook(refine_hook)
            for batch_idx,(images,targets) in enumerate(loader):
                images=images.cuda();targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                    output=model.model(images)
                    if batch_idx==0:
                        replay=model.model(images)
                        stats['null_max_logits']=float((output['pred_logits'].float()-replay['pred_logits'].float()).abs().max())
                        assert stats['null_max_logits']==0
                    q,logits,boxes,selected,snap=captured['reader']
                    indices=model.criterion.matcher({'pred_logits':logits.float(),'pred_boxes':boxes.float()},targets,group_detr=1)
                    valid=snap.valid & (snap.quality>0) & torch.isfinite(snap.features).all(-1)
                    raw=snap.features[valid].float();cats=snap.classes[valid];geometry=snap.geometry[valid].float();quality=snap.quality[valid].float()
                    for b,((src,dst),target) in enumerate(zip(indices,targets)):
                        src,dst=src.to(q.device),dst.to(q.device)
                        if not len(src):continue
                        features=q[b,src];truth=target['labels'][dst]
                        cosine=F.normalize(a.reader.key(features).float(),dim=-1)@F.normalize(a.reader.key(raw).float(),dim=-1).T
                        p=logits[b,src].float().sigmoid().index_select(-1,a.reader.class_ids);p=p/p.sum(-1,keepdim=True).clamp_min(1e-6)
                        cls=p[:,a.reader.class_index[cats]].clamp_min(1e-6).log()
                        geo=-(boxes[b,src,2:].float().clamp_min(1e-6).log()[:,None]-geometry[:,2:].clamp_min(1e-6).log()[None]).abs().mean(-1)
                        age=(snap.step-snap.updated_step[valid]).clamp_min(0).float()/a.reader.config.age_scale
                        c=a.reader.config;scores=cosine/c.temperature+c.class_weight*cls+c.geometry_weight*geo+c.quality_weight*quality.clamp_min(1e-6).log()[None]-c.age_weight*age[None]
                        top,idx=scores.topk(4,dim=-1);weights=top.softmax(-1)
                        fullcos=F.normalize(features.float(),dim=-1)@F.normalize(raw,dim=-1).T
                        stats['matches']+=len(src);stats['head_class_correct']+=int((a.reader.class_ids[p.argmax(-1)]==truth).sum());stats['key_class_correct']+=int((cats[idx[:,0]]==truth).sum());stats['full_cosine_class_correct']+=int((cats[fullcos.argmax(-1)]==truth).sum());stats['mass_true']+=float((weights*(cats[idx]==truth[:,None])).sum());stats['cosine'].append(cosine.max(-1).values.cpu())
            h.remove();rh.remove();n=max(stats['matches'],1)
            for name in ('head_class_correct','key_class_correct','full_cosine_class_correct','mass_true'):stats[name]=stats[name]/n
            for name in ('cosine','active_components','delta_ratio'):stats[name]=quantiles(torch.cat([x.reshape(-1) for x in stats[name]]))
            rows[precision]=stats
        for k,v in a.state_dict().items():torch.testing.assert_close(v.cpu(),state[k],rtol=0,atol=0)
        report['models'][kind]={'weights':str(weight),'precision':rows,'state_unchanged':True}
        atomic_json(out/'result.json',report);del model,a;torch.cuda.empty_cache()
    report.update(passed=True,seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'result.json',report);print('AUDIT_DONE',json.dumps(report),flush=True)

if __name__=='__main__':main()
