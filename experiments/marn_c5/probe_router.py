"""C6 offline benefit-router pilot. Frozen detector/bank, train-only labels.

Two conditional single-query interventions per train image. Full final-head
official loss with reference Hungarian assignments; no validation supervision.
This pilot repeats the prefix (cost reported), not an optimized online UAMR.
"""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',RF_HOME='/root/autodl-tmp/models/roboflow',HF_HOME='/root/autodl-tmp/cache/huggingface',TORCH_HOME='/root/autodl-tmp/cache/torch',XDG_CACHE_HOME='/root/autodl-tmp/cache')
import argparse,copy,gc,hashlib,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO
from pytorch_lightning import seed_everything
ROOT=Path('/root/autodl-tmp/project_marn-detr')
sys.path.insert(0,str(ROOT/'experiments/marn_c4'))
from query_only import load_experiment
from formal_tools import atomic_json,FullCOCOMetrics
from rfdetr.marn.data import MARNDataModule
from rfdetr.training.trainer import build_trainer

class TrainImages(Dataset):
    def __init__(self,n):
        self.coco=COCO(str(ROOT/'datasets/coco/annotations/instances_train2017.json'))
        ids=sorted(self.coco.imgs)
        g=torch.Generator().manual_seed(42018)
        self.ids=[ids[i] for i in torch.randperm(len(ids),generator=g)[:n].tolist()]
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        image_id=self.ids[index];info=self.coco.imgs[image_id];w,h=info['width'],info['height']
        with Image.open(ROOT/'datasets/coco/train2017'/info['file_name']) as im:
            im=TF.normalize(TF.to_tensor(TF.resize(im.convert('RGB'),[512,512])),[.485,.456,.406],[.229,.224,.225])
        anns=[a for a in self.coco.imgToAnns[image_id] if not a.get('iscrowd',0)]
        b=torch.tensor([a['bbox'] for a in anns],dtype=torch.float32).reshape(-1,4)
        b[:,2:]+=b[:,:2];b[:,0::2].clamp_(0,w);b[:,1::2].clamp_(0,h)
        keep=(b[:,2:]>b[:,:2]).all(-1);b=b[keep]
        b=torch.cat(((b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]),-1)/torch.tensor([w,h,w,h])
        return im,{'labels':torch.tensor([a['category_id'] for a in anns],dtype=torch.long)[keep],'boxes':b,'image_id':torch.tensor([image_id])}

def collate(items):return torch.stack([x for x,t in items]),[t for x,t in items]

def features(q,logits,boxes,ids):
    p=logits.float().sigmoid().index_select(-1,ids)
    normalized=p/p.sum(-1,keepdim=True).clamp_min(1e-6)
    ent=-(normalized*normalized.clamp_min(1e-6).log()).sum(-1,keepdim=True)/float(torch.log(torch.tensor(len(ids))))
    top=p.topk(2,dim=-1).values
    return torch.cat((F.layer_norm(q.float(),(q.shape[-1],)),ent,top[:,:,:1],(top[:,:,:1]-top[:,:,1:2]),boxes.float()),-1)

class Router(nn.Module):
    def __init__(self,d):
        super().__init__();self.network=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.network(x).squeeze(-1)

def fixed_cost(criterion,output,targets,indices):
    rows=[]
    for i,target in enumerate(targets):
        out={k:output[k][i:i+1].float() for k in ('pred_logits','pred_boxes')}
        n=torch.tensor(max(len(target['labels']),1),device=out['pred_logits'].device,dtype=torch.float32)
        loss=criterion.loss_labels(out,[target],[indices[i]],n,log=False)
        loss.update(criterion.loss_boxes(out,[target],[indices[i]],n))
        weighted=[value*criterion.weight_dict[key] for key,value in loss.items() if key in criterion.weight_dict]
        assert weighted,'No official loss weights found'
        rows.append(sum(weighted))
    return torch.stack(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--images',type=int,default=4096);parser.add_argument('--tag',default='router_pilot');args=parser.parse_args()
    assert 8<=args.images<=4096
    out=ROOT/'experiments/marn_c5'/args.tag;out.mkdir(exist_ok=False)
    started=time.monotonic();torch.set_num_threads(4);seed_everything(42,workers=True)
    model=load_experiment(ROOT/'runs/mechanism_pair_seed42_20261016_225833/marn/last_regular.pth','cuda').eval()
    model.requires_grad_(False);adapter=model.adapter
    bank={k:v.cpu().clone() for k,v in adapter.state_dict().items()}
    dataset=TrainImages(args.images)
    loader=DataLoader(dataset,batch_size=8,num_workers=4,shuffle=False,collate_fn=collate,pin_memory=True)
    g=torch.Generator(device='cuda').manual_seed(42019)
    control={'branch':-1};xs=[];ys=[];image_ids=[];positives=0
    def intervention(reader,inputs):
        q,logits,boxes,selected,snapshot=inputs;b,n,d=q.shape
        assert n==300
        if control['branch']==-1:
            ranks=torch.rand(b,n,generator=g,device=q.device)
            candidates=ranks.topk(2,dim=-1).indices
            # 75% of examples explore among top-60 predictions; 25% explore all.
            top=logits.float().sigmoid().index_select(-1,reader.class_ids).amax(-1).topk(60,dim=-1).indices
            choices=torch.randint(60,(b,2),generator=g,device=q.device)
            use_foreground=torch.rand(b,1,generator=g,device=q.device)<.75
            candidates=torch.where(use_foreground,top.gather(1,choices),candidates)
            # Ensure the two interventions are distinct, without dropping any row.
            candidates[:,1]=torch.where(candidates[:,1]==candidates[:,0],(candidates[:,1]+1)%n,candidates[:,1])
            ranks.scatter_(1,candidates,-1)
            context=torch.zeros_like(selected).scatter_(1,ranks.topk(29,dim=-1).indices,True)
            control.update(candidates=candidates,context=context,all_features=features(q,logits,boxes,reader.class_ids).detach())
            new=context
        else:
            new=control['context'].clone()
            if control['branch']>=0:new.scatter_(1,control['candidates'][:,control['branch']:control['branch']+1],True)
        return q,logits,boxes,new,snapshot
    handle=adapter.reader.register_forward_pre_hook(intervention)
    for batch_idx,(images,targets) in enumerate(loader):
        images=images.cuda(non_blocking=True);targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
        with torch.no_grad():
            control['branch']=-1
            before=model.model(images)
            if args.images<128 and batch_idx==0:
                control['branch']=-2
                replay=model.model(images)
                for k in ('pred_logits','pred_boxes'):torch.testing.assert_close(before[k],replay[k],rtol=0,atol=0)
            indices=model.criterion.matcher({k:before[k].float() for k in ('pred_logits','pred_boxes')},targets,group_detr=1)
            c0=fixed_cost(model.criterion,before,targets,indices)
            for branch in range(2):
                control['branch']=branch;after=model.model(images)
                delta=c0-fixed_cost(model.criterion,after,targets,indices)
                assert torch.isfinite(delta).all()
                row=torch.arange(len(images),device=images.device)
                x=control['all_features'][row,control['candidates'][:,branch]]
                xs.append(x.cpu());ys.append(delta.cpu());image_ids.extend(int(t['image_id']) for t in targets)
            del before,after
        if (batch_idx+1)%32==0:
            print('ROUTER_PROBE',batch_idx+1,'images',min((batch_idx+1)*8,args.images),'seconds',round(time.monotonic()-started,1),flush=True)
    handle.remove()
    for k,v in adapter.state_dict().items():torch.testing.assert_close(v.cpu(),bank[k],rtol=0,atol=0)
    x=torch.cat(xs);y=torch.cat(ys);imids=torch.tensor(image_ids)
    assert len(x)==2*args.images and torch.isfinite(x).all() and torch.isfinite(y).all()
    unique=torch.tensor(dataset.ids);ng=torch.Generator().manual_seed(42020);unique=unique[torch.randperm(len(unique),generator=ng)]
    holdout=unique[:max(1,len(unique)//5)];test=torch.isin(imids,holdout);train=~test
    assert set(imids[train].tolist()).isdisjoint(imids[test].tolist())
    report={'stage':'probe_complete','images':args.images,'examples':len(x),'train_examples':int(train.sum()),'holdout_examples':int(test.sum()),'source_split':'COCO train2017 only; split by image','detector_bank_frozen':True,'probe_precision':'fp32','context_queries':29,'exploration_queries_per_image':2,'backbone_reuse':False,'limitations':'offline frozen-detector pilot; no online route training, no prefix reuse, not complete UAMR or convergence evidence','label_mean':float(y.mean()),'label_std':float(y.std()),'positive_fraction':float((y>0).float().mean()),'max_abs_label':float(y.abs().max()),'seconds':time.monotonic()-started}
    torch.save({'x':x,'y':y,'image_id':imids,'heldout_image_ids':holdout},out/'train_probe_examples.pt')
    atomic_json(out/'result.json',report)
    if args.images<128:
        report.update(passed=True,stage='smoke_complete');atomic_json(out/'result.json',report);print('ROUTER_SMOKE_COMPLETED',json.dumps(report),flush=True);return
    mean=x[train].mean(0);scale=x[train].std(0).clamp_min(.01);z=((x-mean)/scale).clamp(-10,10)
    ym=y[train].mean();yscale=y[train].std().clamp_min(1e-6);target=(y-ym)/yscale
    router=Router(x.shape[-1]);optimizer=torch.optim.AdamW(router.parameters(),lr=1e-3,weight_decay=.01)
    baseline=float((target[test]-target[train].mean()).square().mean());best=baseline;best_state=None;best_epoch=-1
    for epoch in range(100):
        router.train();optimizer.zero_grad();pred=router(z[train]);loss=F.mse_loss(pred,target[train]);loss.backward();torch.nn.utils.clip_grad_norm_(router.parameters(),1);optimizer.step()
        router.eval()
        with torch.no_grad():val=float(F.mse_loss(router(z[test]),target[test]))
        if val<best:best=val;best_epoch=epoch;best_state=copy.deepcopy(router.state_dict())
    if best_state is not None:router.load_state_dict(best_state)
    router.eval()
    with torch.no_grad():
        pred=router(z[test])*yscale+ym
        a=pred-pred.mean();b=y[test]-y[test].mean();corr=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))
    report.update(holdout_constant_mse=baseline,holdout_best_mse=best,holdout_correlation=corr,best_epoch=best_epoch,gate_passed=best_state is not None and best<.98*baseline and corr>.05)
    torch.save({'format':'marn_offline_router_v1','state_dict':router.state_dict(),'feature_mean':mean,'feature_scale':scale,'label_mean':ym,'label_scale':yscale,'feature_dim':x.shape[-1],'report':report},out/'router.pt')
    if report['gate_passed']:
        mode={'value':'random'}
        router=router.cuda();mean=mean.cuda();scale=scale.cuda()
        route_counts=[]
        def route(reader,inputs):
            q,logits,boxes,selected,snapshot=inputs
            if mode['value']=='learned':
                with torch.no_grad():
                    z=((features(q,logits,boxes,reader.class_ids)-mean)/scale).clamp(-10,10)
                    scores=router(z)*float(yscale)+float(ym)
                    scores,idx=scores.topk(30,dim=-1);selected=torch.zeros_like(selected).scatter_(1,idx,scores>0)
                    route_counts.append(int(selected.sum()))
            return q,logits,boxes,selected,snapshot
        rh=adapter.reader.register_forward_pre_hook(route)
        mc=model.model_config.model_copy(update={'device':'cuda'})
        tc=model.train_config.model_copy(update={'accelerator':'gpu','devices':1,'use_ema':False,'num_workers':8,'persistent_workers':False,'eval_batch_size':8,'compute_val_loss':False,'amp_dtype':'bf16','output_dir':str(out)})
        model.model_config=mc;model.train_config=tc
        dm=MARNDataModule(mc,tc);dm.setup('validate');assert len(dm._dataset_val)==5000
        report['full_val']={}
        for name in ('random','learned'):
            mode['value']=name;folder=out/name;folder.mkdir();route_counts.clear()
            metric=FullCOCOMetrics(folder,max_dets=100,eval_base_model=True,log_per_class_metrics=False,eval_backend=tc.eval_backend)
            trainer=build_trainer(tc,mc,include_training_callbacks=False,enable_model_summary=False)
            trainer.callbacks=[c for c in trainer.callbacks if c.__class__.__name__!='COCOEvalCallback']+[metric]
            trainer.validate(model,datamodule=dm,verbose=False)
            report['full_val'][name]={'metrics':metric.captured['regular'],'selected_queries':sum(route_counts) if route_counts else 150000}
            atomic_json(out/'result.json',report);del trainer,metric;gc.collect()
        rh.remove()
    report.update(passed=True,stage='complete',seconds=time.monotonic()-started)
    atomic_json(out/'result.json',report)
    print('ROUTER_PILOT_COMPLETED',json.dumps(report),flush=True)

if __name__=='__main__':main()
