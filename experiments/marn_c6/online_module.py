"""Online hard-budget UAMR with periodic train-only conditional tail probes."""
import sys,time
sys.path.insert(0,'/root/autodl-tmp/project_marn-detr/experiments/marn_c5')
import torch
from torch import nn
from torch.nn import functional as F
from retrieval_module import RetrievalAdapter,RetrievalExperiment
from tail_probe import TailProbe
from probe_router import features,fixed_cost
from query_only import ExperimentModule,description
from rfdetr.marn.lifecycle import boundary
from pathlib import Path
import os


class AdapterLROverrides:
    def configure_optimizers(self):
        # Scope the extension to this isolated construction, then restore the
        # native helper. Native scheduler construction sees the final groups.
        import rfdetr.training.module_model as native
        original=native.get_param_dict
        adapter_ids={id(p) for p in self.adapter.parameters()};router_ids={id(p) for p in self.adapter.router.parameters()}
        def groups(ns,model):
            result=[]
            for group in original(ns,model):
                buckets={'base':[],'adapter':[],'router':[]}
                for p in group['params']:buckets['router' if id(p) in router_ids else 'adapter' if id(p) in adapter_ids else 'base'].append(p)
                for name,params in buckets.items():
                    if params:
                        item={**group,'params':params}
                        if name!='base':item['lr']=1e-3 if name=='router' else 1e-4
                        result.append(item)
            assert len({id(p) for g in result for p in g['params']})==sum(len(g['params']) for g in result)
            return result
        native.get_param_dict=groups
        try:return super().configure_optimizers()
        finally:native.get_param_dict=original

class BenefitRouter(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(263,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,x):return self.network(x.float()).squeeze(-1)

class ControlExperiment(AdapterLROverrides,ExperimentModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.random.fork_rng(devices=[]):self.adapter.router=BenefitRouter()
        self.supervision_examples=0;self.probe_seconds=0.;self.probe_examples=0;self.probe_batches=0;self.null_verified=False

def export_c6(module,path,source):
    boundary(module.adapter);path=Path(path)
    payload=dict(format='marn_online_inference_v1',kind=module.route_kind if isinstance(module,OnlineExperiment) else 'query_only',source=source,
                 **description(module),model_config=module.model_config.model_dump(mode='json'),train_config=module.train_config.model_dump(mode='json'),
                 online=dict(route_start=module.adapter.route_start,probe_interval=module.probe_interval,retrieval_weight=module.retrieval_weight) if isinstance(module,OnlineExperiment) else {},
                 model={k:v.detach().cpu().clone() for k,v in module.model.state_dict().items()})
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('xb') as f:torch.save(payload,f)
    os.replace(tmp,path)

def load_c6(path,device='cpu'):
    from rfdetr.config import RFDETRSmallConfig,TrainConfig
    from rfdetr.marn.integration import MARNConfig
    p=torch.load(path,map_location='cpu',weights_only=True);assert p['format']=='marn_online_inference_v1'
    mc=RFDETRSmallConfig(**{**p['model_config'],'pretrain_weights':None,'device':'cpu'});tc=TrainConfig(**{**p['train_config'],'accelerator':'cpu','devices':1})
    cls=ControlExperiment if p['kind']=='query_only' else OnlineExperiment
    extra={} if p['kind']=='query_only' else {**p['online'],'route_kind':p['kind']}
    m=cls(mc,tc,variant='query_only' if p['kind']=='query_only' else 'marn',class_ids=p['marn_class_ids'],marn_config=MARNConfig(**p['marn_config']),**extra)
    m.model.load_state_dict(p['model'],strict=True);return m.to(device).eval()

class OnlineAdapter(RetrievalAdapter):
    def route_reader(self,reader,args):
        if self.route_kind!='online' or getattr(self,'_marn_probe_replay',False):return args
        q,logits,boxes,selected,snapshot=args
        step=int(self.committed_step)
        if step<self.route_start:return args
        # Same relative route in repeated training groups, same <=30 positive
        # benefit selection semantics at train and inference.
        n=self.num_queries;groups=q.shape[1]//n
        with torch.no_grad():
            x=features(q[:,:n],logits[:,:n],boxes[:,:n],reader.class_ids)
            score=self.router(x);v,idx=score.topk(self.config.queries_per_group,dim=-1)
            one=torch.zeros_like(selected[:,:n]).scatter_(1,idx,v>0)
            selected=one[:,None].expand(-1,groups,-1).reshape(q.shape[:2])
        self.route_selected_sum+=int(one.sum());self.route_images+=len(q)
        return q,logits,boxes,selected,snapshot

class OnlineExperiment(AdapterLROverrides,RetrievalExperiment):
    def __init__(self,*args,route_kind='online',probe_interval=16,route_start=1000,**kwargs):
        super().__init__(*args,**kwargs)
        self.route_kind=route_kind;self.probe_interval=int(probe_interval)
        self.adapter.__class__=OnlineAdapter
        with torch.random.fork_rng(devices=[]):self.adapter.router=BenefitRouter()
        self.adapter.route_kind=route_kind;self.adapter.route_start=int(route_start)
        self.adapter.route_selected_sum=0;self.adapter.route_images=0
        self.adapter.reader.register_forward_pre_hook(self.adapter.route_reader)
        self.probe=None;self.probe_seconds=0.;self.probe_examples=0;self.probe_batches=0
        self.probe_positive=0;self.probe_abs_sum=0.;self.null_verified=False
        self.last_probe_loss=None

    def on_fit_start(self):
        super().on_fit_start()
        if self.route_kind=='online':self.probe=TailProbe(self)

    def training_step(self,batch,batch_idx):
        result=super().training_step(batch,batch_idx)
        step=int(self.adapter.committed_step)
        aux=self.adapter.router.network[-1].weight.sum()*0
        if self.route_kind=='online' and step>=self.adapter.config.warmup_steps and step%self.probe_interval==0:
            start=time.monotonic();aux=self.route_supervision(batch[1],step);self.probe_seconds+=time.monotonic()-start
        if isinstance(result,dict):result['loss']=result['loss']+.1*aux
        else:result=result+.1*aux
        self.last_probe_loss=float(aux.detach())
        if self.probe is not None:self.probe.reset()
        return result

    def route_supervision(self,targets,step):
        probe=self.probe;a=self.adapter;q,logits,boxes,selected,snap=probe.cache['reader'];b,total,_=q.shape;n=a.num_queries;groups=total//n
        assert total%n==0 and snap is a.snapshot
        gen=torch.Generator(device=q.device).manual_seed(830001+step)
        with torch.no_grad():
            # Keep original batch dimensions in the tail
            # to preserve grouped attention and all native kernel shapes.
            candidate=torch.randint(n,(b,1),generator=gen,device=q.device)
            fg=logits[:,:n].float().sigmoid().index_select(-1,a.reader.class_ids).amax(-1).topk(60,dim=-1).indices
            choice=torch.randint(60,(b,1),generator=gen,device=q.device)
            candidate=torch.where(torch.rand(b,1,generator=gen,device=q.device)<.5,fg.gather(1,choice),candidate)
            ranks=torch.rand(b,n,generator=gen,device=q.device);ranks.scatter_(1,candidate,-1)
            if step>=a.route_start:
                scores=a.router(features(q[:,:n],logits[:,:n],boxes[:,:n],a.reader.class_ids));scores=scores.scatter(1,candidate,-torch.inf)
                values,idx=scores.topk(29,dim=-1);one=torch.zeros_like(selected[:,:n]).scatter_(1,idx,values>0)
            else:one=torch.zeros_like(selected[:,:n]).scatter_(1,ranks.topk(29,dim=-1).indices,True)
            context=one[:,None].expand(-1,groups,-1).reshape(b,total)
            added=one.clone().scatter_(1,candidate,True);added=added[:,None].expand(-1,groups,-1).reshape(b,total)
            # Probe branches share decoder/dropout RNG, and fork_rng restores
            # main training RNG so probes cannot alter subsequent data/dropout.
            cuda_index=q.device.index
            with torch.random.fork_rng(devices=[cuda_index]):
                rng=torch.cuda.get_rng_state(q.device);cpu_rng=torch.get_rng_state()
                before=probe.replay(context)
                torch.cuda.set_rng_state(rng,q.device);torch.set_rng_state(cpu_rng)
                after=probe.replay(added)
                if not self.null_verified:
                    torch.cuda.set_rng_state(rng,q.device);torch.set_rng_state(cpu_rng);null=probe.replay(context)
                    for k in before:torch.testing.assert_close(before[k],null[k],rtol=0,atol=0)
                    self.null_verified=True
            before={k:v[:,:n].float() for k,v in before.items()};after={k:v[:,:n].float() for k,v in after.items()}
            indices=self.criterion.matcher(before,targets,group_detr=1)
            delta=fixed_cost(self.criterion,before,targets,indices)-fixed_cost(self.criterion,after,targets,indices)
            assert torch.isfinite(delta).all()
            x=features(q[:,:n],logits[:,:n],boxes[:,:n],a.reader.class_ids)[torch.arange(b,device=q.device),candidate[:,0]].detach()
        # Explicit robust proxy for the measured heavy-tailed labels; raw
        # conditional differences are still logged, never called AP gains.
        target=(delta/.001).clamp(-10,10).detach()
        loss=F.smooth_l1_loss(a.router(x),target)
        self.probe_examples+=b;self.probe_batches+=1;self.probe_positive+=int((delta>0).sum());self.probe_abs_sum+=float(delta.abs().sum())
        assert a.retrieval_query is None and a.capture is None
        return loss

    def on_train_end(self):
        if self.probe is not None:self.probe.close();self.probe=None
        super().on_train_end()

    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['marn_online_protocol']=dict(route_kind=self.route_kind,probe_interval=self.probe_interval,route_start=self.adapter.route_start,label_scale=.001,label_clip=10,route_weight=.1)

    def on_load_checkpoint(self,checkpoint):
        expected=dict(route_kind=self.route_kind,probe_interval=self.probe_interval,route_start=self.adapter.route_start,label_scale=.001,label_clip=10,route_weight=.1)
        assert checkpoint.get('marn_online_protocol')==expected
        super().on_load_checkpoint(checkpoint)
