"""Parameter-matched query-only control: no cross-image reads, writes, or losses.

The tied key projection transforms the current selected query into its own
conditioning vector. It uses all learned parameters of the MARN adapter.
Compute is reported separately: this does not pretend to match bank-scan FLOPs.
"""
import torch
import os
from pathlib import Path
from torch.nn import functional as F
from rfdetr.config import RFDETRSmallConfig,TrainConfig
from rfdetr.marn.integration import MARNAdapter,MARNConfig
from rfdetr.marn.lifecycle import MARNTrainingModule,boundary,description


class QueryOnlyAdapter(MARNAdapter):
    def forward(self,raw,normalized,reference,heads,bbox_reparam):
        if not self.enabled:return raw
        step=int(self.committed_step)
        if self.training:
            self.window_reads+=1
            self.observed=False
        elif self.pending or self.snapshot is not None:
            raise RuntimeError('Evaluation must start at an optimizer boundary')
        b,q,dim=raw.shape
        if q%self.num_queries:raise ValueError('Invalid group shape')
        groups=q//self.num_queries
        generator=torch.Generator(device=raw.device)
        generator.manual_seed(420001+step*1009+(self.window_reads if self.training else 0))
        ranks=torch.rand(b,self.num_queries,generator=generator,device=raw.device)
        idx=ranks.topk(self.config.queries_per_group,dim=-1).indices
        one=torch.zeros(b,self.num_queries,dtype=torch.bool,device=raw.device)
        one.scatter_(1,idx,True)
        selected=one[:,None].expand(-1,groups,-1).reshape(b,q)
        if step<self.config.warmup_steps:selected=torch.zeros_like(selected)
        bi,qi=selected.nonzero(as_tuple=True)
        memory=torch.zeros_like(raw)
        summary=raw.new_zeros((*raw.shape[:2],3),dtype=torch.float32)
        if bi.numel():
            current=raw[bi,qi]
            keys=self.reader.key(current)
            condition=F.linear(keys,self.reader.key.weight.T).to(current.dtype)
            memory[bi,qi]=condition
            summary[bi,qi]=torch.stack((
                F.cosine_similarity(current.float(),condition.float(),dim=-1),
                keys.float().square().mean(-1).sqrt().tanh(),
                torch.ones_like(keys[:,0],dtype=torch.float32)),dim=-1)
        self.last_read=dict(step=step,selected=int(selected.sum()),accepted=int(selected.sum()),groups=groups)
        return self.refiner(raw,memory,summary,selected)

    @torch.no_grad()
    def observe(self,outputs,all_indices,suffixes,targets):
        if self.enabled and self.training:self.observed=True

    @torch.no_grad()
    def commit(self):
        if not self.enabled:return
        if not self.training or not self.observed:raise RuntimeError('Incomplete query-only optimizer window')
        if self.pending or self.snapshot is not None or self.capture is not None:
            raise RuntimeError('Query-only must never collect prototype state')
        self.committed_step.add_(1)
        self.last_commit=dict(step=int(self.committed_step),microbatches=self.window_reads,
                              unique_objects=0,fill=0,update=0,replace=0,valid_slots=0,covered_classes=0)
        self.window_reads=0
        self.observed=False


class ExperimentModule(MARNTrainingModule):
    def __init__(self,*args,variant='marn',**kwargs):
        if variant not in ('marn','query_only'):raise ValueError(variant)
        self.variant=variant
        super().__init__(*args,**kwargs)
        if variant=='query_only':
            # Same initialized tensors and names, no second random initialization.
            self.adapter.__class__=QueryOnlyAdapter
            self.criterion.marn_observer=self.adapter.observe

    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['marn_experiment_variant']=self.variant

    def on_load_checkpoint(self,checkpoint):
        if checkpoint.get('marn_experiment_variant')!=self.variant:raise ValueError('Wrong experiment variant')
        super().on_load_checkpoint(checkpoint)


def export_experiment(module,path,source):
    boundary(module.adapter)
    if source not in ('regular','ema'):raise ValueError(source)
    path=Path(path)
    payload=dict(format='marn_mechanism_inference_v1',variant=module.variant,source=source,
                 **description(module),model_config=module.model_config.model_dump(mode='json'),
                 train_config=module.train_config.model_dump(mode='json'),
                 model={k:v.detach().cpu().clone() for k,v in module.model.state_dict().items()})
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:torch.save(payload,f)
    os.replace(temp,path)


def load_experiment(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='marn_mechanism_inference_v1':raise ValueError('Wrong artifact format')
    mc=dict(payload['model_config']);mc.update(pretrain_weights=None,device='cpu')
    tc=dict(payload['train_config']);tc.update(accelerator='cpu',devices=1)
    model=ExperimentModule(RFDETRSmallConfig(**mc),TrainConfig(**tc),variant=payload['variant'],
                           class_ids=payload['marn_class_ids'],marn_config=MARNConfig(**payload['marn_config']))
    assert description(model)=={k:payload[k] for k in description(model)}
    model.model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval()
