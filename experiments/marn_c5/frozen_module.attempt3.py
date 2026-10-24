"""Train adapters only; optional train-matched prototype classification loss."""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,'/root/autodl-tmp/project_marn-detr/experiments/marn_c4')
from query_only import ExperimentModule
from rfdetr.marn.integration import MARNAdapter,paired_iou

class SupervisedAdapter(MARNAdapter):
    @torch.no_grad()
    def observe(self,outputs,all_indices,suffixes,targets):
        raw=self.capture
        super().observe(outputs,all_indices,suffixes,targets)
        layer=suffixes.index(f'_{self.config.layer}')
        feats=[];labels=[]
        for b,((src,dst),target) in enumerate(zip(all_indices[layer],targets)):
            keep=src<self.num_queries;src,dst=src[keep],dst[keep]
            if len(src)==0:continue
            cats=target['labels'][dst]
            pred=outputs[layer]
            conf=pred['pred_logits'][b,src,cats].float().sigmoid()
            iou=paired_iou(pred['pred_boxes'][b,src].float(),target['boxes'][dst].float())
            good=(conf>=self.config.confidence_min)&(iou>=self.config.iou_min)
            feats.append(raw[b,src[good]].detach());labels.append(cats[good])
        snap=self.snapshot
        valid=snap.valid&torch.isfinite(snap.features).all(-1)&(snap.quality>0)
        with torch.enable_grad():
            zero=self.reader.key.weight.sum()*0
            self.retrieval_loss=zero;self.retrieval_count=0
            if not feats or not bool(valid.any()):return
            q=torch.cat(feats);cats=torch.cat(labels)
            mem=snap.features[valid].detach();mcats=snap.classes[valid]
            positive=cats[:,None]==mcats[None]
            use=positive.any(-1)&(~positive).any(-1)
            if not bool(use.any()):return
            q=q[use];positive=positive[use]
            keys=F.normalize(self.reader.key(q).float(),dim=-1,eps=1e-6)
            mkeys=F.normalize(self.reader.key(mem).float(),dim=-1,eps=1e-6)
            scores=keys@mkeys.T/self.reader.config.temperature
            self.retrieval_loss=(scores.logsumexp(-1)-scores.masked_fill(~positive,-torch.inf).logsumexp(-1)).mean()
            self.retrieval_count=len(q)

class FrozenExperiment(ExperimentModule):
    def __init__(self,*args,kind='marn',**kwargs):
        self.kind=kind
        super().__init__(*args,**kwargs)
        self.supervision_examples=0
        for p in self.parameters():p.requires_grad_(False)
        for p in self.adapter.parameters():p.requires_grad_(True)
        if kind=='supervised':
            self.adapter.__class__=SupervisedAdapter
            self.adapter.retrieval_loss=None
            self.adapter.retrieval_count=0
            self.criterion.marn_observer=self.adapter.observe

    def training_step(self,batch,batch_idx):
        if self.kind=='supervised':self.adapter.retrieval_loss=None
        result=super().training_step(batch,batch_idx)
        # During bank warmup the frozen detector can bypass every adapter.
        # Keep backward valid without changing any detector parameter.
        loss=result['loss'] if isinstance(result,dict) else result
        if not loss.requires_grad:
            loss=loss+0.0*self.adapter.refiner.out.weight.sum()
            if isinstance(result,dict):result['loss']=loss
            else:result=loss
        if self.kind=='supervised':
            aux=self.adapter.retrieval_loss
            assert aux is not None and torch.isfinite(aux).all()
            assert aux.requires_grad
            self.supervision_examples+=self.adapter.retrieval_count
            assert not self._use_manual_optimization
            if isinstance(result,dict):result['loss']=result['loss']+.05*aux
            else:result=result+.05*aux
            self.log('train/retrieval_loss',aux.detach(),on_step=False,on_epoch=True,batch_size=self.train_config.batch_size)
            self.adapter.retrieval_loss=None
        return result
