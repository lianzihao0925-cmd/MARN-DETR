"""Differentiable query retrieval supervision; detached bank and metadata."""
import sys
sys.path.insert(0,'/root/autodl-tmp/project_marn-detr/experiments/marn_c4')
import torch
from torch.nn import functional as F
from query_only import ExperimentModule
from rfdetr.marn.integration import MARNAdapter,paired_iou

class RetrievalAdapter(MARNAdapter):
    def forward(self,raw,*args,**kwargs):
        if self.enabled and self.training:
            assert getattr(self,'retrieval_query',None) is None
            self.retrieval_query=raw
        return super().forward(raw,*args,**kwargs)

    def observe(self,outputs,all_indices,suffixes,targets):
        if not self.enabled or not self.training:return
        raw=self.retrieval_query
        assert raw is not None
        self.retrieval_query=None
        # Bank writer keeps its original no-grad and optimizer-window lifecycle.
        super().observe(outputs,all_indices,suffixes,targets)
        layer=suffixes.index(f'_{self.config.layer}');feats=[];labels=[]
        for b,((src,dst),target) in enumerate(zip(all_indices[layer],targets)):
            with torch.no_grad():
                src,dst=src.to(raw.device),dst.to(raw.device)
                if 'valid' in target:
                    use=target['valid'][dst].bool();src,dst=src[use],dst[use]
                # Use the same first-group supervision scope as the previous pilot.
                use=src<self.num_queries;src,dst=src[use],dst[use]
                if not len(src):continue
                cats=target['labels'][dst];pred=outputs[layer]
                conf=pred['pred_logits'][b,src,cats].float().sigmoid()
                iou=paired_iou(pred['pred_boxes'][b,src].float(),target['boxes'][dst].float())
                use=(conf>=self.config.confidence_min)&(iou>=self.config.iou_min)&torch.isfinite(raw[b,src]).all(-1)
            feats.append(raw[b,src[use]]);labels.append(cats[use])
        snap=self.snapshot;valid=snap.valid&torch.isfinite(snap.features).all(-1)&(snap.quality>0)
        self.retrieval_loss=self.reader.key.weight.sum()*0+raw.sum()*0;self.retrieval_count=0
        if not feats or not bool(valid.any()):return
        q=torch.cat(feats);cats=torch.cat(labels);mem=snap.features[valid].detach();mcats=snap.classes[valid]
        positive=cats[:,None]==mcats[None];use=positive.any(-1)&(~positive).any(-1)
        if not bool(use.any()):return
        keys=F.normalize(self.reader.key(q[use]).float(),dim=-1,eps=1e-6)
        mkeys=F.normalize(self.reader.key(mem).float(),dim=-1,eps=1e-6)
        scores=keys@mkeys.T/self.reader.config.temperature
        self.retrieval_loss=(scores.logsumexp(-1)-scores.masked_fill(~positive[use],-torch.inf).logsumexp(-1)).mean()
        self.retrieval_count=int(use.sum())

class RetrievalExperiment(ExperimentModule):
    def __init__(self,*args,retrieval_weight=0.05,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.variant=='marn'
        self.retrieval_weight=float(retrieval_weight)
        self.adapter.__class__=RetrievalAdapter
        self.adapter.retrieval_query=None;self.adapter.retrieval_loss=None;self.adapter.retrieval_count=0
        self.criterion.marn_observer=self.adapter.observe
        self.supervision_examples=0

    def training_step(self,batch,batch_idx):
        self.adapter.retrieval_loss=None
        result=super().training_step(batch,batch_idx)
        aux=self.adapter.retrieval_loss
        assert aux is not None and aux.requires_grad and torch.isfinite(aux).all()
        self.supervision_examples+=self.adapter.retrieval_count
        if isinstance(result,dict):result['loss']=result['loss']+self.retrieval_weight*aux
        else:result=result+self.retrieval_weight*aux
        self.log('train/retrieval_loss',aux.detach(),on_step=False,on_epoch=True,batch_size=self.train_config.batch_size)
        self.adapter.retrieval_loss=None
        return result

    def on_save_checkpoint(self,checkpoint):
        assert self.adapter.retrieval_query is None and self.adapter.retrieval_loss is None
        super().on_save_checkpoint(checkpoint)
        checkpoint['marn_retrieval_weight']=self.retrieval_weight

    def on_load_checkpoint(self,checkpoint):
        assert checkpoint.get('marn_retrieval_weight')==self.retrieval_weight
        super().on_load_checkpoint(checkpoint)
