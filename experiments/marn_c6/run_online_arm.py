"""One controlled full-detector pilot arm, including native EMA/checkpoints."""
import os
os.environ.update(RF_HOME='/root/autodl-tmp/models/roboflow',HF_HOME='/root/autodl-tmp/cache/huggingface',TORCH_HOME='/root/autodl-tmp/cache/torch',XDG_CACHE_HOME='/root/autodl-tmp/cache',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
import sys,argparse,copy,gc,hashlib,json,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/project_marn-detr');sys.path.insert(0,str(ROOT/'experiments/marn_c4'))
import torch
from pytorch_lightning import Callback,seed_everything
from rfdetr.config import RFDETRSmallConfig,TrainConfig
from rfdetr.training.trainer import build_trainer
from rfdetr.utilities.tensors import PackedTargets
from rfdetr.marn.data import MARNDataModule
from rfdetr.marn.integration import MARNConfig
from formal_tools import configure_trainer,atomic_json
from online_module import ControlExperiment,OnlineExperiment,export_c6,load_c6

def tensor_hash(items):
    h=hashlib.sha256()
    for k,v in items:h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['query_only','random','online'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    torch.set_num_threads(4);seed_everything(a.seed,workers=True);out=a.output;out.mkdir(parents=True,exist_ok=False)
    values=json.loads((ROOT/'runs/baseline_10ep_seed42/resolved_train_config.json').read_text())
    values.update(output_dir=str(out),epochs=10,seed=a.seed,batch_size=16,grad_accum_steps=1,eval_batch_size=8,num_workers=8,persistent_workers=True,prefetch_factor=2,pin_memory=True,amp_dtype='bf16',use_ema=True,eval_base_model=True,eval_max_dets=100,compute_val_loss=False,num_sanity_val_steps=0,run_test=False,save_dataset_grids=False,progress_bar=None,tensorboard=False,wandb=False,mlflow=False,clearml=False,lr=1e-5,lr_encoder=1.5e-5,warmup_epochs=.5)
    tc=TrainConfig(**values);assert tc.eval_interval==1 and getattr(tc.multi_scale,'value',tc.multi_scale)=='off'
    weights=Path(os.environ['RF_HOME'])/'rf-detr-small.pth';assert hashlib.md5(weights.read_bytes()).hexdigest()=='fb37061c1af7bace359c91b723a8d5c1'
    mc=RFDETRSmallConfig(pretrain_weights=str(weights),num_classes=90,resolution=512,device='cuda',compile=False,cuda_graphs=False,gradient_checkpointing=False)
    class PairedData(MARNDataModule):
        def reset(self):self.target_hash=hashlib.sha256();self.pixel_hash=hashlib.sha256();self.batches=0
        def transfer_batch_to_device(self,batch,device,dataloader_idx):
            if self.trainer.training:
                samples,targets=batch
                for t in ([targets.fields] if isinstance(targets,PackedTargets) else targets):
                    for k in sorted(t):
                        v=t[k];assert v.device.type=='cpu';self.target_hash.update(k.encode());self.target_hash.update(str((v.dtype,tuple(v.shape))).encode());self.target_hash.update(v.contiguous().numpy().tobytes())
                if self.batches<2:self.pixel_hash.update(samples.tensors.contiguous().numpy().tobytes())
                self.batches+=1
            return super().transfer_batch_to_device(batch,device,dataloader_idx)
    dm=PairedData(mc,tc);dm.reset();dm.setup('fit');assert len(dm._dataset_train)==118287 and len(dm._dataset_val)==5000
    ids=sorted(dm._dataset_train.coco.getCatIds());cfg=MARNConfig(warmup_steps=1 if a.smoke else 500)
    cls=ControlExperiment if a.kind=='query_only' else OnlineExperiment
    extra={} if a.kind=='query_only' else dict(route_kind=a.kind,probe_interval=1 if a.smoke else 16,route_start=2 if a.smoke else 1000,retrieval_weight=.05)
    model=cls(copy.deepcopy(mc),copy.deepcopy(tc),variant='query_only' if a.kind=='query_only' else 'marn',class_ids=ids,marn_config=cfg,**extra)
    initial=tensor_hash(model.named_parameters());expected=6 if a.smoke else 7393
    protocol=dict(kind=a.kind,seed=a.seed,screening_epochs=1,horizon_epochs=10,expected_updates=expected,initial_parameter_sha256=initial,initial_weights_md5=hashlib.md5(weights.read_bytes()).hexdigest(),train=tc.model_dump(mode='json'),model=mc.model_dump(mode='json'),marn=cfg.__dict__,extras=extra,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),lr_adapter=1e-4,lr_router=1e-3,label_scale=.001,label_clip=10.,route_weight=.1,interpretation='single-seed short pilot; full model trainable, native loss and schedule; retrieval supervision plus adapter LR differs from earlier pilots; router scaffold in all arms, inactive in controls; not SOTA/convergence evidence')
    atomic_json(out/'protocol.json',protocol)
    class Guard(Callback):
        def __init__(self):self.key=False;self.out=False;self.router=False
        def on_train_epoch_start(self,trainer,module):dm.reset()
        def on_before_optimizer_step(self,trainer,module,optimizer):
            assert torch.stack([torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None]).all()
            for name,param in [('key',module.adapter.reader.key.weight),('out',module.adapter.refiner.out.weight),('router',module.adapter.router.network[-1].weight)]:
                if param.grad is not None and bool(param.grad.abs().max()>0):setattr(self,name,True)
        def on_train_epoch_end(self,trainer,module):
            assert dm.batches==expected
            atomic_json(out/'input_trace.json',dict(batches=dm.batches,target_sha256=dm.target_hash.hexdigest(),first_two_images_sha256=dm.pixel_hash.hexdigest()))
            trainer.should_stop=True
    trainer=build_trainer(tc,mc,enable_model_summary=False,**(dict(limit_train_batches=6,limit_val_batches=2) if a.smoke else {}));ema,metrics=configure_trainer(trainer,tc,out)
    guard=Guard();trainer.callbacks.insert(0,guard);started=time.monotonic();trainer.fit(model,datamodule=dm)
    assert trainer.global_step==expected and int(model.adapter.committed_step)==expected
    assert tensor_hash(model.named_parameters())!=initial and guard.out
    if a.kind!='query_only':assert guard.key and model.supervision_examples>0
    if a.kind=='online':assert guard.router and model.probe_examples>0 and model.null_verified
    # Full native checkpoint must reconstruct the exact trained parameters.
    payload=torch.load(out/'last.ckpt',map_location='cpu',weights_only=False)
    chosen=cls(copy.deepcopy(mc).model_copy(update={'pretrain_weights':None,'device':'cpu'}),copy.deepcopy(tc),variant='query_only' if a.kind=='query_only' else 'marn',class_ids=ids,marn_config=cfg,**extra)
    assert payload['marn_config']==cfg.__dict__ and payload['marn_class_ids']==ids
    chosen.load_state_dict(payload['state_dict'],strict=True)
    assert tensor_hash(chosen.named_parameters())==tensor_hash(model.named_parameters())
    export_c6(model,out/'last_regular.pth','regular');export_c6(ema._average_model.module,out/'last_ema.pth','ema')
    restored=load_c6(out/'last_regular.pth');assert tensor_hash(restored.named_parameters())==tensor_hash(model.named_parameters())
    report=dict(passed=True,kind=a.kind,seed=a.seed,global_step=trainer.global_step,wall_seconds=time.monotonic()-started,supervision_examples=model.supervision_examples,probe_examples=model.probe_examples,probe_batches=model.probe_batches,probe_seconds=model.probe_seconds,null_verified=model.null_verified,key_gradient=guard.key,out_gradient=guard.out,router_gradient=guard.router,native_checkpoint_strict=True,inference_export_strict=True,regular=metrics.captured['regular'],ema=metrics.captured['ema'])
    if a.kind=='online':report.update(probe_positive=model.probe_positive,probe_abs_sum=model.probe_abs_sum,route_images=model.adapter.route_images,route_selected_sum=model.adapter.route_selected_sum)
    atomic_json(out/'completed.json',report);print('ONLINE_ARM_DONE',json.dumps(report),flush=True)

if __name__=='__main__':main()
