"""One formally configured mechanism arm; parent controller supplies explicit budget."""
import os
os.environ.update(RF_HOME='/root/autodl-tmp/models/roboflow',HF_HOME='/root/autodl-tmp/cache/huggingface',
                  TORCH_HOME='/root/autodl-tmp/cache/torch',XDG_CACHE_HOME='/root/autodl-tmp/cache',
                  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from pytorch_lightning import Callback,seed_everything
from rfdetr.config import RFDETRSmallConfig,TrainConfig
from rfdetr.training.trainer import build_trainer
from rfdetr.utilities.tensors import PackedTargets
from rfdetr.marn.data import MARNDataModule
from rfdetr.marn.integration import MARNConfig
from rfdetr.marn.lifecycle import POLICY
from formal_tools import configure_trainer,atomic_json
from query_only import ExperimentModule,export_experiment,load_experiment

ROOT=Path('/root/autodl-tmp/project_marn-detr')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['marn','query_only'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,required=True)
    args=parser.parse_args()
    assert 1<=args.epochs<=10
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);seed_everything(42,workers=True)
    saved=ROOT/'runs/baseline_10ep_seed42/resolved_train_config.json'
    values=json.loads(saved.read_text())
    # Preserve the original 10-epoch schedule, even for the shorter screen.
    # Trainer max_epochs=10 drives estimated_stepping_batches and managed LR.
    values.update(output_dir=str(out),epochs=10,seed=42,batch_size=16,grad_accum_steps=1,
                  eval_batch_size=8,num_workers=8,persistent_workers=True,prefetch_factor=2,
                  pin_memory=True,amp_dtype='bf16',use_ema=True,eval_base_model=True,
                  eval_max_dets=100,compute_val_loss=False,num_sanity_val_steps=0,
                  run_test=False,save_dataset_grids=False,progress_bar=None,
                  tensorboard=False,wandb=False,mlflow=False,clearml=False)
    tc=TrainConfig(**values)
    assert tc.eval_interval==1 and tc.dataset_file=='coco'
    assert getattr(tc.multi_scale,'value',tc.multi_scale)=='off'
    weights=Path(os.environ['RF_HOME'])/'rf-detr-small.pth'
    assert hashlib.md5(weights.read_bytes()).hexdigest()=='fb37061c1af7bace359c91b723a8d5c1'
    mc=RFDETRSmallConfig(pretrain_weights=str(weights),num_classes=90,resolution=512,
                       device='cuda',compile=False,cuda_graphs=False,gradient_checkpointing=False)
    cfg=MARNConfig(warmup_steps=500)
    class PairedData(MARNDataModule):
        def reset_trace(self):
            self.target_hash=hashlib.sha256();self.pixel_hash=hashlib.sha256();self.batches_seen=0
        def transfer_batch_to_device(self,batch,device,dataloader_idx):
            if self.trainer.training:
                samples,targets=batch
                entries=[targets.fields] if isinstance(targets,PackedTargets) else targets
                for target in entries:
                    for name in sorted(target):
                        value=target[name];assert value.device.type=='cpu'
                        self.target_hash.update(name.encode())
                        self.target_hash.update(str((value.dtype,tuple(value.shape))).encode())
                        self.target_hash.update(value.contiguous().numpy().tobytes())
                if self.batches_seen<2:self.pixel_hash.update(samples.tensors.contiguous().numpy().tobytes())
                self.batches_seen+=1
            return super().transfer_batch_to_device(batch,device,dataloader_idx)
    dm=PairedData(mc,tc);dm.reset_trace();dm.setup('fit')
    assert len(dm._dataset_train)==118287 and len(dm._dataset_val)==5000
    ids=sorted(dm._dataset_train.coco.getCatIds())
    model=ExperimentModule(copy.deepcopy(mc),copy.deepcopy(tc),variant=args.variant,
                           class_ids=ids,marn_config=cfg)
    protocol=dict(variant=args.variant,seed=42,screening_epochs=args.epochs,schedule_horizon_epochs=10,
                  train=tc.model_dump(mode='json'),model=mc.model_dump(mode='json'),marn=cfg.__dict__,
                  ema_policy=POLICY,initial_weights_md5=hashlib.md5(weights.read_bytes()).hexdigest(),
                  baseline_config_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  adapter_parameters=sum(p.numel() for p in model.adapter.parameters()),
                  bank_bytes=sum(t.numel()*t.element_size() for t in model.adapter.bank.buffers()),
                  interpretation='single paired seed, bounded screen; not sufficient convergence or multi-seed proof')
    atomic_json(out/'protocol.json',protocol)
    atomic_json(out/'resolved_train_config.json',tc.model_dump(mode='json'))

    class StopAtEpoch(Callback):
        def on_train_epoch_start(self,trainer,module):dm.reset_trace()
        def on_train_epoch_end(self,trainer,module):
            assert dm.batches_seen==7393
            atomic_json(out/('input_trace_epoch_%03d.json'%(trainer.current_epoch+1)),dict(
                batches=dm.batches_seen,target_sha256=dm.target_hash.hexdigest(),
                first_two_images_sha256=dm.pixel_hash.hexdigest()))
            if trainer.current_epoch+1>=args.epochs:trainer.should_stop=True

    trainer=build_trainer(tc,mc,enable_model_summary=False)
    ema,metrics=configure_trainer(trainer,tc,out)
    trainer.callbacks.insert(0,StopAtEpoch())
    started=time.monotonic()
    trainer.fit(model,datamodule=dm)
    expected=args.epochs*7393
    assert trainer.global_step==expected and int(model.adapter.committed_step)==expected
    audit=json.loads((out/'epoch_audit.json').read_text())
    assert len(audit)==args.epochs and all(r['batches']==7393 and r['all_losses_finite'] for r in audit)
    export_experiment(model,out/'last_regular.pth','regular')
    export_experiment(ema._average_model.module,out/'last_ema.pth','ema')
    del trainer,model,ema;gc.collect();torch.cuda.empty_cache()
    # Export selected weights from their own full checkpoint, never the last model.
    selections={}
    for source in ('regular','ema'):
        checkpoint=out/('best_'+source+'.ckpt')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        chosen=ExperimentModule(copy.deepcopy(mc).model_copy(update={'pretrain_weights':None,'device':'cpu'}),
                                copy.deepcopy(tc),variant=args.variant,class_ids=ids,marn_config=cfg)
        state=payload['state_dict'] if source=='regular' else {
            k.removeprefix('module.'):v for k,v in payload['callbacks'][
                'MARNEMACallback:online_bank_exact_copy_v1']['average_model_state_dict'].items()
            if k.startswith('module.')}
        chosen.load_state_dict(state,strict=True)
        export_experiment(chosen,out/('best_'+source+'.pth'),source)
        selections[source]=dict(epoch=payload['epoch']+1,global_step=payload['global_step'])
        restored=load_experiment(out/('best_'+source+'.pth'))
        for key,value in chosen.model.state_dict().items():
            torch.testing.assert_close(value,restored.model.state_dict()[key],rtol=0,atol=0)
        del chosen,restored,payload;gc.collect()
    atomic_json(out/'completed.json',dict(passed=True,variant=args.variant,epochs=args.epochs,
                global_step=expected,wall_seconds=time.monotonic()-started,best=selections))
    print('MECHANISM_ARM_COMPLETED',args.variant,flush=True)


if __name__=='__main__':main()
