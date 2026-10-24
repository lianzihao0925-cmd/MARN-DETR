"""Three short epochs prove last updates on a plateau; then full COCO identity eval."""
import os
os.environ.update(RF_HOME='/root/autodl-tmp/models/roboflow',
                  HF_HOME='/root/autodl-tmp/cache/huggingface',
                  TORCH_HOME='/root/autodl-tmp/cache/torch',
                  XDG_CACHE_HOME='/root/autodl-tmp/cache',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime
import torch
from pytorch_lightning import Callback,seed_everything
from rfdetr.config import TrainConfig,RFDETRSmallConfig
from rfdetr.training.trainer import build_trainer
from rfdetr.marn.data import MARNDataModule
from rfdetr.marn.integration import MARNConfig
from rfdetr.marn.lifecycle import MARNTrainingModule,load_inference,boundary
from formal_tools import configure_trainer,FullCOCOMetrics,atomic_json,export_atomic,AtomicCheckpoint

ROOT=Path('/root/autodl-tmp/project_marn-detr')
HERE=Path(__file__).resolve().parent
OUT=HERE/'check_outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
REPORT=dict(passed=False,stage='configuration',scope='engineering only; synthetic plateau scores are not research AP')


def config(output,epochs):
    values=json.loads((ROOT/'runs/baseline_10ep_seed42/resolved_train_config.json').read_text())
    values.update(output_dir=str(output),epochs=epochs,batch_size=8,grad_accum_steps=2,
                  eval_batch_size=8,num_workers=4,persistent_workers=False,amp_dtype='bf16',
                  use_ema=True,eval_base_model=True,eval_max_dets=100,num_sanity_val_steps=0,
                  compute_val_loss=False,run_test=False,save_dataset_grids=False,progress_bar=None,
                  tensorboard=False,wandb=False,mlflow=False,clearml=False)
    tc=TrainConfig(**values)
    mc=RFDETRSmallConfig(pretrain_weights=os.environ['RF_HOME']+'/rf-detr-small.pth',
                       num_classes=90,resolution=512,device='cuda',compile=False,cuda_graphs=False)
    return mc,tc


class Plateau(Callback):
    def on_validation_epoch_end(self,trainer,module):
        # Intentional monotonic decline verifies independent checkpoint tracks.
        module.log('engineering/regular',.6-.1*trainer.current_epoch,on_step=False,on_epoch=True)
        module.log('engineering/ema',.65-.1*trainer.current_epoch,on_step=False,on_epoch=True)


def plateau_monitors(trainer):
    for callback in trainer.callbacks:
        if isinstance(callback,AtomicCheckpoint) and callback.monitor is not None:
            callback.monitor='engineering/ema' if callback.filename=='best_ema' else 'engineering/regular'


class Audit(Callback):
    def __init__(self,folder,resume=False):self.folder=folder;self.resume=resume;self.rows=[]
    def on_train_start(self,trainer,module):
        if self.resume:
            assert trainer.current_epoch==3 and trainer.global_step==3
            assert int(module.adapter.committed_step)==3
            assert len(trainer.optimizers[0].state)>0
            print('C4_RESUME_START_OK epoch=3 global_step=3',flush=True)
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        if self.resume and not self.rows:assert trainer.current_epoch==3 and batch_idx==0
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        self.rows.append((trainer.global_step,batch_idx))
    def on_train_epoch_start(self,trainer,module):
        self.rows=[]
    def on_train_epoch_end(self,trainer,module):
        if self.resume:
            assert trainer.global_step==4 and len(self.rows)==2
        elif trainer.current_epoch==2:
            trainer.should_stop=True


def main():
    torch.set_num_threads(4);seed_everything(42,workers=True)
    mc,tc=config(OUT,4)
    dm=MARNDataModule(mc,tc);dm.setup('fit')
    ids=sorted(dm._dataset_train.coco.getCatIds())
    model=MARNTrainingModule(copy.deepcopy(mc),copy.deepcopy(tc),class_ids=ids,marn_config=MARNConfig(warmup_steps=1))
    trainer=build_trainer(tc,mc,limit_train_batches=2,limit_val_batches=2,enable_model_summary=False)
    ema,metric=configure_trainer(trainer,tc,OUT)
    plateau_monitors(trainer)
    # Consumers run after metric emission and before the native epoch checkpoint hooks.
    trainer.callbacks.insert(trainer.callbacks.index(metric)+1,Plateau())
    trainer.callbacks.insert(trainer.callbacks.index(metric)+2,Audit(OUT))
    REPORT['stage']='three_short_epochs'
    trainer.fit(model,datamodule=dm)
    positions={}
    for name in ('best_regular','best_ema','last'):
        path=OUT/(name+'.ckpt')
        c=torch.load(path,map_location='cpu',weights_only=False)
        positions[name]=dict(epoch=c['epoch'],global_step=c['global_step'],bytes=path.stat().st_size)
        assert c['optimizer_states'] and c['lr_schedulers'] and c['marn_main_process_rng_v1']
        assert 'MARNEMACallback:online_bank_exact_copy_v1' in c['callbacks']
        if name=='last':assert c['epoch']==2 and c['global_step']==3
        else:assert c['epoch']==0 and c['global_step']==1
        del c
    REPORT['plateau_checkpoint_positions']=positions
    export_atomic(model,OUT/'last_regular_test.pth','regular')
    export_atomic(ema._average_model.module,OUT/'last_ema_test.pth','ema')
    for name,expected in [('regular',model),('ema',ema._average_model.module)]:
        restored=load_inference(OUT/('last_'+name+'_test.pth'))
        for key,value in expected.model.state_dict().items():
            torch.testing.assert_close(value.cpu(),restored.model.state_dict()[key],rtol=0,atol=0)
        del restored
    REPORT['inference_exports']='strict exact states passed'
    del trainer,model,ema;gc.collect();torch.cuda.empty_cache()
    REPORT['stage']='resume_native_epoch_boundary'
    mc,tc=config(OUT,4)
    model=MARNTrainingModule(copy.deepcopy(mc),copy.deepcopy(tc),class_ids=ids,marn_config=MARNConfig(warmup_steps=1))
    trainer=build_trainer(tc,mc,limit_train_batches=2,limit_val_batches=2,enable_model_summary=False)
    ema,metric=configure_trainer(trainer,tc,OUT)
    plateau_monitors(trainer)
    trainer.callbacks.insert(trainer.callbacks.index(metric)+1,Plateau())
    trainer.callbacks.insert(trainer.callbacks.index(metric)+1,Audit(OUT,resume=True))
    trainer.fit(model,datamodule=dm,ckpt_path=str(OUT/'last.ckpt'))
    assert trainer.global_step==4 and int(model.adapter.committed_step)==4
    REPORT['resume']='epoch 3 batch 0, two microbatches, global step 4 passed'
    del trainer,model,ema,dm;gc.collect();torch.cuda.empty_cache()
    REPORT['stage']='full_coco_zero_output_identity_evaluation'
    evalfolder=OUT/'initial_full_coco';evalfolder.mkdir()
    mc,tc=config(evalfolder,1)
    tc.use_ema=False
    dm=MARNDataModule(mc,tc);dm.setup('validate')
    model=MARNTrainingModule(copy.deepcopy(mc),copy.deepcopy(tc),class_ids=ids,marn_config=MARNConfig(warmup_steps=1))
    frozen={k:v.clone() for k,v in model.adapter.state_dict().items()}
    trainer=build_trainer(tc,mc,include_training_callbacks=False,enable_model_summary=False)
    metric=FullCOCOMetrics(evalfolder,max_dets=100,eval_base_model=True,
                           log_per_class_metrics=False,eval_backend=tc.eval_backend)
    trainer.callbacks=[c for c in trainer.callbacks if not isinstance(c,FullCOCOMetrics) and c.__class__.__name__!='COCOEvalCallback']+[metric]
    started=time.monotonic()
    trainer.validate(model,datamodule=dm,verbose=False)
    assert len(dm._dataset_val)==5000
    for key,value in model.adapter.state_dict().items():
        torch.testing.assert_close(value.cpu(),frozen[key].cpu(),rtol=0,atol=0)
    values=metric.captured['regular']
    assert abs(values['AP']-.5292240381240845)<.0005,values
    REPORT['full_coco']=dict(images=5000,metrics=values,frozen_bank=True,
                             seconds=time.monotonic()-started,identity_initialization=True)
    REPORT.update(passed=True,stage='complete')


if __name__=='__main__':
    OUT.mkdir(parents=True)
    started=time.monotonic()
    try:main()
    except BaseException as exc:
        import traceback
        REPORT['error']=repr(exc);traceback.print_exc()
    REPORT['total_seconds']=time.monotonic()-started
    atomic_json(OUT/'c4_result.json',REPORT)
    print('C4_RESULT',json.dumps(REPORT),flush=True)
    sys.exit(0 if REPORT['passed'] else 1)
