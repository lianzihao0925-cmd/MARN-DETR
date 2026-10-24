"""Experiment-level callbacks; installed RF-DETR and MARN source stays unchanged."""
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.marn.lifecycle import MARNEMACallback, boundary, export_inference


def atomic_json(path, value):
    path=Path(path)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)


class FullCOCOMetrics(COCOEvalCallback):
    """Capture native evaluator AP including area splits, without a second compute."""
    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self.output=Path(output)
        self.captured={}

    def _compute_map_metric(self, trainer, metric):
        result=super()._compute_map_metric(trainer,metric)
        source='ema' if metric is self.map_metric_ema else 'regular'
        names={'map':'AP','map_50':'AP50','map_75':'AP75','map_small':'APs',
               'map_medium':'APm','map_large':'APl','mar_100':'AR100'}
        values={out:float(result[key]) for key,out in names.items() if key in result}
        assert len(values)==len(names), (source,list(result))
        assert all(math.isfinite(v) for v in values.values()),values
        self.captured[source]=values
        for key in ('APs','APm','APl'):
            trainer.lightning_module.log(f'val/{source}_{key}',values[key],on_step=False,on_epoch=True)
        atomic_json(self.output/('coco_epoch_%03d_%s.json'%(trainer.current_epoch+1,source)),dict(
            epoch=trainer.current_epoch+1,global_step=trainer.global_step,source=source,
            metrics=values,max_dets=100,backend=self._eval_backend))
        return result


class RNGState(Callback):
    """Capture main-process RNG; do not claim exact persistent-worker replay."""
    def on_save_checkpoint(self, trainer, module, checkpoint):
        state=np.random.get_state()
        checkpoint['marn_main_process_rng_v1']=dict(
            python=random.getstate(),numpy_name=state[0],numpy_keys=state[1].tolist(),
            numpy_position=state[2],numpy_has_gauss=state[3],numpy_cached=state[4],
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())

    def on_load_checkpoint(self, trainer, module, checkpoint):
        self.pending=checkpoint.get('marn_main_process_rng_v1')
        if self.pending is None: raise ValueError('Main-process RNG state missing')

    def on_train_start(self, trainer, module):
        state=getattr(self,'pending',None)
        if state:
            random.setstate(state['python'])
            np.random.set_state((state['numpy_name'],np.asarray(state['numpy_keys'],dtype=np.uint32),
                                 state['numpy_position'],state['numpy_has_gauss'],state['numpy_cached']))
            torch.set_rng_state(state['torch_cpu'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])
            self.pending=None


class AtomicCheckpoint(ModelCheckpoint):
    """Native epoch-boundary loop state with local atomic file replacement."""
    def _save_checkpoint(self, trainer, filepath):
        boundary(trainer.lightning_module.adapter)
        path=Path(filepath)
        temp=path.with_name(path.name+'.tmp')
        if temp.exists(): raise FileExistsError(temp)
        # Keep native bookkeeping/notification with the final destination name.
        trainer.save_checkpoint(str(temp),self.save_weights_only)
        os.replace(temp,path)
        self._last_global_step_saved=trainer.global_step
        self._last_checkpoint_saved=filepath
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(__import__('weakref').proxy(self))
        if path.stem=='last':
            for callback in trainer.callbacks:
                if isinstance(callback,EpochAudit):
                    callback.records[-1]['through_last_checkpoint_seconds']=time.monotonic()-callback.started
                    atomic_json(callback.output/'epoch_audit.json',callback.records)


class EpochAudit(Callback):
    def __init__(self, output):
        self.output=Path(output)
        self.records=[]

    def state_dict(self):return {'records':self.records}

    def load_state_dict(self,state_dict):self.records=state_dict['records']

    def on_train_epoch_start(self, trainer, module):
        self.started=time.monotonic()
        self.batches=0
        self.finite=None
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.batches+=1
        value=torch.isfinite(outputs['loss'].detach()).all()
        self.finite=value if self.finite is None else self.finite & value
        if self.batches%1000==0:
            print('TRAIN_PROGRESS',json.dumps(dict(epoch=trainer.current_epoch+1,batches=self.batches,
                  global_step=trainer.global_step)),flush=True)

    def on_train_epoch_end(self, trainer, module):
        assert self.finite is not None and bool(self.finite)
        boundary(module.adapter)
        record=dict(epoch=trainer.current_epoch+1,global_step=trainer.global_step,
                    batches=self.batches,all_losses_finite=True,
                    before_checkpoint_seconds=time.monotonic()-self.started,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
                    learning_rates=[float(g['lr']) for g in trainer.optimizers[0].param_groups],
                    bank=module.adapter.last_commit)
        self.records.append(record)
        atomic_json(self.output/'epoch_audit.json',self.records)
        print('EPOCH_AUDIT',json.dumps(record),flush=True)


def configure_trainer(trainer, tc, output):
    """Replace only stock EMA/checkpoint/metric callbacks; keep native training stack."""
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    callbacks=[cb for cb in trainer.callbacks if not isinstance(cb,(ModelCheckpoint,RFDETREMACallback,COCOEvalCallback))]
    # EMA precedes audit/checkpoints. Metrics run before checkpoint consumers.
    ema=MARNEMACallback(decay=tc.ema_decay,tau=tc.ema_tau)
    metrics=FullCOCOMetrics(folder,max_dets=100,segmentation=False,eval_interval=1,
                            log_per_class_metrics=False,eval_base_model=True,eval_backend=tc.eval_backend)
    callbacks.extend([ema,metrics,RNGState(),EpochAudit(folder)])
    for name,monitor in [('best_regular','val/mAP_50_95'),('best_ema','val/ema_mAP_50_95'),('last',None)]:
        callbacks.append(AtomicCheckpoint(dirpath=str(folder),filename=name,monitor=monitor,
                         mode='max',save_top_k=1,every_n_epochs=1,save_on_train_epoch_end=True,
                         auto_insert_metric_name=False,enable_version_counter=False))
    trainer.callbacks=callbacks
    return ema,metrics


def export_atomic(module,path,source):
    path=Path(path)
    temp=path.with_name(path.name+'.new')
    if temp.exists():raise FileExistsError(temp)
    export_inference(module,temp,source=source)
    os.replace(temp,path)
