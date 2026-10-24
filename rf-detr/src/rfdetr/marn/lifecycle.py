"""MARN eager/single-GPU EMA and self-describing inference artifacts.

The EMA evaluator uses the online training bank, copied exactly after commit.
It does not generate its own features or average bank slots across replacements.
"""
from dataclasses import asdict
from pathlib import Path
import os

import torch

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.callbacks.ema import RFDETREMACallback
from .integration import MARNModelModule, MARNConfig, attach

POLICY = 'online_bank_exact_copy_v1'
FORMAT = 'marn_small_inference_v1'


def boundary(adapter):
    if adapter.snapshot is not None or adapter.capture is not None or adapter.pending or adapter.observed:
        raise RuntimeError('MARN state must be at a completed optimizer/bank boundary')
    if not adapter.enabled:
        raise RuntimeError('Do not serialize or average a disabled MARN adapter')


def description(module):
    a = module.adapter
    return dict(marn_ema_policy=POLICY, marn_config=asdict(a.config),
                marn_class_ids=a.reader.class_ids.cpu().tolist(),
                marn_reader_config=asdict(a.reader.config),
                marn_refiner_config=dict(rho=a.refiner.rho, beta_max=a.refiner.beta_max,
                                         bottleneck=a.refiner.fuse.out_features))


class MARNTrainingModule(MARNModelModule):
    _marn_ema_policy = POLICY

    def __init__(self, model_config, train_config, *, class_ids, marn_config=None):
        # Skip upstream encoder downloads: initialization is the local complete
        # RF-DETR checkpoint, or a full strict state load immediately afterward.
        RFDETRModelModule.__init__(self, model_config, train_config, load_encoder_weights=False)
        # New CPU-initialized parameters must not advance the detector's RNG.
        with torch.random.fork_rng(devices=[]):
            attach(self, class_ids, marn_config)
        self.strict_loading = True

    @property
    def _use_fused_adamw_ema(self):
        # EMA must run after the bank commit, not inside the optimizer kernel.
        return False

    def on_fit_start(self):
        callbacks = [c for c in self.trainer.callbacks if isinstance(c, RFDETREMACallback)]
        if self.train_config.use_ema:
            if len(callbacks) != 1 or not isinstance(callbacks[0], MARNEMACallback):
                raise RuntimeError('MARN use_ema=True requires exactly one MARNEMACallback')
        elif callbacks:
            raise RuntimeError('EMA callback conflicts with use_ema=False')
        super().on_fit_start()

    def on_save_checkpoint(self, checkpoint):
        boundary(self.adapter)
        super().on_save_checkpoint(checkpoint)
        checkpoint.update(description(self))

    def on_load_checkpoint(self, checkpoint):
        for key, value in description(self).items():
            if checkpoint.get(key) != value:
                raise ValueError(f'MARN checkpoint metadata mismatch: {key}')
        super().on_load_checkpoint(checkpoint)


class MARNEMACallback(RFDETREMACallback):
    """Native EMA arithmetic, except MARN buffers are copied as discrete state."""

    @property
    def state_key(self):
        return 'MARNEMACallback:' + POLICY

    @staticmethod
    def _storage_key(tensor):
        return (str(tensor.device), tensor.data_ptr())

    def on_fit_start(self, trainer, pl_module):
        boundary(pl_module.adapter)
        if not isinstance(pl_module, MARNTrainingModule) or not pl_module.train_config.use_ema:
            raise RuntimeError('MARNEMACallback requires MARNTrainingModule with use_ema=True')
        if trainer.world_size != 1 or str(trainer.precision) not in ('bf16-mixed', '32-true'):
            raise RuntimeError('MARN EMA supports single-GPU BF16/FP32 only')
        if not self._use_buffers or self._update_interval_steps != 1:
            raise ValueError('MARN EMA requires use_buffers=True and update_interval_steps=1')
        if getattr(trainer.optimizers[0], '_fuses_ema', False):
            raise RuntimeError('Fused optimizer/EMA bypasses the bank boundary; unsupported')
        super().on_fit_start(trainer, pl_module)
        # AveragedModel passes detached views: key by storage, not Python id.
        self._copy_buffer_keys = {
            self._storage_key(t) for t in self._average_model.module.adapter.buffers()
        }
        if not self._copy_buffer_keys:
            raise RuntimeError('No MARN EMA buffers found')

    @torch.no_grad()
    def _multi_avg_fn(self, averaged_params, model_params, num_averaged):
        averaged_learned, live_learned = [], []
        for average, live in zip(averaged_params, model_params):
            if self._storage_key(average) in self._copy_buffer_keys:
                average.copy_(live)
            else:
                averaged_learned.append(average)
                live_learned.append(live)
        super()._multi_avg_fn(averaged_learned, live_learned, num_averaged)

    def _update_ema_for_step(self, pl_module, global_step):
        if global_step > self._latest_update_step:
            boundary(pl_module.adapter)
            if int(pl_module.adapter.committed_step) != global_step:
                raise RuntimeError('EMA update occurred before bank commit')
        super()._update_ema_for_step(pl_module, global_step)

    def on_train_end(self, trainer, pl_module):
        # Keep regular and EMA sources distinct. The caller explicitly exports
        # either model; fit teardown must not silently replace regular weights.
        boundary(pl_module.adapter)
        self._swapped_state_dict = None

    def state_dict(self):
        result = super().state_dict()
        result['marn_ema_policy'] = POLICY
        result['marn_decay'] = self._decay
        result['marn_tau'] = self._tau
        result['marn_update_interval'] = self._update_interval_steps
        return result

    def load_state_dict(self, state_dict):
        expected = dict(marn_ema_policy=POLICY, marn_decay=self._decay,
                        marn_tau=self._tau, marn_update_interval=self._update_interval_steps)
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError(f'EMA recovery configuration mismatch: {key}')
        super().load_state_dict(state_dict)


def export_inference(module, path, *, source):
    """Write parameters + frozen bank + reconstruction metadata, no optimizer.

    This explicit MARN format must be loaded by load_inference, not RFDETRSmall
    or the stock lightweight checkpoint loader (which cannot attach MARN).
    """
    boundary(module.adapter)
    if source not in ('regular', 'ema'):
        raise ValueError('source must be regular or ema')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(format=FORMAT, source=source, **description(module),
                   model_config=module.model_config.model_dump(mode='json'),
                   train_config=module.train_config.model_dump(mode='json'),
                   model={k: v.detach().cpu().clone() for k, v in module.model.state_dict().items()})
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as file:
        torch.save(payload, file)
    os.replace(temp, path)


def load_inference(path, *, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT or payload.get('marn_ema_policy') != POLICY:
        raise ValueError('Not a compatible self-describing MARN inference artifact')
    model_config = dict(payload['model_config'])
    model_config.update(pretrain_weights=None, device='cpu')
    train_config = dict(payload['train_config'])
    train_config.update(accelerator='cpu', devices=1)
    model = MARNTrainingModule(RFDETRSmallConfig(**model_config), TrainConfig(**train_config),
                              class_ids=payload['marn_class_ids'],
                              marn_config=MARNConfig(**payload['marn_config']))
    if description(model) != {k: payload[k] for k in description(model)}:
        raise ValueError('MARN inference metadata differs from runtime')
    model.model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    boundary(model.adapter)
    return model
