"""Bounded real-data EMA/bank/export/recovery check; not a training benchmark."""
import os
os.environ.update({
    'RF_HOME': '/root/autodl-tmp/models/roboflow',
    'HF_HOME': '/root/autodl-tmp/cache/huggingface',
    'TORCH_HOME': '/root/autodl-tmp/cache/torch',
    'XDG_CACHE_HOME': '/root/autodl-tmp/cache',
    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    'TOKENIZERS_PARALLELISM': 'false',
})

import copy
import gc
import hashlib
import json
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/project_marn-detr')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'check_outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
REPORT = {'passed': False, 'stage': 'imports',
          'scope': '8 real train microbatches + 2 after native recovery; not AP or speed evidence'}


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import torch
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.config import RFDETRSmallConfig, TrainConfig
    from rfdetr.marn.data import MARNDataModule
    from rfdetr.marn.integration import MARNConfig
    from rfdetr.marn.lifecycle import (MARNTrainingModule, MARNEMACallback,
                                      POLICY, boundary, export_inference, load_inference)

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert shutil.disk_usage(OUT).free > 3 * 1024**3, 'Keep at least 3 GiB free for temporary checks'
    torch.set_num_threads(4)
    seed_everything(42, workers=True)
    torch.cuda.reset_peak_memory_stats()

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape
            torch.testing.assert_close(a.detach().cpu(), b.detach().cpu(), rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), (a.keys(), b.keys())
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b):
                exact(left, right)
        else:
            assert a == b, (a, b)

    def frozen_buffers(adapter):
        return {k: v.detach().cpu().clone() for k, v in adapter.named_buffers()}

    def finite_output(output):
        for name in ('pred_logits', 'pred_boxes'):
            assert torch.isfinite(output[name]).all(), name

    @torch.inference_mode()
    def predict(module, inputs):
        module.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = module.model(inputs)
        finite_output(result)
        return {k: result[k].detach().cpu().clone() for k in ('pred_logits', 'pred_boxes')}

    REPORT['stage'] = 'configuration'
    saved_config = ROOT / 'runs/baseline_10ep_seed42/resolved_train_config.json'
    config = json.loads(saved_config.read_text(encoding='utf-8'))
    # Preserve the baseline's data augmentation; use 8x2 to exercise accumulation
    # at the same effective batch of 16. This is an eight-microbatch diagnostic.
    config.update(output_dir=str(OUT), batch_size=8, grad_accum_steps=2, eval_batch_size=4,
                  num_workers=4, persistent_workers=False, prefetch_factor=2, pin_memory=True,
                  accelerator='gpu', devices=1, epochs=2, use_ema=True,
                  ema_update_interval=1, compute_val_loss=False, eval_base_model=True,
                  run_test=False, save_dataset_grids=False, progress_bar=None,
                  tensorboard=False, wandb=False, mlflow=False, clearml=False)
    tc = TrainConfig(**config)
    assert tc.dataset_file == 'coco' and getattr(tc.multi_scale, 'value', tc.multi_scale) == 'off'
    weights = Path(os.environ['RF_HOME']) / 'rf-detr-small.pth'
    assert weights.is_file(), weights
    mc = RFDETRSmallConfig(pretrain_weights=str(weights), num_classes=90, resolution=512,
                          device='cuda', compile=False, cuda_graphs=False, gradient_checkpointing=False)
    cfg = MARNConfig(warmup_steps=1)  # smoke setting, not a formal experiment protocol
    dm = MARNDataModule(mc, tc)
    dm.setup('fit')
    class_ids = sorted(dm._dataset_train.coco.getCatIds())
    assert len(class_ids) == 80 and len(dm._dataset_train) == 118287
    (OUT / 'protocol.json').write_text(json.dumps({
        'policy': POLICY, 'train': tc.model_dump(mode='json'), 'model': mc.model_dump(mode='json'),
        'baseline_config_sha256': sha(saved_config), 'train_microbatches': 8,
        'resume_microbatches': 2, 'scope': REPORT['scope'],
    }, indent=2), encoding='utf-8')

    class BoundaryCheckpoint(ModelCheckpoint):
        def on_train_epoch_end(self, trainer, pl_module):
            # Same callback identity and directory on recovery. Never overwrite
            # the boundary artifact with the deliberately shortened second epoch.
            if trainer.global_step == 4:
                super().on_train_epoch_end(trainer, pl_module)

    records = []
    REPORT['commits'] = records
    probe = {}
    saved_checkpoint = None
    inspected_parameters = ['model.class_embed.weight',
                            'model.transformer.decoder.marn.reader.key.weight',
                            'model.transformer.decoder.marn.refiner.out.weight']

    class Audit(Callback):
        def __init__(self, ema, recovering=False):
            self.ema, self.recovering = ema, recovering

        def on_train_start(self, trainer, module):
            self.previous = trainer.global_step
            self.microbatches = 0
            self.accepted = 0
            self.before_bank = frozen_buffers(module.adapter)
            self.before_ema = {k: dict(self.ema._average_model.module.named_parameters())[k].detach().clone()
                               for k in inspected_parameters}
            self.previous_averages = int(self.ema._average_model.n_averaged)
            assert module.training and module.adapter.training
            assert module.criterion.marn_observer.__self__ is module.adapter
            assert self.ema._average_model.module.criterion.marn_observer.__self__ is self.ema._average_model.module.adapter
            if self.recovering:
                assert trainer.global_step == 4 and trainer.current_epoch == 1
                exact(module.state_dict(), saved_checkpoint['state_dict'])
                exact(trainer.optimizers[0].state_dict(), saved_checkpoint['optimizer_states'][0])
                exact(trainer.lr_scheduler_configs[0].scheduler.state_dict(), saved_checkpoint['lr_schedulers'][0])
                exact(self.ema.state_dict(), saved_checkpoint['callbacks'][self.ema.state_key])
                assert self.previous_averages == 4
                REPORT['full_state_restore'] = 'passed'
                print('C3B_FULL_STATE_RESTORE_OK epoch=1 global_step=4', flush=True)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if self.recovering and self.microbatches == 0 and self.previous == 4:
                assert batch_idx == 0 and trainer.current_epoch == 1
                print('C3B_RESUME_FIRST_BATCH epoch=1 batch_idx=0', flush=True)
            _, targets = batch
            assert len(targets) == 8
            for target in targets:
                assert target['marn_object_id'].dtype == torch.int64
                assert len(target['marn_object_id']) == len(target['labels']) == len(target['boxes'])

        def on_before_backward(self, trainer, module, loss):
            assert torch.isfinite(loss).all()
            self.microbatches += 1
            self.accepted += module.adapter.last_read['accepted']
            exact(frozen_buffers(module.adapter), self.before_bank)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            # This callback is ordered after MARNEMACallback.
            if trainer.global_step == self.previous:
                assert self.microbatches == 1
                assert int(self.ema._average_model.n_averaged) == self.previous_averages
                exact(frozen_buffers(module.adapter), self.before_bank)
                return
            a, average = module.adapter, self.ema._average_model.module
            boundary(a)
            assert trainer.global_step == self.previous + 1
            assert int(a.committed_step) == trainer.global_step
            assert self.microbatches == a.last_commit['microbatches'] == 2
            assert int(self.ema._average_model.n_averaged) == self.previous_averages + 1
            exact(frozen_buffers(a), frozen_buffers(average.adapter))
            live_params, ema_params = dict(module.named_parameters()), dict(average.named_parameters())
            for key in inspected_parameters:
                live = live_params[key].detach()
                if self.previous_averages == 0:
                    expected = live
                else:
                    decay = self.ema._effective_decay(self.previous_averages)
                    expected = self.before_ema[key] * decay + live * (1 - decay)
                torch.testing.assert_close(ema_params[key], expected, rtol=2e-6, atol=2e-7)
            if a.last_commit['eligible']:
                assert a.last_commit['duplicates'] == a.last_commit['eligible'] - a.last_commit['unique_objects']
            record = dict(a.last_commit, global_step=trainer.global_step, accepted_reads=self.accepted)
            records.append(record)
            print('C3B_COMMIT_EMA_OK', json.dumps(record), flush=True)
            self.previous = trainer.global_step
            self.microbatches = 0
            self.accepted = 0
            self.previous_averages = int(self.ema._average_model.n_averaged)
            self.before_bank = frozen_buffers(a)
            self.before_ema = {k: ema_params[k].detach().clone() for k in inspected_parameters}
            if self.recovering and trainer.global_step == 5:
                trainer.should_stop = True

        def on_validation_epoch_start(self, trainer, module):
            self.val_live = frozen_buffers(module.adapter)
            self.val_ema = frozen_buffers(self.ema._average_model.module.adapter)
            assert not module.training and not self.ema._average_model.module.training

        def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
            samples, targets = batch
            assert all('marn_object_id' not in target for target in targets)
            # Native validation_step forwards regular weights. Explicit EMA forward
            # checks the other model without computing misleading partial-set AP.
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                ema_output = self.ema._average_model.module.model(samples)
            finite_output(ema_output)
            exact(self.val_live, frozen_buffers(module.adapter))
            exact(self.val_ema, frozen_buffers(self.ema._average_model.module.adapter))
            if not self.recovering and batch_idx == 0:
                probe['input'] = samples.tensors[:1].detach().cpu().clone()
            REPORT['validation_bank_frozen'] = 'passed'

        def on_train_epoch_end(self, trainer, module):
            if not self.recovering:
                assert trainer.global_step == 4 and trainer.current_epoch == 0
                assert int(module.adapter.bank.valid.sum()) > 0
                assert sum(r['accepted_reads'] for r in records) > 0
                assert module.adapter.refiner.out.weight.detach().abs().sum() > 0
                export_inference(module, OUT / 'regular_test.pth', source='regular')
                export_inference(self.ema._average_model.module, OUT / 'ema_test.pth', source='ema')
                trainer.should_stop = True

    def make_trainer(ema, audit):
        checkpoint = BoundaryCheckpoint(dirpath=str(OUT), filename='c3b_epoch_boundary',
                                       monitor=None, save_top_k=1, every_n_epochs=1,
                                       save_on_train_epoch_end=True, enable_version_counter=False,
                                       auto_insert_metric_name=False)
        trainer = Trainer(accelerator='gpu', devices=1, precision='bf16-mixed', max_epochs=2,
                          limit_train_batches=8, limit_val_batches=2, num_sanity_val_steps=0,
                          accumulate_grad_batches=2, gradient_clip_val=tc.clip_max_norm,
                          callbacks=[ema, audit, checkpoint], logger=False, enable_checkpointing=True,
                          enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(OUT))
        return trainer

    REPORT['stage'] = 'real-data train and frozen validation'
    model = MARNTrainingModule(copy.deepcopy(mc), copy.deepcopy(tc), class_ids=class_ids, marn_config=cfg)
    ema = MARNEMACallback(decay=tc.ema_decay, tau=tc.ema_tau)
    audit = Audit(ema)
    trainer = make_trainer(ema, audit)
    trainer.fit(model, datamodule=dm)
    ckpt = OUT / 'c3b_epoch_boundary.ckpt'
    assert ckpt.is_file()
    ckpt_hash = sha(ckpt)
    saved_checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert saved_checkpoint['epoch'] == 0 and saved_checkpoint['global_step'] == 4
    assert saved_checkpoint['marn_ema_policy'] == POLICY
    # Exact regular-state retention after fit; no implicit final EMA replacement.
    exact(model.state_dict(), saved_checkpoint['state_dict'])
    exact(ema.state_dict(), saved_checkpoint['callbacks'][ema.state_key])
    print('C3B_NATIVE_BOUNDARY_SAVED epoch=0 global_step=4', flush=True)

    REPORT['stage'] = 'self-describing inference exports'
    inputs = probe['input'].cuda()
    model.cuda()
    expected_regular = predict(model, inputs)
    expected_ema = predict(ema._average_model.module.cuda(), inputs)
    # Clear GPU/CPU model ownership before reconstructing standalone artifacts.
    dm.trainer = None
    del model, trainer, audit, ema
    gc.collect()
    torch.cuda.empty_cache()
    for source, expected in [('regular', expected_regular), ('ema', expected_ema)]:
        path = OUT / f'{source}_test.pth'
        exported = torch.load(path, map_location='cpu', weights_only=True)
        if source == 'regular':
            expected_weights = {k.removeprefix('model.'): v for k, v in saved_checkpoint['state_dict'].items()
                                if k.startswith('model.')}
        else:
            callback = saved_checkpoint['callbacks']['MARNEMACallback:' + POLICY]
            expected_weights = {k.removeprefix('module.model.'): v
                                for k, v in callback['average_model_state_dict'].items()
                                if k.startswith('module.model.')}
        exact(exported['model'], expected_weights)
        restored = load_inference(path, device='cuda')
        before = frozen_buffers(restored.adapter)
        actual = predict(restored, inputs)
        exact(actual, expected)
        exact(frozen_buffers(restored.adapter), before)
        print('C3B_INFERENCE_ROUNDTRIP_OK', source, flush=True)
        del restored, exported, expected_weights, actual
        gc.collect()
        torch.cuda.empty_cache()
    REPORT['inference_regular_and_ema'] = 'exact state and logits/boxes roundtrip passed'

    REPORT['stage'] = 'native full training recovery'
    fresh_mc = copy.deepcopy(mc)
    fresh_mc.pretrain_weights = None  # strict checkpoint load; no download or second initialization checkpoint
    restored = MARNTrainingModule(fresh_mc, copy.deepcopy(tc), class_ids=class_ids, marn_config=cfg)
    resumed_ema = MARNEMACallback(decay=tc.ema_decay, tau=tc.ema_tau)
    resumed_audit = Audit(resumed_ema, recovering=True)
    resume = make_trainer(resumed_ema, resumed_audit)
    resume.fit(restored, datamodule=dm, ckpt_path=str(ckpt))
    assert resume.global_step == 5 and int(restored.adapter.committed_step) == 5
    assert int(resumed_ema._average_model.n_averaged) == 5
    assert len(records) == 5 and records[-1]['microbatches'] == 2
    assert sha(ckpt) == ckpt_hash, 'Recovery overwrote the original boundary checkpoint'
    REPORT['commits'] = records
    REPORT['resume_two_microbatches'] = 'passed'
    REPORT['peak_allocated_GiB'] = torch.cuda.max_memory_allocated() / 1024**3
    REPORT['peak_reserved_GiB'] = torch.cuda.max_memory_reserved() / 1024**3
    REPORT['temporary_weights'] = [{'file': p.name, 'bytes': p.stat().st_size}
                                   for p in sorted(OUT.iterdir()) if p.suffix in ('.ckpt', '.pth')]
    REPORT['passed'] = True
    REPORT['stage'] = 'complete'
    print('C3B_EMA_CHECKPOINT_CHECK_COMPLETED', flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    original_out, original_err = sys.stdout, sys.stderr
    started = time.monotonic()
    with (OUT / 'c3b_check.log').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            main()
        except BaseException as exc:
            REPORT['error'] = repr(exc)
            traceback.print_exc()
        finally:
            REPORT['elapsed_seconds'] = time.monotonic() - started
            (OUT / 'c3b_result.json').write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding='utf-8')
            sys.stdout, sys.stderr = original_out, original_err
    report_zip = HERE / ('c3b_report_' + OUT.name + '.zip')
    with zipfile.ZipFile(report_zip, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in ('c3b_check.log', 'c3b_result.json', 'protocol.json'):
            path = OUT / name
            if path.is_file():
                z.write(path, name)
        for name in ('check_c3b.py', 'lifecycle.py', 'source_guard.json'):
            z.write(HERE / name, 'scripts/' + name)
    print('C3B_REPORT_ZIP', report_zip, flush=True)
    sys.exit(0 if REPORT['passed'] else 1)
