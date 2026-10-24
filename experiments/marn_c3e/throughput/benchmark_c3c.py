"""Paired end-to-end training throughput; installed model source is read-only."""
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
import statistics
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/project_marn-detr')
PACKAGE = ROOT / 'rf-detr/src/rfdetr'
HERE = Path(__file__).resolve().parent
OUT = HERE / 'benchmark_outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
WARMUP = 10
MEASURE = 50
REPETITIONS = 3
BATCH = 16
REPORT = {'measurements_completed': False, 'stage': 'source guard', 'runs': [],
          'scope': 'training only: data loading/transfer, forward, matching/loss, backward, optimizer, bank, EMA'}


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
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_report():
    (OUT / 'c3c_result.json').write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding='utf-8')


def summarize(runs):
    result = {}
    for mode, target in [('baseline', 100.0), ('marn', 80.0)]:
        chosen = [r for r in runs if r['mode'] == mode]
        speeds = [r['images_per_second'] for r in chosen]
        mean = statistics.mean(speeds)
        result[mode] = dict(repetitions=len(chosen), images_per_second=speeds,
                            mean_images_per_second=mean,
                            sample_standard_deviation=statistics.stdev(speeds) if len(speeds) > 1 else None,
                            target_images_per_second=target, mean_meets_target=mean >= target,
                            peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in chosen),
                            training_only_epoch_minutes_estimate=118288 / mean / 60)
    result['marn_speed_fraction_of_baseline'] = (
        result['marn']['mean_images_per_second'] / result['baseline']['mean_images_per_second'])
    return result


def main():
    guard = json.loads((HERE / 'source_guard.json').read_text(encoding='utf-8'))
    for name, expected in guard.items():
        path = PACKAGE / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'Source differs from accepted C3B version: {name}; no training started')
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning import Callback, Trainer, seed_everything
    from rfdetr.config import RFDETRSmallConfig, TrainConfig
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.utilities.tensors import PackedTargets
    from rfdetr.marn.data import MARNDataModule
    from rfdetr.marn.integration import MARNConfig
    from rfdetr.marn.lifecycle import MARNTrainingModule, MARNEMACallback, POLICY, boundary

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    weights = Path(os.environ['RF_HOME']) / 'rf-detr-small.pth'
    assert weights.is_file(), weights
    config_path = ROOT / 'runs/baseline_10ep_seed42/resolved_train_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    # A complete benchmark epoch has only 60 steps. Its LR schedule is a short
    # diagnostic schedule; none of these updates are used for accuracy research.
    config.update(output_dir=str(OUT), batch_size=BATCH, grad_accum_steps=1,
                  num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2,
                  accelerator='gpu', devices=1, epochs=1, use_ema=True, ema_update_interval=1,
                  compute_val_loss=False, compute_train_metrics=False, train_log_on_step=False,
                  run_test=False, save_dataset_grids=False, progress_bar=None,
                  tensorboard=False, wandb=False, mlflow=False, clearml=False)
    tc = TrainConfig(**config)
    assert tc.dataset_file == 'coco' and getattr(tc.multi_scale, 'value', tc.multi_scale) == 'off'
    assert tc.square_resize_div_64 and tc.pad_targets_to is None
    assert Path(tc.dataset_dir).resolve() == (ROOT / 'datasets/coco').resolve()
    mc = RFDETRSmallConfig(pretrain_weights=str(weights), num_classes=90, resolution=512,
                          device='cuda', compile=False, cuda_graphs=False, gradient_checkpointing=False)
    cfg = MARNConfig(warmup_steps=1)
    REPORT['environment'] = dict(python=sys.version, torch=torch.__version__, lightning=pl.__version__,
                                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                 total_gpu_GiB=torch.cuda.get_device_properties(0).total_memory / 1024**3)
    (OUT / 'protocol.json').write_text(json.dumps({
        'train': tc.model_dump(mode='json'), 'model': mc.model_dump(mode='json'),
        'baseline_config_sha256': sha(config_path), 'source_sha256': guard,
        'marn_config': cfg.__dict__, 'marn_ema_policy': POLICY,
        'warmup_batches': WARMUP, 'measured_batches': MEASURE, 'repetitions': REPETITIONS,
        'batch_size': BATCH, 'grad_accum_steps': 1, 'scope': REPORT['scope'],
        'not_timed': ['initialization', 'worker startup/warmup', 'validation', 'checkpoint writes', 'epoch-end teardown'],
    }, indent=2), encoding='utf-8')

    class TraceDataModule(MARNDataModule):
        def reset_trace(self):
            self.target_hash = hashlib.sha256()
            self.pixel_hash = hashlib.sha256()
            self.transfers = 0
            self.active_loader = None

        def train_dataloader(self):
            self.active_loader = super().train_dataloader()
            return self.active_loader

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            samples, targets = batch
            # Public packed fields are CPU tensors here. Hash only small target
            # metadata in the timed section; no added GPU synchronization.
            entries = [targets.fields] if isinstance(targets, PackedTargets) else targets
            for target in entries:
                for name in sorted(target):
                    value = target[name]
                    assert value.device.type == 'cpu'
                    self.target_hash.update(name.encode())
                    self.target_hash.update(str((value.dtype, tuple(value.shape))).encode())
                    self.target_hash.update(value.contiguous().numpy().tobytes())
            if self.transfers < 2:
                # Pixel equality evidence is collected only during excluded warmup.
                self.pixel_hash.update(samples.tensors.contiguous().numpy().tobytes())
            self.transfers += 1
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        def release_loader(self):
            loader = self.active_loader
            iterator = getattr(loader, '_iterator', None) if loader is not None else None
            if iterator is not None:
                shutdown = getattr(iterator, '_shutdown_workers', None)
                if shutdown is not None:
                    shutdown()
                loader._iterator = None
            self.active_loader = None
            self.trainer = None

    REPORT['stage'] = 'dataset initialization'
    seed_everything(42, workers=True)
    dm = TraceDataModule(mc, tc)
    dm.reset_trace()
    dm.setup('fit')
    assert len(dm._dataset_train) == 118287
    class_ids = sorted(dm._dataset_train.coco.getCatIds())

    def bank_info(adapter):
        return dict(valid_slots=int(adapter.bank.valid.sum()), total_slots=int(adapter.bank.valid.numel()),
                    covered_classes=int(adapter.bank.classes[adapter.bank.valid].unique().numel()),
                    committed_step=int(adapter.committed_step))

    class Timer(Callback):
        def __init__(self, mode, repetition):
            self.mode, self.repetition = mode, repetition
            self.done = 0
            self.images = 0
            self.finite = None
            self.result = None
            self.accepted = 0
            self.selected = 0
            self.writes = 0

        def on_train_start(self, trainer, module):
            self.fused = bool(trainer.optimizers[0].param_groups[0].get('fused', False))

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.done += 1
            loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
            assert isinstance(loss, torch.Tensor)
            finite = torch.isfinite(loss.detach()).all()
            self.finite = finite if self.finite is None else self.finite & finite
            if self.done == WARMUP:
                # Timer is after EMA in the callback list: all warmup writes and
                # EMA work have completed before the timed window begins.
                torch.cuda.synchronize()
                assert bool(self.finite)
                if self.mode == 'marn':
                    boundary(module.adapter)
                    self.bank_start = bank_info(module.adapter)
                    assert self.bank_start['valid_slots'] > 0
                torch.cuda.reset_peak_memory_stats()
                print(f'C3C_WARMUP_COMPLETED mode={self.mode} repetition={self.repetition}', flush=True)
                self.started = time.perf_counter()
            elif self.done > WARMUP:
                self.images += len(batch[1])
                if self.mode == 'marn':
                    self.accepted += module.adapter.last_read['accepted']
                    self.selected += module.adapter.last_read['selected']
                    self.writes += module.adapter.last_commit['unique_objects']
                if self.done == WARMUP + MEASURE:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - self.started
                    assert bool(self.finite)
                    assert self.images == BATCH * MEASURE
                    self.result = dict(mode=self.mode, repetition=self.repetition,
                                       warmup_batches=WARMUP, measured_batches=MEASURE,
                                       images=self.images, seconds=elapsed,
                                       images_per_second=self.images / elapsed,
                                       all_losses_finite=True, optimizer_fused=self.fused,
                                       peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
                                       peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024**3)
                    if self.mode == 'marn':
                        boundary(module.adapter)
                        self.result.update(bank_at_measurement_start=self.bank_start,
                                           bank_at_measurement_end=bank_info(module.adapter),
                                           accepted_queries=self.accepted, selected_queries=self.selected,
                                           unique_object_writes=self.writes)
                        assert self.accepted > 0 and self.writes > 0
                    print('C3C_RUN_COMPLETED', json.dumps(self.result), flush=True)

    try:
        for repetition in range(1, REPETITIONS + 1):
            seed = 41 + repetition
            # Alternate order to reduce a systematic warm-cache / thermal bias.
            order = ['baseline', 'marn'] if repetition % 2 else ['marn', 'baseline']
            pair = {}
            for mode in order:
                REPORT['stage'] = f'{mode} repetition {repetition}'
                save_report()
                dm.reset_trace()
                dm.train_config.seed = seed
                seed_everything(seed, workers=True)
                run_tc = copy.deepcopy(tc)
                run_tc.seed = seed
                if mode == 'marn':
                    model = MARNTrainingModule(copy.deepcopy(mc), run_tc, class_ids=class_ids, marn_config=cfg)
                    ema = MARNEMACallback(decay=tc.ema_decay, tau=tc.ema_tau)
                else:
                    model = RFDETRModelModule(copy.deepcopy(mc), run_tc, load_encoder_weights=False)
                    ema = RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau)
                timer = Timer(mode, repetition)
                trainer = Trainer(accelerator='gpu', devices=1, precision='bf16-mixed',
                                  max_epochs=1, limit_train_batches=WARMUP + MEASURE,
                                  limit_val_batches=0, num_sanity_val_steps=0,
                                  accumulate_grad_batches=1, gradient_clip_val=tc.clip_max_norm,
                                  callbacks=[ema, timer], logger=False, enable_checkpointing=False,
                                  enable_progress_bar=False, enable_model_summary=False,
                                  default_root_dir=str(OUT))
                print(f'C3C_RUN_STARTED mode={mode} repetition={repetition} seed={seed}', flush=True)
                try:
                    trainer.fit(model, datamodule=dm)
                    assert timer.result is not None and timer.done == WARMUP + MEASURE
                    assert dm.transfers == WARMUP + MEASURE
                    assert trainer.global_step == WARMUP + MEASURE
                    assert int(ema._average_model.n_averaged) == WARMUP + MEASURE
                    record = dict(timer.result, seed=seed, target_sha256=dm.target_hash.hexdigest(),
                                  first_two_warmup_images_sha256=dm.pixel_hash.hexdigest())
                    pair[mode] = record
                    REPORT['runs'].append(record)
                    save_report()
                finally:
                    dm.release_loader()
                del trainer, model, ema, timer
                gc.collect()
                torch.cuda.empty_cache()
            left, right = pair['baseline'], pair['marn']
            assert left['target_sha256'] == right['target_sha256'], 'Paired target order or augmentations differ'
            assert left['first_two_warmup_images_sha256'] == right['first_two_warmup_images_sha256'], 'Paired initial images differ'
            assert left['optimizer_fused'] == right['optimizer_fused'], 'Different AdamW optimizer execution modes'
            print(f'C3C_PAIR_INPUTS_OK repetition={repetition}', flush=True)
        REPORT['summary'] = summarize(REPORT['runs'])
        REPORT['measurements_completed'] = True
        REPORT['stage'] = 'complete'
        print('C3C_THROUGHPUT_COMPLETED', json.dumps(REPORT['summary']), flush=True)
    finally:
        dm.release_loader()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    original_out, original_err = sys.stdout, sys.stderr
    started = time.monotonic()
    with (OUT / 'c3c_benchmark.log').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            main()
        except BaseException as exc:
            REPORT['error'] = repr(exc)
            traceback.print_exc()
        finally:
            REPORT['total_wall_seconds'] = time.monotonic() - started
            save_report()
            sys.stdout, sys.stderr = original_out, original_err
    report_zip = HERE / ('c3c_report_' + OUT.name + '.zip')
    with zipfile.ZipFile(report_zip, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                z.write(path, path.name)
        for name in ('benchmark_c3c.py', 'source_guard.json'):
            z.write(HERE / name, 'scripts/' + name)
    print('C3C_REPORT_ZIP', report_zip, flush=True)
    sys.exit(0 if REPORT['measurements_completed'] else 1)
