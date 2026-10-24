"""Run on the AutoDL server: a bounded, pretrained RF-DETR Small baseline."""

import os

os.environ.update({
    "RF_HOME": "/root/autodl-tmp/models/roboflow",
    "HF_HOME": "/root/autodl-tmp/cache/huggingface",
    "TORCH_HOME": "/root/autodl-tmp/cache/torch",
    "XDG_CACHE_HOME": "/root/autodl-tmp/cache",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
})

import hashlib
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

START = time.monotonic()
ROOT = Path("/root/autodl-tmp/project_marn-detr")
REPO = ROOT / "rf-detr"
OUT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ["RF_HOME"]) / "rf-detr-small.pth"


def save_json(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True, encoding="utf-8"
    ).strip()


def main():
    # Use a new directory for each experiment. Never silently overwrite a run.
    if (OUT / "protocol.json").exists() or (OUT / "last.ckpt").exists():
        raise RuntimeError("Output already contains a run. Do not restart or overwrite it.")
    assert WEIGHTS.is_file(), str(WEIGHTS)
    md5 = hashlib.md5(WEIGHTS.read_bytes()).hexdigest()
    assert md5 == "fb37061c1af7bace359c91b723a8d5c1", md5

    import torch
    from pytorch_lightning import Callback, seed_everything
    from rfdetr import RFDETRSmall
    import rfdetr.training.trainer as trainer_module

    torch.set_num_threads(4)
    seed_everything(42, workers=True)

    train_config = dict(
        dataset_file="coco",
        dataset_dir=str(ROOT / "datasets/coco"),
        output_dir=str(OUT),
        device="cuda",
        resolution=512,
        square_resize_div_64=True,
        multi_scale="off",
        augmentation_backend="cpu",
        batch_size=16,
        grad_accum_steps=1,
        eval_batch_size=8,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        amp_dtype="bf16",
        epochs=10,
        seed=42,
        optimizer="adamw",
        lr=1e-5,
        lr_encoder=1.5e-5,
        lr_component_decay=0.7,
        lr_vit_layer_decay=0.8,
        weight_decay=1e-4,
        warmup_epochs=0.5,
        lr_scheduler="cosine",
        lr_scheduler_kwargs={"min_factor": 0.1},
        clip_max_norm=0.1,
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        eval_interval=1,
        eval_base_model=True,
        eval_max_dets=100,
        compute_val_loss=False,
        # Keep last.ckpt every epoch and the native best checkpoints.
        # A separate interval archive will not be triggered in this 10-epoch run.
        checkpoint_interval=100,
        num_sanity_val_steps=0,
        early_stopping=False,
        run_test=False,
        save_dataset_grids=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        progress_bar=None,
    )
    model_config = dict(
        pretrain_weights=str(WEIGHTS),
        num_classes=90,
        resolution=512,
        gradient_checkpointing=False,
        device="cuda",
    )
    # Validate all settings on CPU before constructing a GPU model.
    from rfdetr.config import TrainConfig
    validated = TrainConfig(**{k: v for k, v in train_config.items()
                               if k not in {"resolution", "device"}})
    save_json("resolved_train_config.json", validated.model_dump(mode="json"))
    save_json("protocol.json", {
        "description": "10-epoch pretrained baseline; seed 42; not an original pretraining reproduction",
        "model": model_config,
        "train": train_config,
        "weight_md5": md5,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "evaluation": "COCO val2017; maxDets=100; BF16; same protocol before/after training",
        "soft_total_seconds": 13200,
        "seed_count": 1,
    })
    save_json("source_sha256.json", {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((REPO / "src/rfdetr").rglob("*.py"))
    })

    model = RFDETRSmall(**model_config)
    assert model.model_config.num_classes == 90
    eval_config = {k: train_config[k] for k in (
        "dataset_file", "dataset_dir", "device", "resolution", "square_resize_div_64",
        "multi_scale", "augmentation_backend", "num_workers", "pin_memory",
        "persistent_workers", "prefetch_factor", "amp_dtype", "eval_max_dets",
        "compute_val_loss", "progress_bar",
    )}
    eval_config.update(batch_size=8, output_dir=str(OUT / "initial_eval"))
    print("INITIAL_EVALUATION_STARTED maxDets=100", flush=True)
    initial = model.evaluate(split="val", **eval_config)
    save_json("initial_metrics_maxdets100.json", initial)
    print("INITIAL_EVALUATION_COMPLETED", json.dumps(initial), flush=True)

    epoch_records = []

    class Audit(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self.batches = 0
            self.finite = None
            self.loss_batches = 0
            self.started = time.monotonic()
            print(f"EPOCH_STARTED epoch={trainer.current_epoch + 1}/10", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.batches += 1
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if isinstance(loss, torch.Tensor):
                finite = torch.isfinite(loss.detach()).all()
                self.finite = finite if self.finite is None else self.finite & finite
                self.loss_batches += 1
            if self.batches % 500 == 0:
                if self.finite is not None and not bool(self.finite.item()):
                    raise FloatingPointError("Nonfinite training loss")
                value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else None
                print(f"TRAIN_PROGRESS epoch={trainer.current_epoch + 1} "
                      f"batches={self.batches} global_step={trainer.global_step} loss={value}", flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            finite = None if self.finite is None else bool(self.finite.item())
            if finite is False:
                raise FloatingPointError("Nonfinite training loss")
            metrics = {}
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics[key] = float(value.detach().cpu().item())
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
            record = dict(
                epoch=trainer.current_epoch + 1,
                global_step=trainer.global_step,
                training_batches=self.batches,
                full_epoch=self.batches == int(trainer.num_training_batches),
                loss_batches_checked=self.loss_batches,
                all_observed_losses_finite=finite,
                epoch_seconds_before_checkpoint=time.monotonic() - self.started,
                total_elapsed_seconds=time.monotonic() - START,
                learning_rates=[g["lr"] for g in trainer.optimizers[0].param_groups],
                metrics=metrics,
            )
            epoch_records.append(record)
            save_json("epoch_audit.json", epoch_records)
            print("EPOCH_RESULT", json.dumps(record), flush=True)

    OriginalTrainer = trainer_module.Trainer

    class BudgetTrainer(OriginalTrainer):
        def __init__(self, *args, **kwargs):
            remaining = int(13200 - (time.monotonic() - START))
            if remaining <= 0:
                raise RuntimeError("Training time budget exhausted before fit")
            kwargs["max_time"] = {"seconds": remaining}
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), Audit()]
            super().__init__(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats()
    print("FORMAL_BASELINE_STARTED epochs=10 seed=42 batch=16 maxDets=100", flush=True)
    with patch.object(trainer_module, "Trainer", BudgetTrainer):
        model.train(**train_config)

    complete = sum(r["full_epoch"] for r in epoch_records) == 10
    save_json("run_summary.json", {
        "completed_10_full_epochs": complete,
        "full_epochs": sum(r["full_epoch"] for r in epoch_records),
        "elapsed_seconds": time.monotonic() - START,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
        "last_checkpoint_exists": (OUT / "last.ckpt").is_file(),
        "checkpoints": [p.name for p in sorted(OUT.glob("*.pth"))],
    })
    print("BASELINE_10EPOCH_COMPLETED" if complete else "BASELINE_STOPPED_WITHIN_BUDGET", flush=True)
    print("TOTAL_SECONDS", round(time.monotonic() - START, 2), flush=True)


if __name__ == "__main__":
    main()
