import os

os.environ["RF_HOME"] = "/root/autodl-tmp/models/roboflow"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/cache/torch"
os.environ["XDG_CACHE_HOME"] = "/root/autodl-tmp/cache"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from rfdetr import RFDETRSmall

OUT = Path("/root/autodl-tmp/project_marn-detr/runs/epoch_check")
RESULT = {"validation_seconds": 0.0, "checkpoint_writes": []}


class EpochAudit(Callback):
    def __init__(self):
        self.batches = 0
        self.images = 0
        self.loss_ok = None

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        print("EPOCH_STARTED", flush=True)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self.batches += 1
        samples = batch[0]
        tensor = samples.tensors if hasattr(samples, "tensors") else samples
        self.images += int(tensor.shape[0])
        if tuple(tensor.shape[-2:]) != (512, 512):
            raise RuntimeError("实际输入不符合固定512测试设置")

        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("未取得训练loss，需要核对训练输出接口")
        finite = torch.isfinite(loss.detach()).all()
        self.loss_ok = finite if self.loss_ok is None else self.loss_ok & finite

        # 按区间检查，避免每个batch额外同步。
        if self.batches % 100 == 0:
            if not bool(self.loss_ok.item()):
                raise RuntimeError("出现非有限训练loss，停止验收")
        if self.batches % 500 == 0:
            print(
                f"TRAIN_PROGRESS batches={self.batches} "
                f"images={self.images} loss={loss.detach().item():.5f}",
                flush=True,
            )

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.val_started = time.perf_counter()
        print("VALIDATION_STARTED", flush=True)

    def on_validation_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        RESULT["validation_seconds"] += time.perf_counter() - self.val_started
        print("VALIDATION_FINISHED", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.loss_ok is None or not bool(self.loss_ok.item()):
            raise RuntimeError("训练loss检查未通过")
        RESULT.update(
            epoch=int(trainer.current_epoch),
            global_step=int(trainer.global_step),
            training_batches=self.batches,
            training_images=self.images,
            all_observed_losses_finite=True,
            end_learning_rates=[
                float(g["lr"]) for g in trainer.optimizers[0].param_groups
            ],
        )
        # 完整第一轮及其验证完成后停止，不改变名义调度长度。
        trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        RESULT["epoch_through_train_end_seconds"] = (
            time.perf_counter() - self.started
        )
        RESULT["peak_allocated_GiB"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        RESULT["peak_reserved_GiB"] = (
            torch.cuda.max_memory_reserved() / 1024**3
        )


class AuditedTrainer(Trainer):
    def save_checkpoint(self, filepath, *args, **kwargs):
        start = time.perf_counter()
        super().save_checkpoint(filepath, *args, **kwargs)
        RESULT["checkpoint_writes"].append({
            "path": str(filepath),
            "seconds": time.perf_counter() - start,
        })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    last = OUT / "last.ckpt"
    if last.exists():
        raise RuntimeError("已有last.ckpt；停止覆盖，请先核对已有运行")

    seed_everything(42, workers=True)
    torch.set_num_threads(4)
    model = RFDETRSmall(
        pretrain_weights="/root/autodl-tmp/models/roboflow/rf-detr-small.pth",
        device="cuda",
        gradient_checkpointing=False,
    )

    tm = importlib.import_module("rfdetr.training.trainer")

    def make_trainer(*args, **kwargs):
        kwargs["callbacks"] = [
            *(kwargs.get("callbacks") or []), EpochAudit()
        ]
        return AuditedTrainer(*args, **kwargs)

    start = time.perf_counter()
    with patch.object(tm, "Trainer", make_trainer):
        model.train(
            dataset_file="coco",
            dataset_dir="/root/autodl-tmp/project_marn-detr/datasets/coco",
            output_dir=str(OUT),
            device="cuda",
            resolution=512,
            multi_scale="off",
            batch_size=16,
            grad_accum_steps=1,
            eval_batch_size=8,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            amp_dtype="bf16",
            epochs=100,
            seed=42,
            lr=1e-4,
            lr_encoder=1.5e-4,
            weight_decay=1e-4,
            warmup_epochs=0.0,
            clip_max_norm=0.1,
            use_ema=True,
            eval_interval=1,
            eval_base_model=True,
            eval_max_dets=500,
            checkpoint_interval=10,
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

    RESULT["train_call_wall_seconds"] = time.perf_counter() - start
    assert last.is_file(), "未生成last.ckpt"

    # 只读取本次程序自己生成的可信checkpoint。
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    RESULT["last_checkpoint"] = {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "optimizer_states": len(checkpoint.get("optimizer_states") or []),
        "lr_schedulers": len(checkpoint.get("lr_schedulers") or []),
        "callback_keys": list((checkpoint.get("callbacks") or {}).keys()),
        "top_level_keys": list(checkpoint.keys()),
        "size_GiB": last.stat().st_size / 1024**3,
    }
    del checkpoint

    (OUT / "audit.json").write_text(
        json.dumps(RESULT, indent=2), encoding="utf-8"
    )
    print("EPOCH_AUDIT:", json.dumps(RESULT, indent=2), flush=True)
    print("EPOCH_CHECK_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
