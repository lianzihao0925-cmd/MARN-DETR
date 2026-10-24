import os

os.environ["RF_HOME"] = "/root/autodl-tmp/models/roboflow"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/cache/torch"
os.environ["XDG_CACHE_HOME"] = "/root/autodl-tmp/cache"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from rfdetr import RFDETRSmall
from rfdetr.config import TrainConfig


class SpeedMeter(Callback):
    def __init__(self):
        self.count = 0
        self.images = 0
        self.windows = []
        self.started = None

    def on_train_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()
        print("BENCH_TRAIN_STARTED", flush=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        samples = batch[0]
        tensor = samples.tensors if hasattr(samples, "tensors") else samples
        shape = tuple(tensor.shape)
        if self.count == 0:
            print("实际输入形状:", shape, flush=True)
        if shape[-2:] != (512, 512):
            raise RuntimeError(f"固定512测试收到其他尺寸：{shape}")
        self.batch_images = shape[0]

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self.count += 1

        if self.count > 100:
            self.images += self.batch_images

        # 仅在计时边界同步，不在每个 batch 同步。
        if self.count == 100:
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            print("预热完成，开始测速", flush=True)

        elif self.count in (200, 300, 400):
            torch.cuda.synchronize()
            now = time.perf_counter()
            seconds = now - self.started
            record = {
                "images": self.images,
                "seconds": seconds,
                "images_per_second": self.images / seconds,
            }
            self.windows.append(record)
            print("BENCH_WINDOW:", json.dumps(record), flush=True)
            self.images = 0
            self.started = time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    args = parser.parse_args()

    if args.batch <= 0 or args.accum <= 0 or 400 % args.accum:
        raise ValueError("batch、accum 必须为正数，且 accum 必须整除400")

    seed_everything(42, workers=True)
    torch.set_num_threads(4)

    output = Path(
        "/root/autodl-tmp/project_marn-detr/runs/throughput"
    ) / f"b{args.batch}_a{args.accum}"
    output.mkdir(parents=True, exist_ok=True)

    meter = SpeedMeter()
    trainer_module = importlib.import_module("rfdetr.training.trainer")

    def bounded_trainer(*positional, **kwargs):
        callbacks = kwargs.get("callbacks") or []
        removed = (ModelCheckpoint, LearningRateMonitor, EarlyStopping)
        callbacks = [c for c in callbacks if not isinstance(c, removed)]
        callbacks.append(meter)

        kwargs.update(
            callbacks=callbacks,
            max_steps=400 // args.accum,
            max_time={"minutes": 5},
            limit_val_batches=0,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        print("保留的回调:",
              [type(c).__name__ for c in callbacks], flush=True)
        return Trainer(*positional, **kwargs)

    model = RFDETRSmall(
        pretrain_weights="/root/autodl-tmp/models/roboflow/rf-detr-small.pth",
        device="cuda",
        gradient_checkpointing=False,
    )

    settings = dict(
        dataset_file="coco",
        dataset_dir="/root/autodl-tmp/project_marn-detr/datasets/coco",
        output_dir=str(output),
        resolution=512,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multi_scale="off",
        amp_dtype="bf16",
        device="cuda",
        progress_bar=None,
        save_dataset_grids=False,
    )
    for name in ("tensorboard", "wandb", "mlflow", "clearml"):
        if name in TrainConfig.model_fields:
            settings[name] = False

    print("物理batch:", args.batch,
          "有效batch:", args.batch * args.accum, flush=True)

    wall_start = time.perf_counter()
    with patch.object(trainer_module, "Trainer", bounded_trainer):
        model.train(**settings)

    result = {
        "physical_batch": args.batch,
        "accumulation": args.accum,
        "effective_batch": args.batch * args.accum,
        "batches_completed": meter.count,
        "windows": meter.windows,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
        "train_call_wall_seconds": time.perf_counter() - wall_start,
    }
    if meter.windows:
        images = sum(w["images"] for w in meter.windows)
        seconds = sum(w["seconds"] for w in meter.windows)
        result["measured_images_per_second"] = images / seconds
        result["projected_train_only_minutes"] = 118287 / (images / seconds) / 60

    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("BENCH_RESULT:", json.dumps(result, indent=2), flush=True)
    print(
        "BENCH_COMPLETED" if meter.count == 400 else "BENCH_INCOMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
