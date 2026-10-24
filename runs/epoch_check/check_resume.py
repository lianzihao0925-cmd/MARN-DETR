# 复用工程验收脚本中相同的环境设置。
import check_epoch

import importlib
from pathlib import Path
from unittest.mock import patch

import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from rfdetr import RFDETRSmall

ROOT = Path("/root/autodl-tmp/project_marn-detr")
CHECKPOINT = ROOT / "runs/epoch_check/last.ckpt"
OUTPUT = ROOT / "runs/resume_check"


class ResumeCheck(Callback):
    def __init__(self, saved):
        self.saved = saved
        self.count = 0
        self.loss_ok = None

    def on_train_start(self, trainer, pl_module):
        expected_step = self.saved["global_step"]
        assert trainer.global_step == expected_step, (
            trainer.global_step, expected_step
        )

        current = trainer.optimizers[0].state_dict()
        previous = self.saved["optimizer_states"][0]

        current_lrs = [g["lr"] for g in current["param_groups"]]
        saved_lrs = [g["lr"] for g in previous["param_groups"]]
        assert current_lrs == saved_lrs, "恢复后的学习率不一致"

        def steps(state):
            result = {}
            for key, value in state["state"].items():
                step = value.get("step")
                if step is not None:
                    result[key] = (
                        step.item() if torch.is_tensor(step) else step
                    )
            return result

        assert steps(current), "恢复后的优化器没有step状态"
        assert steps(current) == steps(previous), "优化器step状态不一致"

        schedulers = trainer.lr_scheduler_configs
        assert len(schedulers) == len(self.saved["lr_schedulers"])
        for config, saved_state in zip(
            schedulers, self.saved["lr_schedulers"]
        ):
            state = config.scheduler.state_dict()
            for key in ("last_epoch", "_step_count", "base_lrs", "_last_lr"):
                if key in saved_state:
                    assert state.get(key) == saved_state[key], (
                        f"调度器状态不一致：{key}"
                    )

        print(
            "RESUME_STATE_OK",
            f"epoch={trainer.current_epoch}",
            f"global_step={trainer.global_step}",
            flush=True,
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        loss = outputs["loss"]
        finite = torch.isfinite(loss.detach()).all()
        self.loss_ok = finite if self.loss_ok is None else self.loss_ok & finite
        self.count += 1

        if self.count == 10:
            assert bool(self.loss_ok.item()), "恢复后出现非有限loss"
            assert trainer.global_step == self.saved["global_step"] + 10
            print(
                "RESUME_10_BATCHES_OK",
                f"global_step={trainer.global_step}",
                f"loss={loss.detach().item():.5f}",
                flush=True,
            )
            trainer.should_stop = True


def main():
    # 仅加载本次运行自己生成的可信checkpoint。
    saved = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    probe = ResumeCheck(saved)

    seed_everything(42, workers=True)
    torch.set_num_threads(4)
    module = importlib.import_module("rfdetr.training.trainer")

    def bounded_trainer(*args, **kwargs):
        callbacks = kwargs.get("callbacks") or []
        # 去掉所有checkpoint回调，保留EMA等训练回调。
        kwargs["callbacks"] = [
            c for c in callbacks if not isinstance(c, ModelCheckpoint)
        ] + [probe]
        kwargs.update(
            enable_checkpointing=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
        )
        return Trainer(*args, **kwargs)

    model = RFDETRSmall(
        pretrain_weights="/root/autodl-tmp/models/roboflow/rf-detr-small.pth",
        device="cuda",
        gradient_checkpointing=False,
    )
    with patch.object(module, "Trainer", bounded_trainer):
        model.train(
            dataset_file="coco",
            dataset_dir=str(ROOT / "datasets/coco"),
            output_dir=str(OUTPUT),
            resume=str(CHECKPOINT),
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
            early_stopping=False,
            run_test=False,
            save_dataset_grids=False,
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
            progress_bar=None,
        )

    assert probe.count == 10, "未完成预期的10个batch"
    print("RESUME_CHECK_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
