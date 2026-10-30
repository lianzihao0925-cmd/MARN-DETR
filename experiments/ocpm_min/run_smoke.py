"""Tiny OCPM smoke test: GPU step, matching hand-off, bank timing, checkpoint round trip.

Run from the project root, either of:

    python experiments/ocpm_min/run_smoke.py
    python -m experiments.ocpm_min.run_smoke

Any failed assertion exits non-zero with a ``FAILED`` marker in ``smoke.log``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "rf-detr" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "rf-detr" / "src"))

from experiments.ocpm_min import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line options for the smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(common.ROOT / "ocpm_runs" / "smoke"),
        help="directory that receives smoke.log / smoke_result.json / bank_summary.json",
    )
    parser.add_argument("--limit-train-batches", type=int, default=6)
    parser.add_argument("--limit-val-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=common.SEED)
    return parser.parse_args()


class SmokeProbe:
    """Collect the evidence the smoke assertions are made of."""

    def __init__(self, adapter) -> None:
        import torch

        self.adapter = adapter
        self.torch = torch
        self.losses: list[float] = []
        self.finite_loss = True
        self.pending_before_step = 0
        self.observer_calls = 0
        self.snapshot_step_matches = True
        self.commit_after_step = True
        self.val_bank_hashes: list[str] = []
        self.optimizer_steps = 0
        self.bank_hashes_during_validation: list[tuple[str, str]] = []

    @staticmethod
    def _bank_hash(bank) -> str:
        import hashlib

        digest = hashlib.sha256()
        for name, value in sorted(bank.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, self.torch.Tensor):
            value = float(loss.detach().item())
            self.losses.append(value)
            self.finite_loss = self.finite_loss and bool(self.torch.isfinite(loss.detach()).all().item())
        if int(self.adapter.committed_step) != int(trainer.global_step):
            self.commit_after_step = False

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self.optimizer_steps += 1
        self.pending_before_step = max(self.pending_before_step, len(self.adapter.pending))
        if self.adapter.observed:
            self.observer_calls += 1
        snapshot = self.adapter.snapshot
        if snapshot is not None and int(snapshot.step) != int(self.adapter.committed_step):
            self.snapshot_step_matches = False

    def on_validation_start(self, trainer, pl_module) -> None:
        self.bank_hashes_during_validation.append(("start", self._bank_hash(self.adapter.bank)))

    def on_validation_end(self, trainer, pl_module) -> None:
        self.bank_hashes_during_validation.append(("end", self._bank_hash(self.adapter.bank)))

    def on_test_start(self, trainer, pl_module) -> None:
        self.bank_hashes_during_validation.append(("test_start", self._bank_hash(self.adapter.bank)))

    def on_test_end(self, trainer, pl_module) -> None:
        self.bank_hashes_during_validation.append(("test_end", self._bank_hash(self.adapter.bank)))


def main() -> int:
    """Run the smoke test and write its artifacts."""
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "smoke.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    results: dict[str, object] = {"passed": False, "condition": "smoke"}
    try:
        import torch
        from rfdetr.marn.lifecycle import MARNTrainingModule
        from rfdetr.training.trainer import build_trainer

        common.seed_everything_fixed(args.seed)
        weight_md5 = common.assert_weights()
        model_config = common.build_model_config()
        train_config = common.build_train_config(output, epochs=1)
        log(f"SMOKE_START output={output} seed={args.seed} weights_md5={weight_md5}")

        datamodule = common.build_datamodule(model_config, train_config)
        categories = common.class_ids(datamodule)
        if len(categories) != 80:
            raise AssertionError(f"expected 80 COCO categories, got {len(categories)}")
        module = MARNTrainingModule(
            model_config,
            train_config,
            class_ids=categories,
            marn_config=common.build_marn_config(),
        )
        adapter = module.adapter

        probe = SmokeProbe(adapter)
        order = common.DataOrderProbe(3)
        resources = common.ResourceAudit()

        trainer = build_trainer(
            train_config,
            model_config,
            enable_model_summary=False,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        probes = [
            common.make_callback(
                on_train_batch_end=probe.on_train_batch_end,
                on_before_optimizer_step=probe.on_before_optimizer_step,
                on_validation_start=probe.on_validation_start,
                on_validation_end=probe.on_validation_end,
                on_test_start=probe.on_test_start,
                on_test_end=probe.on_test_end,
            ),
            common.make_callback(on_train_batch_start=order.on_train_batch_start),
            common.make_callback(
                on_train_batch_start=resources.on_train_batch_start,
                on_train_batch_end=resources.on_train_batch_end,
            ),
        ]
        common.configure_training(
            trainer, train_config, output, extra_callbacks=probes, with_checkpoint=False
        )
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(module, datamodule=datamodule)

        # (a) GPU forward/backward produced finite losses only.
        if not probe.losses or not probe.finite_loss:
            raise AssertionError(f"non-finite or missing training loss: {probe.losses[:5]}")
        results["finite_losses"] = probe.losses

        # (b) The criterion handed final matching over to the OCPM observer.
        if probe.observer_calls == 0:
            raise AssertionError("criterion.marn_observer was never invoked")
        if probe.pending_before_step == 0:
            raise AssertionError("no pending prototype candidates before the optimizer step")
        results["observer_calls"] = probe.observer_calls
        results["pending_before_step"] = probe.pending_before_step

        # (c) Snapshot is the pre-step bank and the commit happens after the step.
        if not probe.snapshot_step_matches:
            raise AssertionError("training snapshot was not taken at the optimizer boundary")
        if not probe.commit_after_step:
            raise AssertionError("bank commit did not run after the optimizer step")
        if int(adapter.committed_step) != int(trainer.global_step):
            raise AssertionError(
                f"committed_step {int(adapter.committed_step)} != global_step {int(trainer.global_step)}"
            )
        results["optimizer_steps"] = probe.optimizer_steps
        results["global_step"] = int(trainer.global_step)
        results["committed_step"] = int(adapter.committed_step)

        # (d) Validation never mutates the bank.
        if len(probe.bank_hashes_during_validation) >= 2:
            phases = probe.bank_hashes_during_validation
            if phases[0][1] != phases[-1][1]:
                raise AssertionError(f"validation/test mutated the bank: {phases}")
            results["bank_hashes"] = phases
        else:
            raise AssertionError("validation did not run; bank invariance cannot be proven")

        # (e) Checkpoint round trip reconstructs identical parameters.
        checkpoint_path = output / "smoke_last.ckpt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        trainer.save_checkpoint(str(checkpoint_path))
        reference_hash = common.sha256_of_parameters(module)
        restored = MARNTrainingModule(
            common.build_model_config(),
            common.build_train_config(output, epochs=1),
            class_ids=categories,
            marn_config=common.build_marn_config(),
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored.load_state_dict(payload["state_dict"], strict=True)
        restored_hash = common.sha256_of_parameters(restored)
        if restored_hash != reference_hash:
            raise AssertionError("checkpoint did not reconstruct identical parameters")
        results["parameter_hash"] = reference_hash
        results["checkpoint_bytes"] = checkpoint_path.stat().st_size

        # (f) Reliability/metadata columns and quality accumulators are populated.
        quality = common.quality_stats(adapter)
        if not quality["components"]["Q_write"]["count"]:
            raise AssertionError("Q_write was never evaluated")
        if quality["bank_conf_known"] == 0 or quality["bank_iou_known"] == 0:
            raise AssertionError("bank conf/IoU were never populated")
        if quality["bank_write_quality_known"] == 0:
            raise AssertionError("bank write_quality was never populated")
        if not quality["bank_feature_layer_values"]:
            raise AssertionError("feature_layer was never persisted")
        if not quality["bank_match_source_values"]:
            raise AssertionError("match_source was never persisted")
        for name, key in (("nonfinite_conf", "conf"), ("nonfinite_iou", "iou")):
            if quality[name] > 0:
                raise AssertionError(f"non-finite {key} values entered the bank")
        results["quality"] = quality["stats"]

        bank = common.bank_snapshot_stats(adapter)
        results["bank"] = bank
        results["data_order_fingerprint"] = order.fingerprint
        results["resources"] = resources.summary(train_config, output)

        common.atomic_json(output / "bank_summary.json", bank)
        common.atomic_json(output / "quality_stats.json", quality)
        results["passed"] = True
        common.atomic_json(output / "smoke_result.json", results)
        log("SMOKE_PASSED " + str({k: results[k] for k in ('global_step', 'committed_step', 'optimizer_steps')}))
        return 0
    except Exception as error:  # noqa: BLE001 - the smoke driver must report every failure
        results["error"] = f"{type(error).__name__}: {error}"
        results["traceback"] = traceback.format_exc()
        common.atomic_json(output / "smoke_result.json", results)
        log("FAILED " + results["error"])
        log(results["traceback"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
