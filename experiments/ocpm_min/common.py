"""Shared configuration, callbacks and metric collection for the OCPM A/B/C run.

Every entry point is expected to be launched from the project root:

    python experiments/ocpm_min/run_condition.py --condition B --output ...

This module only depends on the installed ``rfdetr`` package and the repository's
existing experiment tooling.  It never mutates ``rf-detr/src`` or ``rf-detr/tests``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# --- Repository paths (the remote project root is the historical AutoDL layout) ---
ROOT = Path("/root/autodl-tmp/project_marn-detr")
REPO = ROOT / "rf-detr"
BASELINE_CONFIG = ROOT / "runs/baseline_10ep_seed42/resolved_train_config.json"
DATASET_DIR = ROOT / "datasets/coco"
WEIGHTS = Path("/root/autodl-tmp/models/roboflow/rf-detr-small.pth")

WEIGHT_MD5 = "fb37061c1af7bace359c91b723a8d5c1"
SEED = 42
TRAIN_IMAGES = 118287
VAL_IMAGES = 5000
NUM_CLASSES = 90  # RF-DETR Small head width; COCO has 80 sparse raw category ids.
QUALITY_ETA = 0.5  # frozen for B/C so Q_write is measured instead of unknown
SLOTS_PER_CLASS = 4
QUERIES_PER_GROUP = 30
DUPLICATE_COSINE = 0.95


def _add_repo_src_to_path() -> None:
    """Make ``import rfdetr`` resolve to this checkout without installing it."""
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


_add_repo_src_to_path()


# --------------------------------------------------------------------------- io


def atomic_json(path: str | Path, value: Any) -> None:
    """Write JSON via a temporary file so a crash never leaves half a document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows to a JSONL file and return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def file_md5(path: str | Path) -> str:
    """Return the MD5 of a file without loading it in one piece."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of_parameters(module: Any) -> str:
    """Hash parameters and buffers in name order; used for checkpoint equality checks."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------- configurations


def load_baseline_config() -> dict[str, Any]:
    """Read the frozen 10-epoch baseline configuration that defines the shared protocol."""
    if not BASELINE_CONFIG.is_file():
        raise FileNotFoundError(f"baseline configuration missing: {BASELINE_CONFIG}")
    values = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    if values["seed"] != SEED:
        raise ValueError(f"baseline seed is {values['seed']}, expected {SEED}")
    return values


# Only step count / evaluation / data-order related fields are overridden.  Every
# optimiser, augmentation and EMA knob is inherited from the frozen baseline file.
_FIXED_OVERRIDES: dict[str, Any] = {
    # 3 epochs == 3 x 7393 = 22179 optimisation steps per condition.
    "epochs": 3,
    "seed": SEED,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "eval_batch_size": 8,
    "num_workers": 8,
    "amp_dtype": "bf16",
    "use_ema": True,
    "eval_max_dets": 100,
    "eval_interval": 1,
    "tensorboard": False,
    "wandb": False,
    "mlflow": False,
    "clearml": False,
    "lr": 1e-5,
    "lr_encoder": 1.5e-5,
    "warmup_epochs": 0.5,
    "dataset_dir": str(DATASET_DIR),
    "log_per_class_metrics": True,
    "num_sanity_val_steps": 0,
    "run_test": False,
    "save_dataset_grids": False,
    "progress_bar": None,
    "early_stopping": False,
}


def build_train_config(output_dir: str | Path, **overrides: Any) -> Any:
    """Build the shared ``TrainConfig``; A, B and C must receive an identical object."""
    from rfdetr.config import TrainConfig

    values = load_baseline_config()
    values.update(_FIXED_OVERRIDES)
    values["output_dir"] = str(output_dir)
    unknown = set(overrides) - set(values)
    if unknown:
        raise KeyError(f"unknown TrainConfig override(s): {sorted(unknown)}")
    values.update(overrides)
    return TrainConfig(**values)


def build_model_config() -> Any:
    """Small RF-DETR architecture identical for all three conditions."""
    from rfdetr.config import RFDETRSmallConfig

    return RFDETRSmallConfig(
        pretrain_weights=str(WEIGHTS),
        num_classes=NUM_CLASSES,
        resolution=512,
        device="cuda",
        compile=False,
        cuda_graphs=False,
        gradient_checkpointing=False,
    )


def build_marn_config(
    quality_eta: float | None = QUALITY_ETA,
    queries_per_group: int | None = None,
) -> Any:
    """Frozen OCPM configuration shared by conditions B and C.

    ``queries_per_group`` defaults to the frozen protocol value; the coverage
    ablation passes a larger value so every query can be refined.
    """
    from rfdetr.marn.integration import MARNConfig

    return MARNConfig(
        layer=0,
        slots_per_class=SLOTS_PER_CLASS,
        queries_per_group=QUERIES_PER_GROUP if queries_per_group is None else int(queries_per_group),
        quality_eta=quality_eta,
    )


def assert_weights() -> str:
    """Refuse to start unless the official checkpoint is present with the frozen MD5."""
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"pretrained weights missing: {WEIGHTS}")
    digest = file_md5(WEIGHTS)
    if digest != WEIGHT_MD5:
        raise ValueError(f"weights MD5 {digest} != expected {WEIGHT_MD5}")
    return digest


# ---------------------------------------------------------------------- data setup


def seed_everything_fixed(seed: int = SEED) -> None:
    """Seed python/numpy/torch and every DataLoader worker from one call."""
    import torch
    from pytorch_lightning import seed_everything

    torch.set_num_threads(4)
    seed_everything(seed, workers=True)


def build_datamodule(
    model_config: Any,
    train_config: Any,
    expected_train: int = TRAIN_IMAGES,
    expected_val: int = VAL_IMAGES,
) -> Any:
    """Build the COCO data module used by every condition and assert its size contract.

    ``expected_train`` defaults to the full COCO train split; the low-data protocol
    passes the size of its own subset so the contract still guards a typo.
    """
    from rfdetr.marn.data import MARNDataModule

    datamodule = MARNDataModule(model_config, train_config)
    datamodule.setup("fit")
    train_size = len(datamodule._dataset_train)
    val_size = len(datamodule._dataset_val)
    if train_size != int(expected_train):
        raise AssertionError(f"train dataset has {train_size} images, expected {expected_train}")
    if val_size != int(expected_val):
        raise AssertionError(f"val dataset has {val_size} images, expected {expected_val}")
    return datamodule


def class_ids(datamodule: Any) -> list[int]:
    """Sorted raw COCO category ids taken from the training annotation file."""
    return sorted(int(value) for value in datamodule._dataset_train.coco.getCatIds())


# ------------------------------------------------------------- data order proof


def _batch_image_ids(targets: Any) -> list[int]:
    """Extract ``image_id`` values from a collated target (packed or plain)."""
    fields = getattr(targets, "fields", None)
    if isinstance(fields, dict):
        ids = fields.get("image_id")
        if ids is not None:
            return sorted({int(v) for v in ids.reshape(-1).tolist()})
        return []
    if isinstance(targets, list):
        values: set[int] = set()
        for item in targets:
            if isinstance(item, dict) and "image_id" in item:
                values.update(int(v) for v in item["image_id"].reshape(-1).tolist())
        return sorted(values)
    return []


class DataOrderProbe:
    """Record the image ids of the first batches to prove the three runs share an order."""

    def __init__(self, batches: int = 3) -> None:
        self.batches = batches
        self.records: list[dict[str, Any]] = []
        self.fingerprint: str | None = None

    @staticmethod
    def _fallback_signature(batch: Any) -> str:
        """Hash the raw batch when packed targets expose no image ids."""
        digest = hashlib.sha256()
        targets = batch[1]
        fields = getattr(targets, "fields", None)
        tensors = list(fields.values()) if isinstance(fields, dict) else []
        if not tensors and isinstance(targets, list):
            for item in targets:
                if isinstance(item, dict):
                    tensors.extend(value for value in item.values() if hasattr(value, "numpy"))
        for tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), str(value.dtype))).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        if batch_idx >= self.batches or self.fingerprint is not None:
            return
        ids = _batch_image_ids(batch[1])
        self.records.append(
            {"batch_idx": int(batch_idx), "image_ids": ids or None, "batch_sha256": None if ids else self._fallback_signature(batch)}
        )
        if len(self.records) == self.batches:
            payload = json.dumps(self.records, sort_keys=True).encode("utf-8")
            self.fingerprint = hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------- evaluation hooks


def _as_flat_list(value: Any) -> list[float]:
    """Normalise a tensor/list metric component into a flat python list."""
    if hasattr(value, "reshape"):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _build_capture_callback(tc: Any, output_dir: Path) -> Any:
    """Create a COCO callback that keeps the native evaluator and captures every number."""
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

    class CaptureCOCO(COCOEvalCallback):
        """Stock COCO evaluator with the raw result dictionaries retained per source."""

        def __init__(self) -> None:
            super().__init__(
                max_dets=tc.eval_max_dets,
                segmentation=False,
                eval_interval=1,
                log_per_class_metrics=True,
                eval_base_model=tc.eval_base_model,
                eval_backend=tc.eval_backend,
            )
            self.captured: dict[str, dict[str, Any]] = {}
            self.per_class: dict[str, dict[int, float]] = {}

        def _compute_map_metric(self, trainer: Any, metric: Any) -> dict[str, Any]:
            result = super()._compute_map_metric(trainer, metric)
            source = "ema" if metric is self.map_metric_ema else "regular"
            names = {
                "map": "AP",
                "map_50": "AP50",
                "map_75": "AP75",
                "map_small": "APs",
                "map_medium": "APm",
                "map_large": "APl",
            }
            values = {out: float(result[key]) for key, out in names.items() if key in result}
            if len(values) != len(names):
                raise AssertionError(f"evaluator returned {sorted(result)} for {source}")
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError(f"non-finite COCO metric in {source}: {values}")
            self.captured[source] = values

            class_map: dict[int, float] = {}
            if "map_per_class" in result and "classes" in result:
                per_class = _as_flat_list(result["map_per_class"])
                categories = _as_flat_list(result["classes"])
                for category, ap in zip(categories, per_class):
                    class_map[int(category)] = float(ap)
            self.per_class[source] = class_map
            atomic_json(
                output_dir / f"coco_epoch_{trainer.current_epoch + 1:03d}_{source}.json",
                {
                    "epoch": int(trainer.current_epoch + 1),
                    "global_step": int(trainer.global_step),
                    "source": source,
                    "metrics": values,
                    "max_dets": int(tc.eval_max_dets),
                    "backend": self._eval_backend,
                },
            )
            return result

    return CaptureCOCO()


def _callback_base() -> type:
    """Return ``pytorch_lightning.Callback`` without importing Lightning at module import."""
    from pytorch_lightning import Callback

    return Callback


def make_callback(**hooks: Any) -> Any:
    """Wrap plain hook functions in a real ``Lightning.Callback`` instance.

    Lightning stores callback state dicts and inspects ``isinstance(x, Callback)``
    in several places, so probes stay plain objects and are adapted here.
    """
    base = _callback_base()
    namespace = {name: staticmethod(function) for name, function in hooks.items()}
    return type("_ProbeCallback", (base,), namespace)()


def configure_training(
    trainer: Any,
    train_config: Any,
    output_dir: str | Path,
    extra_callbacks: Iterable[Any] = (),
    *,
    with_checkpoint: bool = True,
) -> tuple[Any, Any]:
    """Swap the stock EMA/metric/checkpoint callbacks for MARN-safe equivalents.

    Mirrors ``experiments/marn_c4/formal_tools.configure_trainer`` but keeps every
    probe callback the caller supplies and never writes outside ``output_dir``.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.marn.lifecycle import MARNEMACallback
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
    from rfdetr.training.callbacks.ema import RFDETREMACallback

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    kept = [
        callback
        for callback in trainer.callbacks
        if not isinstance(callback, (ModelCheckpoint, RFDETREMACallback, COCOEvalCallback))
    ]
    ema = MARNEMACallback(decay=train_config.ema_decay, tau=train_config.ema_tau)
    capture = _build_capture_callback(train_config, folder)
    callbacks = kept + [ema, capture, *extra_callbacks]
    if with_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(folder),
                filename="last",
                monitor=None,
                save_top_k=0,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            )
        )
    trainer.callbacks = callbacks
    return ema, capture


def evaluate_coco(model_config: Any, train_config: Any, module: Any, datamodule: Any) -> dict[str, Any]:
    """Run the repository's native COCO evaluator on the full val2017 split."""
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
    from rfdetr.training.trainer import build_trainer

    output_dir = Path(train_config.output_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(train_config, model_config, include_training_callbacks=False)
    capture = _build_capture_callback(train_config, output_dir)
    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, COCOEvalCallback)] + [capture]
    trainer.validate(module, datamodule=datamodule)

    source = "regular"
    if source not in capture.captured:
        raise AssertionError(f"evaluator produced no metrics: {sorted(capture.captured)}")
    metrics = dict(capture.captured[source])
    class_map = {int(k): float(v) for k, v in capture.per_class.get(source, {}).items()}
    metrics["class_AP"] = {
        str(category): (value if math.isfinite(value) and value >= 0 else None)
        for category, value in sorted(class_map.items())
    }
    metrics["max_dets"] = int(train_config.eval_max_dets)
    metrics["eval_backend"] = str(train_config.eval_backend)
    return metrics


# ------------------------------------------------------------------ bank metrics


def bank_snapshot_stats(adapter: Any) -> dict[str, Any]:
    """Describe the prototype bank without inventing values for empty slots."""
    import torch
    import torch.nn.functional as F

    bank = adapter.bank
    valid = bank.valid.bool()
    total = int(valid.numel())
    filled = int(valid.sum())
    step = int(adapter.committed_step)

    classes = bank.classes[valid]
    covered = sorted({int(value) for value in classes.tolist()})

    duplicate_pairs = 0
    comparable_pairs = 0
    for category in covered:
        rows = (bank.classes == category) & valid
        features = bank.features[rows].float()
        if features.shape[0] < 2:
            continue
        normalized = F.normalize(features, dim=-1, eps=1e-6)
        similarity = normalized @ normalized.T
        upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        counts = similarity[upper]
        comparable_pairs += int(counts.numel())
        duplicate_pairs += int((counts > DUPLICATE_COSINE).sum())

    ages = (step - bank.updated_step[valid]).clamp_min(0).float()
    if ages.numel():
        quantiles = torch.quantile(ages, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist()
    else:
        quantiles = [None] * 5
    counts = bank.update_count[valid]
    reliability_known = int(bank.reliability_known[valid].sum())
    last_commit = dict(getattr(adapter, "last_commit", {}) or {})
    evictions = last_commit.get("evictions", [])
    reasons: dict[str, int] = {}
    for row in evictions:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1

    return {
        "step": step,
        "slots_total": total,
        "slots_filled": filled,
        "fill_rate": filled / total if total else 0.0,
        "classes_covered": len(covered),
        "class_coverage": len(covered) / 80.0,
        "duplicate_pair_count": duplicate_pairs,
        "comparable_pair_count": comparable_pairs,
        "duplicate_rate": duplicate_pairs / comparable_pairs if comparable_pairs else None,
        "age_quantiles": {
            "p0": quantiles[0],
            "p25": quantiles[1],
            "p50": quantiles[2],
            "p75": quantiles[3],
            "p100": quantiles[4],
        },
        "update_count_sum": int(counts.sum()),
        "slots_updated": int((counts > 1).sum()),
        "reliability_known_slots": reliability_known,
        "write_quality_known_slots": int(torch.isfinite(bank.write_quality[valid]).sum()),
        "unknown_mask_set_slots": int((bank.unknown_mask[valid] != 0).sum()),
        "last_step_operations": {
            key: int(last_commit.get(key, 0)) for key in ("fill", "update", "replace")
        },
        "eviction_reasons": reasons,
    }


def read_records(adapter: Any, *, condition: str, step: int) -> list[dict[str, Any]]:
    """Flatten ``PrototypeReader.last_read_records`` into JSONL rows."""
    last = dict(getattr(adapter, "last_read", {}) or {})
    rows: list[dict[str, Any]] = []
    for record in last.get("records", []):
        rows.append(
            {
                "condition": condition,
                "step": int(step),
                "batch_index": record.get("batch_index"),
                "query_index": record.get("query_index"),
                "prototype_index": record.get("prototype_index"),
                "class_compatibility": record.get("class_compatibility"),
                "geometry_compatibility": record.get("geometry_compatibility"),
                "similarity": record.get("similarity"),
                "age": record.get("age"),
                "prototype_version": record.get("prototype_version"),
                "accepted": bool(record.get("accepted")),
                "actual_correction": bool(record.get("actual_correction")),
            }
        )
    return rows


def quality_stats(adapter: Any) -> dict[str, Any]:
    """Snapshot the adapter's quality accumulators plus the persisted reliability columns."""
    import torch

    from rfdetr.marn.integration import _quality_stats_snapshot

    bank = adapter.bank
    valid = bank.valid.bool()
    components = _quality_stats_snapshot(adapter.quality_stats)
    source = "live_accumulator"
    if not components.get("Q_write", {}).get("count"):
        last_commit = getattr(adapter, "last_commit", {}) or {}
        persisted = last_commit.get("quality_components")
        if persisted:
            components = persisted
            source = "last_commit"
    payload = {
        "components": components,
        "components_source": source,
        "stats": {key: int(value) for key, value in adapter.stats.items()},
        "quality_eta": adapter.config.quality_eta,
        "bank_conf_known": int(torch.isfinite(bank.conf[valid]).sum()),
        "bank_iou_known": int(torch.isfinite(bank.iou[valid]).sum()),
        "bank_write_quality_known": int(torch.isfinite(bank.write_quality[valid]).sum()),
        "bank_feature_layer_values": sorted({int(v) for v in bank.feature_layer[valid].tolist()}),
        "bank_match_source_values": sorted({int(v) for v in bank.match_source[valid].tolist()}),
        "unknown_mask_values": sorted({int(v) for v in bank.unknown_mask[valid].tolist()}),
        "nonfinite_conf": int((~torch.isfinite(bank.conf[valid])).sum()),
        "nonfinite_iou": int((~torch.isfinite(bank.iou[valid])).sum()),
    }
    if valid.any():
        conf = bank.conf[valid].float()
        iou = bank.iou[valid].float()
        payload["conf_quantiles"] = _quantiles(conf)
        payload["iou_quantiles"] = _quantiles(iou)
    else:
        payload["conf_quantiles"] = {}
        payload["iou_quantiles"] = {}
    return payload


def _quantiles(values: Any) -> dict[str, float | None]:
    import torch

    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {}
    quantiles = torch.quantile(finite, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist()
    return dict(zip(("p0", "p25", "p50", "p75", "p100"), quantiles, strict=True))


# --------------------------------------------------------------------- resources


class ResourceAudit:
    """Track wall time and peak GPU memory for one run."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.batch_seconds: list[float] = []
        self._batch_started: float | None = None

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        self._batch_started = time.monotonic()

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._batch_started is not None:
            self.batch_seconds.append(time.monotonic() - self._batch_started)

    def summary(self, train_config: Any, output_dir: str | Path) -> dict[str, Any]:
        import torch

        elapsed = time.monotonic() - self.started
        batches = sorted(self.batch_seconds)
        median = batches[len(batches) // 2] if batches else None
        checkpoints = sorted(Path(output_dir).glob("*.ckpt")) + sorted(Path(output_dir).glob("*.pth"))
        return {
            "wall_seconds": elapsed,
            "train_batches_timed": len(batches),
            "batch_seconds_mean": sum(batches) / len(batches) if batches else None,
            "batch_seconds_median": median,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
            "epochs": int(train_config.epochs),
            "batch_size": int(train_config.batch_size),
            "grad_accum_steps": int(train_config.grad_accum_steps),
            "checkpoint_files": [
                {"name": path.name, "bytes": path.stat().st_size} for path in checkpoints
            ],
        }
