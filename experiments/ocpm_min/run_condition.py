"""Run one controlled OCPM condition (A: no OCPM, B: random read, C: random read + new fields).

    python experiments/ocpm_min/run_condition.py --condition B --output /root/.../B

All three conditions share the initial checkpoint, the seed, the data order, the
batch/accumulation/AMP/EMA settings and the evaluation function.  The only
difference between B and C is the assertion that the extra provenance columns and
``write_quality`` separation are actually in force; both use the same deterministic
random prototype read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    """Parse the condition selector and run controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("A", "B", "C"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="training epochs; must be identical across A/B/C (default 3)",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="0 means a full epoch; any positive value caps training batches",
    )
    parser.add_argument("--quality-eta", type=float, default=common.QUALITY_ETA)
    parser.add_argument("--lr", type=float, default=0.0, help="override the base lr (0 keeps the protocol)")
    parser.add_argument("--rho", type=float, default=0.0, help="MGQR residual budget as a fraction of the query norm (0 keeps 0.1)")
    parser.add_argument("--beta-max", type=float, default=0.0, help="MGQR gate ceiling (0 keeps 0.5)")
    parser.add_argument("--queries-per-group", type=int, default=0, help="queries refined per group (0 keeps 30; 300 covers every query)")
    parser.add_argument("--lr-encoder", type=float, default=0.0, help="override the encoder lr")
    parser.add_argument("--dataset-dir", default="", help="override the COCO root (low-data subset)")
    parser.add_argument("--expect-train-images", type=int, default=0, help="expected train size of that root")
    parser.add_argument(
        "--read-policy",
        choices=("random", "similarity"),
        default="random",
        help="random = deterministic valid-slot draw (B/C); similarity = the repository reader (D)",
    )
    return parser.parse_args()


MAX_READ_RECORDS_PER_STEP = 200


def install_random_reader(adapter, seed: int) -> None:
    """Replace learned retrieval with a deterministic random pick over valid slots.

    The pick is fully vectorised: one device sync per step instead of one per
    selected query.  Cosine / class / geometry never choose the slot; they are
    only recorded, for a bounded sample of selections, as diagnostics.
    """
    import torch
    import torch.nn.functional as F

    base = adapter.reader
    config = base.config

    def forward(queries, logits, boxes, selected, snapshot):  # noqa: D401 - nn.Module hook
        base.last_read_records = []
        memory = torch.zeros_like(queries)
        summary = queries.new_zeros((*queries.shape[:2], 3), dtype=torch.float32)
        available = torch.zeros_like(selected)
        bi, qi = selected.nonzero(as_tuple=True)
        n = int(bi.numel())
        if n == 0:
            return memory, summary, available
        empty = dict.fromkeys(
            ("prototype_index", "class_compatibility", "geometry_compatibility", "similarity",
             "age", "prototype_version", "updated_step", "feature_layer", "match_source", "unknown_mask"),
            None,
        )
        empty.update(accepted=False, actual_correction=False)
        finite = torch.isfinite(snapshot.features).all(-1) & torch.isfinite(snapshot.geometry).all(-1)
        valid_index = (snapshot.valid & finite).nonzero().flatten()
        if valid_index.numel() == 0:
            for b, q in zip(bi.tolist(), qi.tolist()):
                base.last_read_records.append({"batch_index": int(b), "query_index": int(q), **empty})
            return memory, summary, available

        device = queries.device
        step = int(adapter.committed_step)
        generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + step)
        positions = torch.randint(int(valid_index.numel()), (n,), generator=generator)
        slots = valid_index.cpu()[positions].to(device)
        memory[bi, qi] = snapshot.features[slots].float()
        available[bi, qi] = True

        stride = max(1, n // MAX_READ_RECORDS_PER_STEP)
        rows = torch.arange(0, n, stride, device=device)
        b_sel, q_sel, s_sel = bi[rows], qi[rows], slots[rows]
        feats = snapshot.features[s_sel].float()
        qfeat = queries[b_sel, q_sel].float()
        qkeys = F.normalize(base.key(qfeat).float(), dim=-1, eps=1e-6)
        mkeys = F.normalize(base.key(feats).float(), dim=-1, eps=1e-6)
        similarity = (qkeys * mkeys).sum(-1)
        probs = logits[b_sel, q_sel].float().sigmoid().index_select(-1, base.class_ids)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-6)
        compact = base.class_index[snapshot.classes[s_sel]].clamp_min(0)
        class_compat = probs.gather(1, compact[:, None]).squeeze(1)
        qwh = boxes[b_sel, q_sel, 2:].float().clamp_min(1e-6).log()
        mwh = snapshot.geometry[s_sel, 2:].float().clamp_min(1e-6).log()
        geometry_score = -(qwh - mwh).abs().mean(-1)
        ages = (int(snapshot.step) - snapshot.updated_step[s_sel]).clamp_min(0)

        b_list = b_sel.tolist()
        q_list = q_sel.tolist()
        s_list = s_sel.tolist()
        sim_list = similarity.tolist()
        cc_list = class_compat.tolist()
        geo_list = geometry_score.tolist()
        age_list = ages.tolist()
        ver_list = snapshot.update_count[s_sel].tolist()
        upd_list = snapshot.updated_step[s_sel].tolist()
        lay_list = snapshot.feature_layer[s_sel].tolist()
        src_list = snapshot.match_source[s_sel].tolist()
        msk_list = snapshot.unknown_mask[s_sel].tolist()
        for row in range(len(b_list)):
            base.last_read_records.append(
                {
                    "batch_index": int(b_list[row]),
                    "query_index": int(q_list[row]),
                    "prototype_index": int(s_list[row]),
                    "class_compatibility": float(cc_list[row]),
                    "geometry_compatibility": float(geo_list[row]),
                    "similarity": float(sim_list[row]),
                    "age": int(age_list[row]),
                    "prototype_version": int(ver_list[row]),
                    "updated_step": int(upd_list[row]),
                    "feature_layer": int(lay_list[row]),
                    "match_source": int(src_list[row]),
                    "unknown_mask": int(msk_list[row]),
                    "accepted": True,
                    "actual_correction": False,
                }
            )
        _ = config  # configuration is inherited, not re-created
        return memory, summary, available

    base.forward = forward


def assert_write_quality_is_not_maintenance() -> None:
    """Static guard: Q_write must not enter the eviction ordering or the fallback score."""
    source = (common.REPO / "src/rfdetr/marn/integration.py").read_text(encoding="utf-8")
    if "quality_fallback" in source:
        raise AssertionError("integration.py contains a quality fallback hook")
    if "_quality_diag, write_quality" in source:
        raise AssertionError("integration.py binds quality_diag to write_quality")
    marker = "keys.append((known, conf_value, iou_value,"
    if marker not in source:
        raise AssertionError("eviction key construction was not found; maintenance ordering changed")
    block = source.split(marker, 1)[1].split("))", 1)[0]
    for forbidden in ("write_quality", "quality_diag", "redundancy_only"):
        if forbidden in block:
            raise AssertionError(f"eviction ordering consumes {forbidden}")


class TrainingProbe:
    """Loss finiteness, bank commit timing and cumulative mutation counters."""

    def __init__(self, adapter) -> None:
        import torch

        self.adapter = adapter
        self.torch = torch
        self.finite = True
        self.batches = 0
        self.commit_lag = 0
        self.operations = {"fill": 0, "update": 0, "replace": 0}
        self.eviction_reasons: dict[str, int] = {}
        self.unknown_quality_steps = 0
        self.quality_steps = 0
        self.quality_components: dict[str, dict] = {}

    def _merge_quality(self, components) -> None:
        """Accumulate the per-step Q_write component statistics stored in last_commit."""
        if not components:
            return
        self.quality_steps += 1
        for name, item in components.items():
            dst = self.quality_components.setdefault(
                name, {"count": 0, "finite": 0, "unknown": 0, "sum": 0.0, "min": None, "max": None}
            )
            dst["count"] += int(item.get("count", 0) or 0)
            dst["finite"] += int(item.get("finite", 0) or 0)
            dst["unknown"] += int(item.get("unknown", 0) or 0)
            dst["sum"] += float(item.get("sum", 0.0) or 0.0)
            low, high = item.get("min"), item.get("max")
            if low is not None:
                dst["min"] = low if dst["min"] is None else min(dst["min"], low)
            if high is not None:
                dst["max"] = high if dst["max"] is None else max(dst["max"], high)
        for dst in self.quality_components.values():
            dst["mean"] = (dst["sum"] / dst["finite"]) if dst["finite"] else None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, self.torch.Tensor):
            self.finite = self.finite and bool(self.torch.isfinite(loss.detach()).all().item())
        self.batches += 1
        if int(self.adapter.committed_step) != int(trainer.global_step):
            self.commit_lag += 1
        last = dict(getattr(self.adapter, "last_commit", {}) or {})
        if last.get("step") == int(trainer.global_step):
            for key in self.operations:
                self.operations[key] += int(last.get(key, 0))
            for row in last.get("evictions", []):
                reason = row["reason"]
                self.eviction_reasons[reason] = self.eviction_reasons.get(reason, 0) + 1
            self._merge_quality(last.get("quality_components"))


class ReadRecorder:
    """Stream ``PrototypeReader.last_read_records`` into a JSONL audit file."""

    def __init__(self, adapter, condition: str, path: Path) -> None:
        self.adapter = adapter
        self.condition = condition
        self.path = path
        self.rows = 0

    RECORD_EVERY = 25

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step % self.RECORD_EVERY:
            return
        rows = common.read_records(self.adapter, condition=self.condition, step=step)
        if rows:
            self.rows += common.append_jsonl(self.path, rows)


def build_module(condition: str, model_config, train_config, categories, quality_eta, queries_per_group: int = 0):
    """Construct the LightningModule for one condition."""
    from rfdetr.marn.lifecycle import MARNTrainingModule
    from rfdetr.training.module_model import RFDETRModelModule

    if condition == "A":
        return RFDETRModelModule(model_config, train_config, load_encoder_weights=False), None
    module = MARNTrainingModule(
        model_config,
        train_config,
        class_ids=categories,
        marn_config=common.build_marn_config(quality_eta, queries_per_group or None),
    )
    return module, module.adapter


def main() -> int:
    """Run one condition end to end and write its five artifacts."""
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    condition = args.condition

    def log(message: str) -> None:
        print(message, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    summary: dict[str, object] = {"condition": condition, "passed": False}
    try:
        import torch
        from rfdetr.training.trainer import build_trainer

        common.seed_everything_fixed(args.seed)
        weight_md5 = common.assert_weights()
        model_config = common.build_model_config()
        config_overrides = {}
        if args.epochs:
            config_overrides["epochs"] = int(args.epochs)
        if args.dataset_dir:
            config_overrides["dataset_dir"] = args.dataset_dir
        if args.lr:
            config_overrides["lr"] = float(args.lr)
        if args.lr_encoder:
            config_overrides["lr_encoder"] = float(args.lr_encoder)
        train_config = common.build_train_config(output, **config_overrides)
        # ``output_dir`` is the only field that legitimately differs between A/B/C.
        comparable_config = train_config.model_dump(mode="json")
        comparable_config.pop("output_dir", None)
        train_hash = hashlib.sha256(
            json.dumps(comparable_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if condition == "C":
            assert_write_quality_is_not_maintenance()

        datamodule = common.build_datamodule(
            model_config,
            train_config,
            expected_train=args.expect_train_images or common.TRAIN_IMAGES,
        )
        categories = common.class_ids(datamodule)
        module, adapter = build_module(
            condition, model_config, train_config, categories, args.quality_eta, args.queries_per_group
        )
        if adapter is not None:
            if args.rho:
                adapter.refiner.rho = float(args.rho)
            if args.beta_max:
                adapter.refiner.beta_max = float(args.beta_max)
            if args.rho or args.beta_max:
                print(f"MGQR bounds: rho={adapter.refiner.rho} beta_max={adapter.refiner.beta_max}", flush=True)
        if adapter is not None and args.read_policy == "random":
            install_random_reader(adapter, args.seed)

        order = common.DataOrderProbe(3)
        resources = common.ResourceAudit()
        extras: list[object] = [
            common.make_callback(on_train_batch_start=order.on_train_batch_start),
            common.make_callback(
                on_train_batch_start=resources.on_train_batch_start,
                on_train_batch_end=resources.on_train_batch_end,
            ),
        ]
        probe = None
        recorder = None
        if adapter is not None:
            probe = TrainingProbe(adapter)
            recorder = ReadRecorder(adapter, condition, output / f"read_records_{condition}.jsonl")
            extras.append(common.make_callback(on_train_batch_end=probe.on_train_batch_end))
            extras.append(common.make_callback(on_train_batch_end=recorder.on_train_batch_end))
            if recorder.path.exists():
                recorder.path.unlink()

        trainer_kwargs: dict[str, object] = {"enable_model_summary": False}
        if args.limit_batches > 0:
            trainer_kwargs["limit_train_batches"] = args.limit_batches
        trainer = build_trainer(train_config, model_config, **trainer_kwargs)
        if adapter is None:
            trainer.callbacks = list(trainer.callbacks) + extras
        else:
            common.configure_training(trainer, train_config, output, extra_callbacks=extras, with_checkpoint=False)

        torch.cuda.reset_peak_memory_stats()
        log(f"{condition}_TRAIN_START output={output} seed={args.seed} limit_batches={args.limit_batches}")
        trainer.fit(module, datamodule=datamodule)
        if probe is not None and not probe.finite:
            raise FloatingPointError("non-finite training loss")
        if probe is not None and probe.commit_lag:
            raise AssertionError(f"bank commit lagged the optimizer step {probe.commit_lag} times")
        global_step = int(trainer.global_step)
        log(f"{condition}_TRAIN_DONE global_step={global_step} batches={trainer.global_step}")

        trainer.save_checkpoint(str(output / f"{condition}_last.ckpt"))
        metrics = common.evaluate_coco(model_config, train_config, module, datamodule)
        log(f"{condition}_EVAL_DONE " + json.dumps({k: metrics[k] for k in ('AP', 'AP50', 'AP75')}))

        metrics_payload = {
            "condition": condition,
            "seed": int(args.seed),
            "global_step": global_step,
            "epochs": int(train_config.epochs),
            "data_order_fingerprint": order.fingerprint,
            "data_order_batches": order.records,
            "weights_md5": weight_md5,
            "train_config_sha256": train_hash,
            "read_policy": (
                "official_detector"
                if condition == "A"
                else ("deterministic_random_slot" if args.read_policy == "random" else "repository_similarity")
            ),
            "quality_eta": None if condition == "A" else float(args.quality_eta),
            "mgqr_rho": float(args.rho) if args.rho else None,
            "mgqr_beta_max": float(args.beta_max) if args.beta_max else None,
            "queries_per_group": int(args.queries_per_group) if args.queries_per_group else None,
            "metrics": metrics,
        }
        common.atomic_json(output / f"{condition}_metrics.json", metrics_payload)
        common.atomic_json(output / f"{condition}_resources.json", resources.summary(train_config, output))

        if adapter is None:
            bank = {"applicable": False, "reason": "condition A does not attach OCPM"}
            quality = {"applicable": False, "reason": "condition A has no prototype bank"}
        else:
            bank = common.bank_snapshot_stats(adapter)
            bank["cumulative_operations"] = probe.operations if probe else None
            bank["cumulative_eviction_reasons"] = probe.eviction_reasons if probe else None
            bank["read_events"] = recorder.rows if recorder else 0
            quality = common.quality_stats(adapter)
            quality["accumulated_steps"] = probe.quality_steps
            quality["accumulated_components"] = probe.quality_components
            if condition == "C":
                if bank["fill_rate"] <= 0:
                    raise AssertionError("condition C produced an empty bank; new fields cannot be verified")
                if quality["bank_write_quality_known"] == 0:
                    raise AssertionError("condition C did not persist write_quality")
                if quality["bank_feature_layer_values"] != [0]:
                    raise AssertionError(
                        f"condition C feature_layer values {quality['bank_feature_layer_values']} != [0]"
                    )
                if quality["bank_match_source_values"] != [1]:
                    raise AssertionError(
                        f"condition C match_source values {quality['bank_match_source_values']} != [1]"
                    )
        common.atomic_json(output / f"{condition}_bank.json", bank)
        common.atomic_json(output / f"{condition}_quality.json", quality)

        summary.update(
            passed=True,
            global_step=global_step,
            metrics_file=str(output / f"{condition}_metrics.json"),
        )
        common.atomic_json(output / "run_summary.json", summary)
        log(f"{condition}_PASSED")
        return 0
    except Exception as error:  # noqa: BLE001 - every failure must be reported explicitly
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        common.atomic_json(output / "run_summary.json", summary)
        log("FAILED " + summary["error"])
        log(summary["traceback"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
