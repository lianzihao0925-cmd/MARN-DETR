"""C2 single-GPU eager detection integration. No routing probes or EMA yet."""
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import (
    MATCH_SOURCE_FINAL,
    UNKNOWN_CONF,
    UNKNOWN_FEATURE_LAYER,
    UNKNOWN_IOU,
    UNKNOWN_MATCH_SOURCE,
    UNKNOWN_WRITE_QUALITY,
    PrototypeBank,
    PrototypeReader,
    BoundedMGQR,
)


@dataclass(frozen=True)
class MARNConfig:
    layer: int = 0
    slots_per_class: int = 4
    queries_per_group: int = 30
    confidence_min: float = 0.2
    iou_min: float = 0.5
    similarity_update: float = 0.9
    feature_ema: float = 0.9
    warmup_steps: int = 1  # smoke-test setting; formal schedule is not yet frozen
    # Deliberately has no calibrated default.  None keeps Q_write unknown until
    # an experiment freezes eta in the run configuration.
    quality_eta: float | None = None


def _quality_stats_template() -> dict[str, dict[str, Any]]:
    return {
        name: {"count": 0, "finite": 0, "unknown": 0, "sum": 0.0,
               "min": None, "max": None}
        for name in ("R", "S", "N", "D_c", "V", "Q_write")
    }


def _record_quality_stats(stats: dict[str, dict[str, Any]], name: str, values: Tensor) -> None:
    """Accumulate compact, JSON-safe component diagnostics."""
    item = stats[name]
    values = values.detach().float().reshape(-1)
    item["count"] += int(values.numel())
    finite = torch.isfinite(values)
    item["finite"] += int(finite.sum())
    item["unknown"] += int((~finite).sum())
    if finite.any():
        valid = values[finite]
        item["sum"] += float(valid.sum())
        low, high = float(valid.min()), float(valid.max())
        item["min"] = low if item["min"] is None else min(item["min"], low)
        item["max"] = high if item["max"] is None else max(item["max"], high)


def _quality_stats_snapshot(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return JSON-safe component counts with a finite-value mean."""
    result = {}
    for name, item in stats.items():
        row = dict(item)
        row["mean"] = item["sum"] / item["finite"] if item["finite"] else None
        result[name] = row
    return result


def compute_write_quality(
    conf: Tensor,
    iou: Tensor,
    stability: Tensor,
    novelty: Tensor,
    coverage: Tensor,
    eta: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute R, V and Q_write without silently filling missing signals.

    ``stability``, ``novelty`` and ``coverage`` are expected to be normalized
    component signals.  Unknown components remain NaN and make Q_write NaN;
    callers can then record the reason without silently substituting the
    diagnostic quality for the write score.
    """
    base = conf.float()
    if eta is None or not math.isfinite(float(eta)) or not 0 <= float(eta) <= 1:
        eta_tensor = base.new_full(conf.shape, float("nan"))
    else:
        eta_tensor = base.new_full(conf.shape, float(eta))
    input_finite = (
        torch.isfinite(conf) & torch.isfinite(iou) & torch.isfinite(stability)
        & torch.isfinite(novelty) & torch.isfinite(coverage)
    )
    c = base.clamp(0, 1)
    l = iou.float().clamp(0, 1)
    r = 1 - torch.sqrt((1 - c).square() + (1 - l).square()) / math.sqrt(2.0)
    n = novelty.float().clamp(0, 1)
    d = coverage.float().clamp(0, 1)
    v = torch.sqrt((n * d).clamp(0, 1))
    q = r * stability.float() * (eta_tensor + (1 - eta_tensor) * v)
    known = (input_finite & torch.isfinite(r) & torch.isfinite(stability)
             & torch.isfinite(n) & torch.isfinite(d) & torch.isfinite(eta_tensor))
    q = torch.where(known & torch.isfinite(q), q, torch.full_like(q, float("nan")))
    return r, v, q


def paired_iou(a, b):
    a0, a1 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b0, b1 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.minimum(a1, b1) - torch.maximum(a0, b0)).clamp_min(0).prod(-1)
    return inter / (a[:, 2:].prod(-1) + b[:, 2:].prod(-1) - inter).clamp_min(1e-6)


class MARNAdapter(nn.Module):
    def __init__(self, dim, class_ids, num_queries, config=None):
        super().__init__()
        self.config = config or MARNConfig()
        self.num_queries = num_queries
        if not 0 < self.config.queries_per_group <= num_queries:
            raise ValueError("Invalid per-group query budget")
        self.bank = PrototypeBank(class_ids, dim, self.config.slots_per_class)
        self.reader = PrototypeReader(dim, class_ids)
        self.refiner = BoundedMGQR(dim)
        self.register_buffer("committed_step", torch.tensor(0, dtype=torch.long))
        self.enabled = True
        self.snapshot = None
        self.pending = {}
        self.capture = None
        self.last_commit = {}
        self.last_read = {}
        self.window_reads = 0
        self.observed = False
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            k: 0
            for k in (
                "matches", "candidate_total", "invalid", "invalid_values", "low_confidence", "low_iou",
                "duplicates", "eligible", "hard_gate_passed", "quality_unknown",
                "class_consistent", "class_inconsistent",
            )
        }
        self.class_candidates = {}
        self.class_eligible = {}
        self._conf_values = []
        self._iou_values = []
        self.quality_stats = _quality_stats_template()

    @staticmethod
    def _quantiles(values):
        """Return fixed diagnostic quantiles without introducing a gate."""
        if not values:
            return {}
        tensor = torch.as_tensor(values, dtype=torch.float32)
        probabilities = torch.as_tensor((0.0, 0.25, 0.5, 0.75, 1.0))
        quantiles = torch.quantile(tensor, probabilities)
        return {
            name: float(value)
            for name, value in zip(("p0", "p25", "p50", "p75", "p100"), quantiles.tolist(), strict=True)
        }

    @staticmethod
    def _increment_class_count(counts, labels):
        """Accumulate counts keyed by stringified raw category IDs."""
        for label in labels.detach().cpu().tolist():
            key = str(int(label))
            counts[key] = counts.get(key, 0) + 1

    @torch.no_grad()
    def _novelty_and_coverage(self, label: int, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Return bank-observable N_i and D_c for one candidate.

        A class with no established prototype, or only one valid slot, is
        treated as an uncovered class for maintenance purposes.  This keeps
        rare-class coverage eligible without pretending that within-class
        dispersion has been measured.  The resulting policy is logged through
        the component statistics and remains separate from retrieval scores.
        """
        snapshot = self.snapshot
        if snapshot is None:
            nan = feature.new_tensor(float("nan"))
            return nan, nan
        valid = snapshot.valid & torch.isfinite(snapshot.features).all(-1)
        same = valid & (snapshot.classes == int(label))
        rows = same.nonzero().flatten()
        if rows.numel() == 0:
            return feature.new_tensor(1.0), feature.new_tensor(1.0)
        bank_features = snapshot.features[rows].float()
        candidate = feature.float().unsqueeze(0)
        similarities = F.cosine_similarity(bank_features, candidate, dim=-1)
        novelty = (1 - similarities.max()).clamp(0, 1)
        if rows.numel() < 2:
            coverage = feature.new_tensor(1.0)
        else:
            normalized = F.normalize(bank_features, dim=-1, eps=1e-6)
            pairwise = normalized @ normalized.T
            upper_mask = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
            count = rows.numel() * (rows.numel() - 1) // 2
            coverage = ((1 - pairwise[upper_mask]).sum() / max(count, 1)).clamp(0, 1)
        return novelty, coverage

    @staticmethod
    def _candidate_priority(candidate: tuple[Any, ...]) -> tuple[int, float]:
        """Compare candidates using Q_write only when it is measured."""
        write_quality = candidate[1]
        if write_quality is not None and math.isfinite(float(write_quality)):
            return 1, float(write_quality)
        # Unknown Q_write values are tied deliberately.  The diagnostic
        # quality must never become an implicit write/maintenance score.
        return 0, 0.0

    @staticmethod
    def _json_float(value) -> float | None:
        """Convert a scalar to JSON-safe float metadata, preserving unknown."""
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def _eviction_reason(keys, victim: int) -> str:
        """Explain the lexicographic maintenance choice without a score sum."""
        selected = keys[victim]
        if selected[0] == 0:
            return "reliability_unknown"
        same_reliability = [key for key in keys if key[:3] == selected[:3]]
        if len(same_reliability) > 1 and selected[3] < max(key[3] for key in same_reliability):
            return "redundant"
        if len(same_reliability) > 1 and selected[4] < max(key[4] for key in same_reliability):
            return "aged"
        return "low_reliability"

    def forward(self, raw, normalized, reference, heads, bbox_reparam):
        if not self.enabled:
            return raw
        if self.capture is not None:
            raise RuntimeError("Previous training capture was not consumed by criterion")
        step = int(self.committed_step)
        if self.training:
            if self.snapshot is None:
                self.snapshot = self.bank.snapshot(step)
                self.window_reads = 0
            snapshot = self.snapshot
            self.window_reads += 1
        else:
            if self.pending or self.snapshot is not None:
                raise RuntimeError("Evaluation must start at an optimizer boundary")
            snapshot = self.bank.snapshot(step)

        # Heads are supplied by the current LWDETR instance, never registered twice.
        # Class/geometry priors are observations; new key and residual modules learn.
        with torch.no_grad():
            cls_head, box_head = heads
            logits = cls_head(normalized.detach())
            delta = box_head(normalized.detach())
            if bbox_reparam:
                boxes = torch.cat((delta[..., :2] * reference[..., 2:] + reference[..., :2],
                                   delta[..., 2:].exp() * reference[..., 2:]), -1)
            else:
                boxes = (delta + reference).sigmoid()
        if self.training:
            self.capture = raw.detach()

        b, q, _ = raw.shape
        if q % self.num_queries:
            raise ValueError("Query count is not a multiple of official group size")
        groups = q // self.num_queries
        # Random routing is isolated from detector/dropout/DataLoader RNG.
        # Each repeated training group receives the same relative query selection.
        gen = torch.Generator(device=raw.device)
        gen.manual_seed(420001 + step * 1009 + (self.window_reads if self.training else 0))
        ranks = torch.rand(b, self.num_queries, generator=gen, device=raw.device)
        idx = ranks.topk(self.config.queries_per_group, dim=-1).indices
        one = torch.zeros(b, self.num_queries, dtype=torch.bool, device=raw.device)
        one.scatter_(1, idx, True)
        selected = one[:, None].expand(-1, groups, -1).reshape(b, q)
        if step < self.config.warmup_steps:
            selected = torch.zeros_like(selected)
        memory, summary, accepted = self.reader(raw, logits, boxes, selected, snapshot)
        refined = self.refiner(raw, memory, summary, accepted)
        correction_mask = (refined.float() - raw.float()).abs().sum(-1) > 0
        for record in self.reader.last_read_records:
            if record["prototype_index"] is not None:
                record["actual_correction"] = bool(
                    correction_mask[record["batch_index"], record["query_index"]]
                )
        self.last_read = {"step": step, "selected": int(selected.sum()),
                          "accepted": int(accepted.sum()), "groups": groups,
                          "records": list(self.reader.last_read_records)}
        return refined

    @torch.no_grad()
    def observe(self, outputs, all_indices, suffixes, targets, *, final_indices=None, feature_indices=None):
        if not self.enabled or not self.training:
            return
        if self.capture is None or self.snapshot is None:
            raise RuntimeError("Missing insertion-layer capture or step snapshot")
        self.observed = True
        final_index = suffixes.index("")
        feature_index = suffixes.index(f"_{self.config.layer}")
        final_outputs = outputs[final_index]
        feature_outputs = outputs[feature_index]
        if final_indices is None:
            final_indices = all_indices[final_index]
        if feature_indices is None:
            feature_indices = [source for source, _target in final_indices]
        if len(final_indices) != len(feature_indices) or len(final_indices) != len(targets):
            raise ValueError("Final matching and layer-0 feature indices must align with targets")
        raw = self.capture
        self.capture = None
        if "pred_logits" not in feature_outputs or feature_outputs["pred_logits"].shape[:2] != raw.shape[:2]:
            raise ValueError("Layer-0 output queries must align with captured raw features")
        if final_outputs["pred_logits"].shape[:2] != raw.shape[:2]:
            raise ValueError("Final output queries must align with captured raw features")
        grouped = {}
        matched_rows = zip(final_indices, feature_indices, targets, strict=True)
        for b, ((src, dst), feature_src, target) in enumerate(matched_rows):
            src, dst = src.to(raw.device), dst.to(raw.device)
            feature_src = feature_src.to(raw.device)
            if feature_src.shape != src.shape or not torch.equal(feature_src, src):
                raise ValueError("Final matching query indices must index layer-0 raw features unchanged")
            if "valid" in target:
                keep = target["valid"][dst].bool()
                src, dst, feature_src = src[keep], dst[keep], feature_src[keep]
            if src.numel() == 0:
                continue
            if "marn_object_id" not in target:
                raise RuntimeError(
                    "COCO annotation IDs missing; use C2 checker dataset until formal transforms are wired"
                )
            if len(target["marn_object_id"]) != len(target["labels"]):
                raise RuntimeError("Annotation IDs no longer align after transforms")
            labels = target["labels"][dst]
            features = raw[b, feature_src].float()
            geometry = final_outputs["pred_boxes"][b, src].float()
            logits = final_outputs["pred_logits"][b, src].float()
            conf = logits[torch.arange(src.numel(), device=raw.device), labels].sigmoid()
            iou = paired_iou(geometry, target["boxes"][dst].float())
            valid = (torch.isfinite(features).all(-1) & torch.isfinite(geometry).all(-1)
                     & (geometry[:, 2:] > 0).all(-1) & torch.isfinite(conf) & torch.isfinite(iou))
            self.stats["matches"] += src.numel()
            self.stats["candidate_total"] += src.numel()
            self.stats["invalid"] += int((~valid).sum())
            self.stats["invalid_values"] += int((~valid).sum())
            self.stats["low_confidence"] += int((valid & (conf < self.config.confidence_min)).sum())
            self.stats["low_iou"] += int(
                (valid & (conf >= self.config.confidence_min) & (iou < self.config.iou_min)).sum()
            )
            self._increment_class_count(self.class_candidates, labels)
            finite_conf = conf[torch.isfinite(conf)].detach().float().cpu().tolist()
            finite_iou = iou[torch.isfinite(iou)].detach().float().cpu().tolist()
            self._conf_values.extend(finite_conf)
            self._iou_values.extend(finite_iou)
            class_ids = self.reader.class_ids.to(logits.device)
            predicted_class = class_ids[logits.index_select(-1, class_ids).argmax(-1)]
            class_consistent = predicted_class == labels
            self.stats["class_consistent"] += int(class_consistent.sum())
            self.stats["class_inconsistent"] += int((~class_consistent).sum())
            keep = valid & (conf >= self.config.confidence_min) & (iou >= self.config.iou_min)
            rows = keep.nonzero().flatten()
            self.stats["hard_gate_passed"] += int(rows.numel())
            self._increment_class_count(self.class_eligible, labels[rows])
            image_id = int(target["image_id"].reshape(-1)[0])
            for r in rows.tolist():
                object_id = int(target["marn_object_id"][dst[r]])
                label = int(labels[r])
                key = (image_id, object_id)
                grouped.setdefault(key, []).append({
                    "quality_diag": float((conf[r] * iou[r].clamp(0, 1)).sqrt()),
                    "label": label,
                    "feature": features[r].clone(),
                    "geometry": geometry[r].clone(),
                    "query_id": int(src[r]),
                    "conf": conf[r].detach(),
                    "iou": iou[r].detach(),
                    "class_consistent": bool(class_consistent[r]),
                })

        # Estimate feature stability only when repeated observations of the
        # same object are present in the current observed window.  A singleton
        # has no stability evidence and is explicitly marked unknown.
        for key, records in grouped.items():
            feature_stack = torch.stack([item["feature"] for item in records])
            if len(records) > 1:
                mean_feature = F.normalize(feature_stack.float().mean(0, keepdim=True), dim=-1, eps=1e-6)
                stability_values = F.cosine_similarity(feature_stack.float(), mean_feature, dim=-1).clamp(0, 1)
            else:
                stability_values = feature_stack.new_full((1,), float("nan"))
            for item, stability in zip(records, stability_values):
                novelty, coverage = self._novelty_and_coverage(item["label"], item["feature"])
                r_value, v_value, write_quality = compute_write_quality(
                    item["conf"].reshape(1), item["iou"].reshape(1),
                    stability.reshape(1), novelty.reshape(1), coverage.reshape(1),
                    self.config.quality_eta,
                )
                write_quality_value = float(write_quality[0])
                components = {
                    "R": float(r_value[0]),
                    "S": float(stability),
                    "N": float(novelty),
                    "D_c": float(coverage),
                    "V": float(v_value[0]),
                }
                for name, value in components.items():
                    _record_quality_stats(self.quality_stats, name, torch.as_tensor([value]))
                _record_quality_stats(self.quality_stats, "Q_write", write_quality)
                self.stats["eligible"] += 1
                if not math.isfinite(write_quality_value):
                    self.stats["quality_unknown"] += 1
                candidate = (
                    item["quality_diag"], write_quality_value, item["label"],
                    item["feature"], item["geometry"], item["query_id"],
                    float(item["conf"]), float(item["iou"]), components,
                    self.config.layer, MATCH_SOURCE_FINAL,
                )
                old = self.pending.get(key)
                if old is not None:
                    self.stats["duplicates"] += 1
                    if self._candidate_priority(old) >= self._candidate_priority(candidate):
                        continue
                # Clone individual candidates; do not retain a full activation graph/storage.
                self.pending[key] = candidate

    @torch.no_grad()
    def commit(self):
        if not self.enabled:
            return
        if not self.training or self.capture is not None or self.snapshot is None or not self.observed:
            raise RuntimeError("Commit requires a completed training loss/backward/optimizer window")
        bank = self.bank
        step = int(self.committed_step) + 1
        operations = {"fill": 0, "update": 0, "replace": 0}
        update_logs = []
        eviction_logs = []
        count = len(self.pending)
        # MARN_C3E_COMMIT_V1: discrete slot metadata crosses to the host once.
        # Keep every floating-point similarity/priority/update on its original
        # device with the original tensor shape and operation order. Candidates
        # remain sorted globally; only scalar decisions move to the host.
        classes = bank.classes.cpu().tolist()
        validity = bank.valid.cpu().tolist()
        slot_lists = {}
        for i, label in enumerate(classes):
            slot_lists.setdefault(label, []).append(i)
        slot_tensors = {}

        def argmax(values):
            # torch.argmax returns the first maximum, including the first NaN.
            for i, value in enumerate(values):
                if math.isnan(value):
                    return i
            return max(range(len(values)), key=values.__getitem__)

        for (image_id, object_id), (quality_diag, write_quality, label, feature,
                                    geometry, _query_id, conf, iou, _components,
                                    feature_layer, match_source) in sorted(self.pending.items()):
            indices = slot_lists.get(label, [])
            if not indices:
                raise ValueError(f"No bank slots for raw category {label}")
            if label not in slot_tensors:
                slot_tensors[label] = bank.classes.new_tensor(indices)
            slots = slot_tensors[label]
            valid_indices = [
                i for i in indices
                if validity[i]
                and bool(torch.isfinite(bank.features[i]).all())
                and bool(torch.isfinite(bank.geometry[i]).all())
            ]
            slot, action = None, None
            matched_similarity = None
            if valid_indices:
                valid_slots = bank.classes.new_tensor(valid_indices)
                sim = F.cosine_similarity(bank.features[valid_slots].float(), feature[None], dim=-1)
                values = sim.cpu().tolist()
                best = argmax(values)
                if values[best] >= self.config.similarity_update:
                    slot, action = valid_indices[best], "update"
                    matched_similarity = float(values[best])
            if slot is None:
                empty = [
                    i for i in indices
                    if not validity[i]
                    or not bool(torch.isfinite(bank.features[i]).all())
                    or not bool(torch.isfinite(bank.geometry[i]).all())
                ]
                if empty:
                    slot, action = empty[0], "fill"
                else:
                    normalized = F.normalize(bank.features[slots].float(), dim=-1)
                    sims = normalized @ normalized.T
                    sims.fill_diagonal_(-1)
                    redundancy = sims.max(-1).values.clamp_min(0)
                    age = (step - bank.updated_step[slots]).float() / self.reader.config.age_scale
                    # Maintenance is lexicographic: reliability components
                    # first, then redundancy and age.  Q_write is retained as
                    # metadata and is not a substitute for reliability.
                    conf_values = bank.conf[slots].float()
                    iou_values = bank.iou[slots].float()
                    reliability_values = bank.reliability_known[slots].cpu().tolist()
                    keys = []
                    for pos, slot_index in enumerate(indices):
                        known = int(reliability_values[pos]) and int(
                            math.isfinite(float(conf_values[pos])) and math.isfinite(float(iou_values[pos]))
                        )
                        # Reliability is deliberately component-wise.  Q_write
                        # remains an auxiliary maintenance field and cannot
                        # replace conf/IoU in this ordering.
                        conf_value = float(conf_values[pos]) if known else 0.0
                        iou_value = float(iou_values[pos]) if known else 0.0
                        keys.append((known, conf_value, iou_value,
                                     -float(redundancy[pos]), -float(age[pos]), -slot_index))
                    victim = min(range(len(indices)), key=keys.__getitem__)
                    slot, action = indices[victim], "replace"
                    eviction_logs.append({
                        "slot": int(slot),
                        "class": int(label),
                        "age": int(max(step - int(bank.updated_step[slot]), 0)),
                        "redundancy": float(redundancy[victim]),
                        "reason": self._eviction_reason(keys, victim),
                        "reliability_known": bool(keys[victim][0]),
                    })
            if action == "update":
                # EMA features/geometry and each metadata component. Unknown
                # checkpoint values are replaced only by a measured update.
                momentum = self.config.feature_ema
                before_age = max(step - int(bank.updated_step[slot]), 0)
                before = {
                    "conf": self._json_float(bank.conf[slot]),
                    "iou": self._json_float(bank.iou[slot]),
                    "Q_write": self._json_float(bank.write_quality[slot]),
                    "age": before_age,
                }
                bank.features[slot].lerp_(feature, 1 - momentum)
                bank.geometry[slot].lerp_(geometry, 1 - momentum)
                conf_tensor = torch.as_tensor(conf, device=feature.device)
                iou_tensor = torch.as_tensor(iou, device=feature.device)
                if bool(bank.reliability_known[slot]) and math.isfinite(float(bank.conf[slot])):
                    bank.conf[slot].lerp_(conf_tensor, 1 - momentum)
                else:
                    bank.conf[slot].copy_(conf_tensor)
                if bool(bank.reliability_known[slot]) and math.isfinite(float(bank.iou[slot])):
                    bank.iou[slot].lerp_(iou_tensor, 1 - momentum)
                else:
                    bank.iou[slot].copy_(iou_tensor)
                diag_tensor = torch.as_tensor(quality_diag, device=feature.device)
                bank.quality_diag[slot].lerp_(diag_tensor, 1 - momentum)
                bank.quality[slot].copy_(bank.quality_diag[slot])
                if math.isfinite(float(write_quality)):
                    write_tensor = torch.as_tensor(write_quality, device=feature.device)
                    if math.isfinite(float(bank.write_quality[slot])):
                        bank.write_quality[slot].lerp_(write_tensor, 1 - momentum)
                    else:
                        bank.write_quality[slot].copy_(write_tensor)
                bank.reliability_known[slot] = True
                bank.feature_layer[slot] = feature_layer
                bank.match_source[slot] = match_source
            else:
                bank.features[slot].copy_(feature)
                bank.geometry[slot].copy_(geometry)
                bank.conf[slot] = conf
                bank.iou[slot] = iou
                bank.quality_diag[slot] = quality_diag
                bank.quality[slot] = quality_diag
                bank.write_quality[slot] = write_quality
                bank.reliability_known[slot] = math.isfinite(conf) and math.isfinite(iou)
                bank.update_count[slot] = 1
                bank.feature_layer[slot] = feature_layer
                bank.match_source[slot] = match_source
            # Keep unknown provenance explicit.  quality_diag is diagnostic
            # only and is never used to clear or set a maintenance bit.
            unknown_mask = int(bank.unknown_mask[slot])
            unknown_mask &= ~(UNKNOWN_FEATURE_LAYER | UNKNOWN_MATCH_SOURCE)
            if math.isfinite(conf):
                unknown_mask &= ~UNKNOWN_CONF
            else:
                unknown_mask |= UNKNOWN_CONF
            if math.isfinite(iou):
                unknown_mask &= ~UNKNOWN_IOU
            else:
                unknown_mask |= UNKNOWN_IOU
            if math.isfinite(write_quality):
                unknown_mask &= ~UNKNOWN_WRITE_QUALITY
            elif not math.isfinite(float(bank.write_quality[slot])):
                unknown_mask |= UNKNOWN_WRITE_QUALITY
            bank.unknown_mask[slot] = unknown_mask
            if action == "update":
                bank.update_count[slot] += 1
            bank.updated_step[slot] = step
            bank.valid[slot] = True
            validity[slot] = True
            bank.source_image[slot], bank.source_object[slot] = image_id, object_id
            operations[action] += 1
            if action == "update":
                update_logs.append({
                    "slot": int(slot),
                    "class": int(label),
                    "similarity": matched_similarity,
                    "update_count": int(bank.update_count[slot]),
                    "before": before,
                    "after": {
                        "conf": self._json_float(bank.conf[slot]),
                        "iou": self._json_float(bank.iou[slot]),
                        "Q_write": self._json_float(bank.write_quality[slot]),
                        "age": 0,
                    },
                })
        self.committed_step.fill_(step)
        consistency_total = self.stats["class_consistent"] + self.stats["class_inconsistent"]
        self.last_commit = {"step": step, "microbatches": self.window_reads,
                            "unique_objects": count, **self.stats, **operations,
                            "valid_slots": sum(validity),
                            "covered_classes": len({c for c, v in zip(classes, validity) if v}),
                            "class_candidates": dict(self.class_candidates),
                            "class_eligible": dict(self.class_eligible),
                            "category_consistency_rate": (
                                self.stats["class_consistent"] / consistency_total
                                if consistency_total else None
                            ),
                            "conf_quantiles": self._quantiles(self._conf_values),
                            "iou_quantiles": self._quantiles(self._iou_values),
                            "updates": update_logs,
                            "evictions": eviction_logs,
                            "quality_eta": self.config.quality_eta,
                            "quality_components": _quality_stats_snapshot(self.quality_stats)}
        self.pending.clear()
        self.snapshot = None
        self.observed = False
        self._reset_stats()


def attach(module, class_ids, config=None):
    config = config or MARNConfig()
    model = module.model
    decoder = model.transformer.decoder
    if config.layer != 0 or decoder.num_layers < 2:
        raise ValueError("C2 supports insertion after layer 0 with a decoder tail")
    if module.model_config.compile or module.model_config.cuda_graphs or module.model_config.gradient_checkpointing:
        raise ValueError("C2 requires eager execution without activation checkpointing")
    # MARN_C3B_EMA_GATE_V1: C2 stays guarded; C3B verifies its callback at fit start.
    if module.train_config.use_ema and getattr(module, "_marn_ema_policy", None) != "online_bank_exact_copy_v1":
        raise ValueError("MARN EMA requires the C3B MARNTrainingModule and MARNEMACallback")
    if model.segmentation_head is not None or getattr(model, "use_grouppose_keypoints", False):
        raise ValueError("C2 supports ordinary detection only")
    adapter = MARNAdapter(decoder.d_model, class_ids, model.num_queries, config)
    decoder.marn = adapter.to(next(model.parameters()).device)
    # Plain bound method; criterion neither owns nor serializes a second copy.
    module.criterion.marn_observer = adapter.observe
    return adapter


from rfdetr.training.module_model import RFDETRModelModule


class MARNModelModule(RFDETRModelModule):
    def __init__(self, model_config, train_config, *, class_ids, marn_config=None):
        super().__init__(model_config, train_config)
        attach(self, class_ids, marn_config)

    @property
    def adapter(self):
        return self.model.transformer.decoder.marn

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # Lightning executes training_step/backward in the closure. Commit only
        # after the actual optimizer call returns; accumulation-only batches do not call here.
        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.adapter.commit()
        return result

    def on_fit_start(self):
        if self.trainer.world_size != 1 or self.trainer.precision not in ("bf16-mixed", "32-true"):
            raise ValueError("C2 supports single-GPU BF16 or FP32 only (no skipped FP16 optimizer steps)")
        super().on_fit_start()

    def on_save_checkpoint(self, checkpoint):
        adapter = self.adapter
        if adapter.pending or adapter.snapshot is not None or adapter.capture is not None:
            raise RuntimeError("Save only after the optimizer and bank commit")
        super().on_save_checkpoint(checkpoint)
        checkpoint["marn_config"] = asdict(adapter.config)
        checkpoint["marn_class_ids"] = adapter.reader.class_ids.cpu().tolist()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("marn_config") != asdict(self.adapter.config):
            raise ValueError("MARN configuration differs from checkpoint")
        if checkpoint.get("marn_class_ids") != self.adapter.reader.class_ids.cpu().tolist():
            raise ValueError("MARN class IDs differ from checkpoint")
        super().on_load_checkpoint(checkpoint)
