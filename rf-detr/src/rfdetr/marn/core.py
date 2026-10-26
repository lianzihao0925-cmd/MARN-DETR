"""Stage C1: prototype read path and T=1 bounded query refinement.

This module does not yet implement training-bank writes or RF-DETR wiring.
All tensors use batch-first [B, Q, D]. No validation targets enter retrieval.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Bit flags persisted with every slot so missing provenance/reliability is
# explicit after checkpoint migration.  A set bit means the corresponding
# field is unknown and must not be used as a measured value.
UNKNOWN_CONF = 1 << 0
UNKNOWN_IOU = 1 << 1
UNKNOWN_WRITE_QUALITY = 1 << 2
UNKNOWN_FEATURE_LAYER = 1 << 3
UNKNOWN_MATCH_SOURCE = 1 << 4

FEATURE_LAYER_UNKNOWN = -1
MATCH_SOURCE_UNKNOWN = 0
MATCH_SOURCE_FINAL = 1


@dataclass(frozen=True)
class BankSnapshot:
    features: Tensor
    geometry: Tensor
    classes: Tensor
    conf: Tensor
    iou: Tensor
    quality_diag: Tensor
    write_quality: Tensor
    updated_step: Tensor
    update_count: Tensor
    reliability_known: Tensor
    valid: Tensor
    feature_layer: Tensor
    match_source: Tensor
    unknown_mask: Tensor
    step: int


class PrototypeBank(nn.Module):
    """Class-balanced raw-feature storage; caller owns snapshot lifetime.

    Take one snapshot at optimizer-step start, reuse it across microbatches,
    and commit pending writes only after backward and the optimizer update.
    C2 will implement the writer and its data provenance checks.
    """

    def __init__(self, class_ids, dim=256, slots_per_class=4):
        super().__init__()
        ids = torch.as_tensor(class_ids, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0 or len(set(ids.tolist())) != ids.numel():
            raise ValueError("class_ids must be a nonempty sequence of unique raw category IDs")
        if dim <= 0 or slots_per_class <= 0 or (ids < 0).any():
            raise ValueError("Invalid bank dimensions or class IDs")
        classes = ids.repeat_interleave(slots_per_class)
        n = classes.numel()
        self.register_buffer("features", torch.zeros(n, dim))
        self.register_buffer("geometry", torch.zeros(n, 4))
        self.register_buffer("classes", classes)
        # Candidate reliability is stored component-wise.  NaN plus the
        # explicit mask means old checkpoints cannot be mistaken for measured
        # confidence or localization quality.
        self.register_buffer("conf", torch.full((n,), float("nan")))
        self.register_buffer("iou", torch.full((n,), float("nan")))
        # ``quality_diag`` preserves the old sqrt(conf * IoU) diagnostic.  It
        # is intentionally separate from ``write_quality`` and is not used by
        # retrieval.  ``write_quality`` is NaN until every required component
        # of Q_write is available and eta is explicitly configured.
        self.register_buffer("quality_diag", torch.zeros(n))
        self.register_buffer("write_quality", torch.full((n,), float("nan")))
        # Kept as a compatibility buffer for checkpoints written before the
        # quality split.  New code mirrors quality_diag here but never reads
        # it for retrieval or maintenance decisions.
        self.register_buffer("quality", torch.zeros(n))
        self.register_buffer("updated_step", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("update_count", torch.zeros(n, dtype=torch.long))
        self.register_buffer("reliability_known", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("valid", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("feature_layer", torch.full((n,), FEATURE_LAYER_UNKNOWN, dtype=torch.int16))
        self.register_buffer("match_source", torch.full((n,), MATCH_SOURCE_UNKNOWN, dtype=torch.int8))
        self.register_buffer(
            "unknown_mask",
            torch.full(
                (n,),
                UNKNOWN_CONF | UNKNOWN_IOU | UNKNOWN_WRITE_QUALITY
                | UNKNOWN_FEATURE_LAYER | UNKNOWN_MATCH_SOURCE,
                dtype=torch.uint8,
            ),
        )
        # Training provenance. These will be removable from deployment artifacts.
        self.register_buffer("source_image", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("source_object", torch.full((n,), -1, dtype=torch.long))

    @torch.no_grad()
    def snapshot(self, step: int) -> BankSnapshot:
        if step < 0:
            raise ValueError("step must be nonnegative")
        # detach() alone aliases storage and is unsafe if a bank is later modified.
        return BankSnapshot(
            *(getattr(self, name).detach().clone() for name in (
                "features", "geometry", "classes", "conf", "iou", "quality_diag",
                "write_quality", "updated_step", "update_count", "reliability_known", "valid"
                , "feature_layer", "match_source", "unknown_mask"
            )), step=int(step)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Migrate pre-split checkpoints without inventing Q_write values."""
        diag_key = prefix + "quality_diag"
        write_key = prefix + "write_quality"
        legacy_key = prefix + "quality"
        conf_key = prefix + "conf"
        iou_key = prefix + "iou"
        count_key = prefix + "update_count"
        reliability_key = prefix + "reliability_known"
        feature_layer_key = prefix + "feature_layer"
        match_source_key = prefix + "match_source"
        unknown_mask_key = prefix + "unknown_mask"
        conf_missing = conf_key not in state_dict
        iou_missing = iou_key not in state_dict
        write_missing = write_key not in state_dict
        feature_layer_missing = feature_layer_key not in state_dict
        match_source_missing = match_source_key not in state_dict
        if diag_key not in state_dict:
            if legacy_key in state_dict:
                state_dict[diag_key] = state_dict[legacy_key].detach().clone()
            else:
                state_dict[diag_key] = self.quality_diag.detach().clone()
        if write_key not in state_dict:
            # Missing write quality is explicitly unknown, rather than a
            # fabricated zero or one that could affect future maintenance.
            state_dict[write_key] = torch.full_like(self.write_quality, float("nan"))
        if legacy_key not in state_dict:
            state_dict[legacy_key] = state_dict[diag_key].detach().clone()
        if conf_key not in state_dict:
            state_dict[conf_key] = torch.full_like(self.conf, float("nan"))
        if iou_key not in state_dict:
            state_dict[iou_key] = torch.full_like(self.iou, float("nan"))
        if count_key not in state_dict:
            state_dict[count_key] = torch.zeros_like(self.update_count)
        if reliability_key not in state_dict:
            state_dict[reliability_key] = torch.zeros_like(self.reliability_known)
        if feature_layer_key not in state_dict:
            state_dict[feature_layer_key] = torch.full_like(
                self.feature_layer, FEATURE_LAYER_UNKNOWN
            )
        if match_source_key not in state_dict:
            state_dict[match_source_key] = torch.full_like(
                self.match_source, MATCH_SOURCE_UNKNOWN
            )
        if unknown_mask_key not in state_dict:
            unknown = torch.zeros_like(self.unknown_mask)
            if conf_missing:
                unknown |= UNKNOWN_CONF
            else:
                unknown |= (~torch.isfinite(state_dict[conf_key])).to(unknown.dtype) * UNKNOWN_CONF
            if iou_missing:
                unknown |= UNKNOWN_IOU
            else:
                unknown |= (~torch.isfinite(state_dict[iou_key])).to(unknown.dtype) * UNKNOWN_IOU
            if write_missing:
                unknown |= UNKNOWN_WRITE_QUALITY
            else:
                unknown |= (~torch.isfinite(state_dict[write_key])).to(unknown.dtype) * UNKNOWN_WRITE_QUALITY
            if feature_layer_missing:
                unknown |= UNKNOWN_FEATURE_LAYER
            if match_source_missing:
                unknown |= UNKNOWN_MATCH_SOURCE
            state_dict[unknown_mask_key] = unknown
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )


@dataclass(frozen=True)
class RetrievalConfig:
    key_dim: int = 64
    top_k: int = 4
    temperature: float = 0.2
    class_weight: float = 0.2
    geometry_weight: float = 0.1
    # Deprecated checkpoint field; retrieval no longer consumes Q_write.
    quality_weight: float = 0.2
    age_weight: float = 0.1
    age_scale: float = 1000.0
    min_cosine: float = 0.0


class PrototypeReader(nn.Module):
    """Exact full-bank scoring for selected queries, then top-k value mixing.

    Class compatibility normalizes sigmoid scores across actual COCO IDs.
    Geometry uses log width/height only, never cross-image absolute position.
    The first version uses identity-initialized shared learned key projection
    so the cosine rejection threshold is meaningful at initialization.
    """

    def __init__(self, dim, class_ids, config=None):
        super().__init__()
        self.config = config or RetrievalConfig()
        c = self.config
        if not (1 <= c.key_dim <= dim and c.top_k > 0 and c.temperature > 0 and c.age_scale > 0):
            raise ValueError("Invalid retrieval configuration")
        ids = torch.as_tensor(class_ids, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0 or len(set(ids.tolist())) != ids.numel() or (ids < 0).any():
            raise ValueError("Invalid class IDs")
        if not -1 <= c.min_cosine <= 1:
            raise ValueError("min_cosine must lie in [-1, 1]")
        self.register_buffer("class_ids", ids)
        self.max_class_id = int(ids.max())
        # Map original category IDs to compact compatibility-vector positions.
        lut = torch.full((int(ids.max()) + 1,), -1, dtype=torch.long)
        lut[ids] = torch.arange(ids.numel())
        self.register_buffer("class_index", lut)
        self.key = nn.Linear(dim, c.key_dim, bias=False)
        with torch.no_grad():
            self.key.weight.zero_()
            self.key.weight[:, :c.key_dim].copy_(torch.eye(c.key_dim))
        self.last_read_records = []

    def forward(self, queries, logits, boxes, selected, snapshot):
        self.last_read_records = []
        if queries.ndim != 3 or selected.shape != queries.shape[:2] or selected.dtype != torch.bool:
            raise ValueError("Expected queries [B,Q,D] and bool selection [B,Q]")
        if logits.shape[:2] != queries.shape[:2] or boxes.shape != (*queries.shape[:2], 4):
            raise ValueError("Logit/box query dimensions do not match")
        if logits.shape[-1] <= self.max_class_id:
            raise ValueError("Logits do not contain all original COCO category slots")
        # Bank tensors are contracts owned by PrototypeBank and the training-only writer.
        memory = torch.zeros_like(queries)
        summary = queries.new_zeros((*queries.shape[:2], 3), dtype=torch.float32)
        available = torch.zeros_like(selected)
        bi, qi = selected.nonzero(as_tuple=True)
        if bi.numel() == 0:
            return memory, summary, available

        q = queries[bi, qi]
        raw = snapshot.features
        finite = torch.isfinite(raw).all(-1) & torch.isfinite(snapshot.geometry).all(-1)
        valid = snapshot.valid & finite
        # Prevent invalid NaN slots from contaminating matmul, including its backward.
        safe_raw = torch.where(valid[:, None], raw, torch.zeros_like(raw))
        safe_geometry = torch.where(valid[:, None], snapshot.geometry, torch.zeros_like(snapshot.geometry))
        qkeys = F.normalize(self.key(q).float(), dim=-1, eps=1e-6)
        mkeys = F.normalize(self.key(safe_raw).float(), dim=-1, eps=1e-6)
        cosine = qkeys @ mkeys.T
        c = self.config
        category_prob = logits[bi, qi].float().sigmoid().index_select(-1, self.class_ids)
        category_prob = category_prob / category_prob.sum(-1, keepdim=True).clamp_min(1e-6)
        compact_ids = self.class_index[snapshot.classes]
        if bool((compact_ids < 0).any()):
            raise ValueError("Bank contains a class not present in reader class_ids")
        class_score = category_prob[:, compact_ids].clamp_min(1e-6).log()
        qwh = boxes[bi, qi, 2:].float().clamp_min(1e-6).log()
        mwh = safe_geometry[:, 2:].float().clamp_min(1e-6).log()
        geometry_score = -(qwh[:, None] - mwh[None]).abs().mean(-1)
        age = (snapshot.step - snapshot.updated_step).clamp_min(0).float() / c.age_scale
        # Retrieval relevance is deliberately independent of Q_write.  The
        # deprecated quality_weight field remains in the config only so old
        # metadata can be loaded, but it is ignored here.
        scores = (cosine / c.temperature + c.class_weight * class_score
                  + c.geometry_weight * geometry_score
                  - c.age_weight * age[None])
        eligible = valid[None] & (cosine >= c.min_cosine) & torch.isfinite(scores)
        scores = scores.masked_fill(~eligible, -torch.inf)
        k = min(c.top_k, raw.shape[0])
        values, indices = scores.topk(k, dim=-1)
        good = eligible.gather(1, indices)
        has_memory = good.any(-1)
        # Avoid softmax(-inf,...,-inf) for empty/rejected rows.
        safe_scores = torch.where(has_memory[:, None], values, torch.zeros_like(values))
        weights = safe_scores.softmax(-1) * good
        mixed = (weights[..., None] * safe_raw[indices].float()).sum(1)
        stats = torch.stack((
            (weights * cosine.gather(1, indices)).sum(-1),
            (weights * geometry_score.gather(1, indices)).sum(-1),
            has_memory.float(),
        ), dim=-1)
        memory[bi, qi] = mixed.to(memory.dtype)
        summary[bi, qi] = stats
        available[bi, qi] = has_memory
        # Keep the fixed random query-selection protocol unchanged while
        # exposing a per-read audit trail.  These records are derived only
        # from the immutable snapshot passed to this call; Q_write is never a
        # retrieval condition.
        for row, (batch_index, query_index) in enumerate(zip(bi.tolist(), qi.tolist(), strict=True)):
            good_positions = good[row].nonzero(as_tuple=False).flatten().tolist()
            if not good_positions:
                self.last_read_records.append({
                    "batch_index": int(batch_index),
                    "query_index": int(query_index),
                    "prototype_index": None,
                    "class_compatibility": None,
                    "geometry_compatibility": None,
                    "similarity": None,
                    "age": None,
                    "prototype_version": None,
                    "updated_step": None,
                    "feature_layer": None,
                    "match_source": None,
                    "unknown_mask": None,
                    "accepted": False,
                    "actual_correction": False,
                })
                continue
            for position in good_positions:
                slot = int(indices[row, position])
                self.last_read_records.append({
                    "batch_index": int(batch_index),
                    "query_index": int(query_index),
                    "prototype_index": slot,
                    "class_compatibility": float(category_prob[row, compact_ids[slot]]),
                    "geometry_compatibility": float(geometry_score[row, slot]),
                    "similarity": float(cosine[row, slot]),
                    "age": int(max(snapshot.step - int(snapshot.updated_step[slot]), 0)),
                    "prototype_version": int(snapshot.update_count[slot]),
                    "updated_step": int(snapshot.updated_step[slot]),
                    "feature_layer": int(snapshot.feature_layer[slot]),
                    "match_source": int(snapshot.match_source[slot]),
                    "unknown_mask": int(snapshot.unknown_mask[slot]),
                    "accepted": True,
                    "actual_correction": False,
                })
        return memory, summary, available


class BoundedMGQR(nn.Module):
    """Outline section 4.5: zero-output, separately bounded residual and gate."""

    def __init__(self, dim=256, bottleneck=64, rho=0.1, beta_max=0.5):
        super().__init__()
        if dim <= 0 or bottleneck <= 0 or rho <= 0 or not 0 < beta_max <= 1:
            raise ValueError("Invalid MGQR configuration")
        self.rho = float(rho)
        self.beta_max = float(beta_max)
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.fuse = nn.Linear(3 * dim + 3, bottleneck)
        self.out = nn.Linear(bottleneck, dim, bias=False)
        self.gate = nn.Linear(dim + 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, queries, memory, summary, accepted):
        if queries.shape != memory.shape or summary.shape != (*queries.shape[:2], 3):
            raise ValueError("Invalid MGQR tensor shapes")
        if accepted.shape != queries.shape[:2] or accepted.dtype != torch.bool:
            raise ValueError("accepted must be a bool [B,Q] mask")
        bi, qi = accepted.nonzero(as_tuple=True)
        if bi.numel() == 0:
            return queries
        q, m, e = queries[bi, qi], memory[bi, qi], summary[bi, qi]
        qn, mn = self.q_norm(q), self.m_norm(m)
        e = e.to(qn.dtype)
        fused = torch.cat((qn, mn, qn * mn, e), dim=-1)
        direction = self.out(F.gelu(self.fuse(fused))).float()
        limit = self.rho * q.float().norm(dim=-1, keepdim=True)
        factor = (limit / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        bounded = direction * factor
        beta = self.beta_max * self.gate(torch.cat((qn, e), dim=-1)).float().sigmoid()
        result = queries.clone()
        result[bi, qi] = (q.float() + beta * bounded).to(q.dtype)
        return result
