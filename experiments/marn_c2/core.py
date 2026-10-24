"""Stage C1: prototype read path and T=1 bounded query refinement.

This module does not yet implement training-bank writes or RF-DETR wiring.
All tensors use batch-first [B, Q, D]. No validation targets enter retrieval.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BankSnapshot:
    features: Tensor
    geometry: Tensor
    classes: Tensor
    quality: Tensor
    updated_step: Tensor
    valid: Tensor
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
        self.register_buffer("quality", torch.zeros(n))
        self.register_buffer("updated_step", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(n, dtype=torch.bool))
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
                "features", "geometry", "classes", "quality", "updated_step", "valid"
            )), step=int(step)
        )


@dataclass(frozen=True)
class RetrievalConfig:
    key_dim: int = 64
    top_k: int = 4
    temperature: float = 0.2
    class_weight: float = 0.2
    geometry_weight: float = 0.1
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

    def forward(self, queries, logits, boxes, selected, snapshot):
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
        finite &= torch.isfinite(snapshot.quality)
        valid = snapshot.valid & finite & (snapshot.quality > 0)
        # Prevent invalid NaN slots from contaminating matmul, including its backward.
        safe_raw = torch.where(valid[:, None], raw, torch.zeros_like(raw))
        safe_geometry = torch.where(valid[:, None], snapshot.geometry, torch.zeros_like(snapshot.geometry))
        safe_quality = torch.where(valid, snapshot.quality, torch.zeros_like(snapshot.quality))
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
        scores = (cosine / c.temperature + c.class_weight * class_score
                  + c.geometry_weight * geometry_score
                  + c.quality_weight * safe_quality.clamp_min(1e-6).log()[None]
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
            (weights * safe_quality[indices]).sum(-1),
            has_memory.float(),
        ), dim=-1)
        memory[bi, qi] = mixed.to(memory.dtype)
        summary[bi, qi] = stats
        available[bi, qi] = has_memory
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
