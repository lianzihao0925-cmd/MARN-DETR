"""Paired, train-only intervention labels for the future UAMR router.

This module deliberately lives outside :mod:`rfdetr.marn.integration`.  It
does not change OCPM retrieval and it does not make a routing decision.  The
only target it constructs is the measured change in the official detection
loss when the same query is evaluated with memory disabled and enabled.

The two branches must use the same input, model state, bank snapshot and
Hungarian assignment.  The assignment is supplied by the caller rather than
recomputed for the intervention branch; otherwise the label would mix memory
benefit with a matching change.  Labels require ground truth and therefore are
guarded to the training/offline stage.  Validation and test code must not call
this module to produce an online decision.

Confidence, classification loss, localization loss, single-object IoU,
query/prototype similarity and ``Q_write`` remain separate signals.  None of
them is used as a proxy for ``benefit`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairedInterventionContext:
    """Provenance contract for one paired label collection.

    The booleans are intentionally explicit.  A caller that cannot establish
    one of these invariants must not create a benefit label.  ``stage`` is
    restricted to training because all losses below consume ground-truth
    targets.
    """

    stage: str = "train"
    same_input: bool = True
    same_model_state: bool = True
    same_bank_snapshot: bool = True
    same_matching: bool = True
    same_rng_state: bool = True

    def validate(self) -> None:
        if self.stage.lower() not in {"train", "training", "offline_train"}:
            raise ValueError("UAMR paired labels are training/offline only; validation and test are read-only")
        checks = {
            "same_input": self.same_input,
            "same_model_state": self.same_model_state,
            "same_bank_snapshot": self.same_bank_snapshot,
            "same_matching": self.same_matching,
            "same_rng_state": self.same_rng_state,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("Cannot create a paired intervention label; invariants failed: " + ", ".join(failed))


@dataclass(frozen=True)
class PairedInterventionLabels:
    """Per-image losses and benefits for a fixed matching pair.

    All fields are one-dimensional tensors in input batch order.  The
    ``full_image_*`` values include all matched objects and therefore expose
    query interactions.  ``local_*`` values cover the selected query when one
    is supplied; otherwise they are the mean over matched object pairs.  A
    local value can be unavailable when the selected query is unmatched, which
    is represented by ``local_valid=False`` and a zero placeholder.

    ``benefit`` is an alias for ``full_image_benefit`` for callers that want a
    single conservative task label.  The local label remains available for a
    later object-level router study and must not be silently substituted.
    """

    classification_loss_without: Tensor
    classification_loss_with: Tensor
    localization_loss_without: Tensor
    localization_loss_with: Tensor
    full_image_loss_without: Tensor
    full_image_loss_with: Tensor
    full_image_benefit: Tensor
    local_classification_loss_without: Tensor
    local_classification_loss_with: Tensor
    local_localization_loss_without: Tensor
    local_localization_loss_with: Tensor
    local_loss_without: Tensor
    local_loss_with: Tensor
    local_benefit: Tensor
    local_valid: Tensor
    valid: Tensor

    @property
    def benefit(self) -> Tensor:
        """Primary conservative label: lower full-image loss is positive."""

        return self.full_image_benefit

    def detached(self) -> "PairedInterventionLabels":
        """Return labels detached from model graphs for storage or router fitting."""

        values = {
            field: getattr(self, field).detach()
            for field in self.__dataclass_fields__
        }
        return PairedInterventionLabels(**values)


def _as_device_scalar(value: Tensor | float, device: torch.device) -> Tensor:
    result = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.numel() != 1:
        raise ValueError("num_boxes must be a scalar")
    return result.to(device=device, dtype=torch.float32)


def _weighted_detection_components(criterion: Any, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]],
                                  indices: list[tuple[Tensor, Tensor]], num_boxes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate official classification/localization components with fixed indices."""

    label_losses = criterion.loss_labels(outputs, targets, indices, num_boxes, log=False)
    box_losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)
    weights = getattr(criterion, "weight_dict", None)
    if not isinstance(weights, dict):
        raise ValueError("Criterion must expose weight_dict to define the official detection loss")
    if "loss_ce" not in weights:
        raise ValueError("Criterion weight_dict is missing loss_ce")
    classification = label_losses["loss_ce"] * float(weights["loss_ce"])
    localization_terms = [
        box_losses[key] * float(weights[key])
        for key in ("loss_bbox", "loss_giou")
        if key in box_losses and key in weights
    ]
    if not localization_terms:
        raise ValueError("Criterion weight_dict has no weighted localization loss")
    localization = torch.stack(localization_terms).sum()
    return classification, localization, classification + localization


def _single_pair_target(target: dict[str, Tensor], target_index: int) -> dict[str, Tensor]:
    """Keep only the fields consumed by the detection criterion for one pair."""

    result = {
        "labels": target["labels"][target_index : target_index + 1],
        "boxes": target["boxes"][target_index : target_index + 1],
    }
    if "valid" in target:
        result["valid"] = target["valid"][target_index : target_index + 1]
    return result


def _local_pair_components(
    criterion: Any,
    outputs_without: dict[str, Tensor],
    outputs_with: dict[str, Tensor],
    target: dict[str, Tensor],
    source_index: int,
    target_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluate both branches on one matched query-target pair.

    This is intentionally an offline diagnostic path.  It uses the criterion's
    own classification and box losses, but isolates one query and one target so
    a local label cannot be confused with confidence, IoU, or feature
    similarity.
    """

    device = outputs_without["pred_logits"].device
    # The branch is sliced down to one query below, so its local source index
    # is zero even when the original query had a larger batch index.
    source = torch.zeros(1, dtype=torch.long, device=device)
    target_row = torch.as_tensor([0], dtype=torch.long, device=device)
    pair_indices = [(source, target_row)]
    pair_target = [_single_pair_target(target, target_index)]
    one = torch.ones((), dtype=torch.float32, device=device)
    without = {key: value[:, source_index : source_index + 1] for key, value in outputs_without.items()
               if isinstance(value, Tensor) and value.ndim >= 2}
    with_memory = {key: value[:, source_index : source_index + 1] for key, value in outputs_with.items()
                   if isinstance(value, Tensor) and value.ndim >= 2}
    c0, l0, t0 = _weighted_detection_components(criterion, without, pair_target, pair_indices, one)
    c1, l1, t1 = _weighted_detection_components(criterion, with_memory, pair_target, pair_indices, one)
    return c0, c1, l0, l1, t0, t1


def _normalize_selected_queries(selected: Sequence[Any] | Tensor | None, batch_size: int) -> list[list[int] | None]:
    if selected is None:
        return [None] * batch_size
    if isinstance(selected, Tensor):
        if selected.ndim == 1 and selected.numel() == batch_size:
            return [[int(v)] for v in selected.detach().cpu().tolist()]
        if selected.ndim == 2 and selected.shape[0] == batch_size:
            return [[int(v) for v in row.detach().cpu().tolist()] for row in selected]
        raise ValueError("selected_query_indices tensor must have shape [B] or [B,K]")
    if len(selected) != batch_size:
        raise ValueError("selected_query_indices must have one entry per image")
    result: list[list[int] | None] = []
    for row in selected:
        if row is None:
            result.append(None)
        elif isinstance(row, Tensor):
            result.append([int(v) for v in row.detach().cpu().reshape(-1).tolist()])
        elif isinstance(row, (list, tuple)):
            result.append([int(v) for v in row])
        else:
            result.append([int(row)])
    return result


@torch.no_grad()
def compute_paired_intervention_labels(
    criterion: Any,
    outputs_without_memory: dict[str, Tensor],
    outputs_with_memory: dict[str, Tensor],
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],
    *,
    context: PairedInterventionContext,
    num_boxes: Tensor | float | None = None,
    selected_query_indices: Sequence[Any] | Tensor | None = None,
) -> PairedInterventionLabels:
    """Create train-only paired intervention labels.

    Args:
        criterion: The official ``SetCriterion`` instance.
        outputs_without_memory: Detector outputs with memory disabled.
        outputs_with_memory: Detector outputs with memory enabled.
        targets: Ground-truth targets for the training batch.
        indices: One fixed final-layer Hungarian assignment per image.  It is
            reused for both branches and must not be recomputed after memory
            intervention.
        context: Explicit provenance contract.  All invariants must be true.
        num_boxes: Optional shared official normalizer.  If omitted, it is
            computed once from the no-memory branch and reused for both.
        selected_query_indices: Optional query index per image (or multiple
            indices per image) for the local/object-level label.  Without it,
            local losses are averaged over all matched pairs.

    Returns:
        Per-image classification, localization, full-image and local benefits,
        where benefit is ``loss_without_memory - loss_with_memory``.  Positive
        values therefore indicate measured improvement.
    """

    context.validate()
    required = ("pred_logits", "pred_boxes")
    if any(key not in outputs_without_memory or key not in outputs_with_memory for key in required):
        raise ValueError("Both intervention branches must contain pred_logits and pred_boxes")
    batch_size = outputs_without_memory["pred_logits"].shape[0]
    if outputs_with_memory["pred_logits"].shape[:2] != outputs_without_memory["pred_logits"].shape[:2]:
        raise ValueError("Intervention branches must have identical query dimensions")
    if len(targets) != batch_size or len(indices) != batch_size:
        raise ValueError("targets and indices must match the output batch size")
    if outputs_with_memory["pred_logits"].shape != outputs_without_memory["pred_logits"].shape:
        raise ValueError("Intervention branches must have identical pred_logits shape")
    if outputs_with_memory["pred_boxes"].shape != outputs_without_memory["pred_boxes"].shape:
        raise ValueError("Intervention branches must have identical pred_boxes shape")

    device = outputs_without_memory["pred_logits"].device
    if num_boxes is None:
        num_boxes_tensor = criterion.num_boxes_for_targets(outputs_without_memory, targets)
    else:
        num_boxes_tensor = _as_device_scalar(num_boxes, device)
    selected = _normalize_selected_queries(selected_query_indices, batch_size)

    cls0: list[Tensor] = []
    cls1: list[Tensor] = []
    loc0: list[Tensor] = []
    loc1: list[Tensor] = []
    full0: list[Tensor] = []
    full1: list[Tensor] = []
    local_cls0: list[Tensor] = []
    local_cls1: list[Tensor] = []
    local_loc0: list[Tensor] = []
    local_loc1: list[Tensor] = []
    local_total0: list[Tensor] = []
    local_total1: list[Tensor] = []
    local_valid: list[bool] = []

    for image_index, target in enumerate(targets):
        out0 = {key: value[image_index : image_index + 1] for key, value in outputs_without_memory.items()
                if isinstance(value, Tensor) and value.ndim >= 2}
        out1 = {key: value[image_index : image_index + 1] for key, value in outputs_with_memory.items()
                if isinstance(value, Tensor) and value.ndim >= 2}
        image_indices = []
        for source, target_index in [indices[image_index]]:
            image_indices.append((source.to(device=device), target_index.to(device=device)))
        c0, l0, t0 = _weighted_detection_components(criterion, out0, [target], image_indices, num_boxes_tensor)
        c1, l1, t1 = _weighted_detection_components(criterion, out1, [target], image_indices, num_boxes_tensor)
        cls0.append(c0); cls1.append(c1); loc0.append(l0); loc1.append(l1); full0.append(t0); full1.append(t1)

        pairs = list(zip(indices[image_index][0].detach().cpu().tolist(), indices[image_index][1].detach().cpu().tolist()))
        wanted = selected[image_index]
        if wanted is None:
            chosen = pairs
        else:
            wanted_set = set(wanted)
            chosen = [(src, tgt) for src, tgt in pairs if src in wanted_set]
        if not chosen:
            zero = t0.detach() * 0.0
            local_cls0.append(zero); local_cls1.append(zero); local_loc0.append(zero); local_loc1.append(zero)
            local_total0.append(zero); local_total1.append(zero); local_valid.append(False)
            continue
        pair_values = []
        for src, tgt in chosen:
            pair_values.append(_local_pair_components(
                criterion,
                out0,
                out1,
                target,
                int(src),
                int(tgt),
            ))
        stacked = [torch.stack([row[column] for row in pair_values]).mean() for column in range(6)]
        local_cls0.append(stacked[0]); local_cls1.append(stacked[1]); local_loc0.append(stacked[2]); local_loc1.append(stacked[3])
        local_total0.append(stacked[4]); local_total1.append(stacked[5]); local_valid.append(True)

    full_without = torch.stack(full0)
    full_with = torch.stack(full1)
    local_without = torch.stack(local_total0)
    local_with = torch.stack(local_total1)
    return PairedInterventionLabels(
        classification_loss_without=torch.stack(cls0),
        classification_loss_with=torch.stack(cls1),
        localization_loss_without=torch.stack(loc0),
        localization_loss_with=torch.stack(loc1),
        full_image_loss_without=full_without,
        full_image_loss_with=full_with,
        full_image_benefit=full_without - full_with,
        local_classification_loss_without=torch.stack(local_cls0),
        local_classification_loss_with=torch.stack(local_cls1),
        local_localization_loss_without=torch.stack(local_loc0),
        local_localization_loss_with=torch.stack(local_loc1),
        local_loss_without=local_without,
        local_loss_with=local_with,
        local_benefit=local_without - local_with,
        local_valid=torch.as_tensor(local_valid, dtype=torch.bool, device=device),
        valid=torch.isfinite(full_without) & torch.isfinite(full_with),
    )


def summarize_benefits(benefits: Tensor, *, valid: Tensor | None = None, zero_tolerance: float = 0.0) -> dict[str, float | int]:
    """Summarize finite positive, zero and negative paired benefits.

    ``zero_tolerance`` is a reporting convention only; it is not a training
    threshold and defaults to exact zero.  The summary excludes invalid rows
    from fractions and reports them separately.
    """

    values = benefits.detach().float().reshape(-1)
    finite = torch.isfinite(values)
    if valid is not None:
        finite &= valid.detach().reshape(-1).to(device=finite.device, dtype=torch.bool)
    invalid = int((~finite).sum())
    values = values[finite]
    if values.numel() == 0:
        return {"count": 0, "invalid_count": invalid, "positive_count": 0, "zero_count": 0,
                "negative_count": 0, "positive_fraction": 0.0, "zero_fraction": 0.0,
                "negative_fraction": 0.0, "mean": 0.0, "std": 0.0}
    positive = values > zero_tolerance
    negative = values < -zero_tolerance
    zero = ~(positive | negative)
    count = int(values.numel())
    return {
        "count": count,
        "invalid_count": invalid,
        "positive_count": int(positive.sum()),
        "zero_count": int(zero.sum()),
        "negative_count": int(negative.sum()),
        "positive_fraction": float(positive.float().mean()),
        "zero_fraction": float(zero.float().mean()),
        "negative_fraction": float(negative.float().mean()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
    }


def audit_benefit_predictions(predicted: Tensor, actual: Tensor, *, valid: Tensor | None = None) -> dict[str, float | int]:
    """Audit a prospective router against measured paired benefits.

    This function reports evidence only.  It never enables UAMR and has no
    implicit correlation or accuracy threshold.
    """

    prediction = predicted.detach().float().reshape(-1)
    target = actual.detach().float().reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("predicted and actual benefits must have the same shape")
    keep = torch.isfinite(prediction) & torch.isfinite(target)
    if valid is not None:
        keep &= valid.detach().reshape(-1).to(device=keep.device, dtype=torch.bool)
    prediction = prediction[keep]; target = target[keep]
    if prediction.numel() == 0:
        return {"count": 0, "pearson": None, "mse": None, "mae": None, "sign_accuracy": None}
    p_centered = prediction - prediction.mean(); t_centered = target - target.mean()
    denominator = (p_centered.square().sum().sqrt() * t_centered.square().sum().sqrt()).clamp_min(1e-12)
    pearson = float((p_centered * t_centered).sum() / denominator)
    sign_accuracy = float((torch.sign(prediction) == torch.sign(target)).float().mean())
    return {
        "count": int(prediction.numel()),
        "pearson": pearson,
        "mse": float((prediction - target).square().mean()),
        "mae": float((prediction - target).abs().mean()),
        "sign_accuracy": sign_accuracy,
        "actual_positive_fraction": summarize_benefits(target)["positive_fraction"],
        "actual_zero_fraction": summarize_benefits(target)["zero_fraction"],
        "actual_negative_fraction": summarize_benefits(target)["negative_fraction"],
    }


def summarize_paired_labels(labels: PairedInterventionLabels, *, zero_tolerance: float = 0.0) -> dict[str, dict[str, float | int]]:
    """Return the required positive/zero/negative audit for both label scopes."""

    return {
        "full_image": summarize_benefits(labels.full_image_benefit, valid=labels.valid,
                                          zero_tolerance=zero_tolerance),
        "local": summarize_benefits(labels.local_benefit, valid=labels.local_valid & labels.valid,
                                     zero_tolerance=zero_tolerance),
    }

