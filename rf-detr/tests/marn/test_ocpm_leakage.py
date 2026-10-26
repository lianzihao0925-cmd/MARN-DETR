"""Minimal OCPM snapshot and phase-boundary leakage tests.

These tests intentionally do not add a new maintenance policy.  They exercise
the current snapshot/commit contract and make the missing model-version drift
metadata visible as a failing audit requirement.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rfdetr.marn.core import (
    MATCH_SOURCE_FINAL,
    UNKNOWN_CONF,
    UNKNOWN_FEATURE_LAYER,
    UNKNOWN_IOU,
    UNKNOWN_MATCH_SOURCE,
    UNKNOWN_WRITE_QUALITY,
    PrototypeBank,
    PrototypeReader,
    RetrievalConfig,
)
from rfdetr.marn.integration import MARNAdapter, MARNConfig


DIM = 64


def _adapter() -> MARNAdapter:
    adapter = MARNAdapter(
        DIM,
        [1],
        num_queries=2,
        config=MARNConfig(slots_per_class=2, queries_per_group=1, warmup_steps=0),
    )
    adapter.train()
    return adapter


def _candidate(value: float = 1.0, image_id: int = 1, object_id: int = 1):
    feature = torch.zeros(DIM)
    feature[0] = value
    geometry = torch.tensor([0.5, 0.5, 0.4, 0.4])
    return (
        0.8,  # quality_diag
        0.7,  # write_quality
        1,  # class
        feature,
        geometry,
        0,  # query id
        0.9,  # conf
        0.8,  # IoU
        {},
        0,  # feature_layer
        MATCH_SOURCE_FINAL,  # match_source
    ), (image_id, object_id)


def _commit_one(adapter: MARNAdapter, *, value: float = 1.0) -> None:
    adapter.snapshot = adapter.bank.snapshot(int(adapter.committed_step))
    candidate, key = _candidate(value=value)
    adapter.pending[key] = candidate
    adapter.observed = True
    adapter.commit()


def _read(adapter: MARNAdapter, snapshot):
    queries = torch.zeros(1, 1, DIM)
    queries[..., 0] = 1.0
    logits = torch.zeros(1, 1, 2)
    logits[..., 1] = 5.0
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
    selected = torch.ones(1, 1, dtype=torch.bool)
    return adapter.reader(queries, logits, boxes, selected, snapshot)


def _bank_state(bank: PrototypeBank):
    return {name: value.detach().clone() for name, value in bank.state_dict().items()}


def test_current_sample_cannot_read_its_pending_write():
    adapter = _adapter()
    old_snapshot = adapter.bank.snapshot(0)
    candidate, key = _candidate()
    adapter.snapshot = old_snapshot
    adapter.pending[key] = candidate
    adapter.observed = True

    _memory, _summary, accepted = _read(adapter, old_snapshot)

    assert not bool(accepted.any()), "pending candidate leaked into the pre-commit read"


def test_optimizer_window_reads_only_old_snapshot():
    adapter = _adapter()
    old_snapshot = adapter.bank.snapshot(0)
    adapter.bank.features[0, 0] = 1.0
    adapter.bank.geometry[0] = torch.tensor([0.5, 0.5, 0.4, 0.4])
    adapter.bank.valid[0] = True
    adapter.bank.updated_step[0] = 0
    _memory, _summary, accepted = _read(adapter, old_snapshot)

    assert not bool(accepted.any()), "bank mutation became visible through an old snapshot"


def test_commit_makes_prototype_visible_only_after_optimizer_boundary():
    adapter = _adapter()
    _commit_one(adapter)

    new_snapshot = adapter.bank.snapshot(int(adapter.committed_step))
    _memory, _summary, accepted = _read(adapter, new_snapshot)

    assert bool(accepted.any())
    assert int(adapter.committed_step) == 1


@pytest.mark.parametrize("phase", ["validation", "test"])
def test_validation_and_test_reads_do_not_mutate_bank(phase: str):
    adapter = _adapter()
    _commit_one(adapter)
    adapter.eval()
    before = _bank_state(adapter.bank)
    raw = torch.zeros(1, 2, DIM)
    normalized = raw.clone()
    reference = torch.zeros(1, 2, 4)
    heads = (nn.Linear(DIM, 2), nn.Linear(DIM, 4))
    with torch.no_grad():
        adapter(raw, normalized, reference, heads, False)
    after = _bank_state(adapter.bank)

    assert before.keys() == after.keys(), phase
    for name in before:
        assert torch.equal(before[name], after[name]), f"{phase} mutated bank field {name}"


def test_batch_write_order_cannot_change_snapshot_reads():
    adapter = _adapter()
    _commit_one(adapter)
    snapshot = adapter.bank.snapshot(int(adapter.committed_step))
    queries = torch.zeros(2, 1, DIM)
    queries[..., 0] = 1.0
    logits = torch.zeros(2, 1, 2)
    logits[..., 1] = 5.0
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]], [[0.5, 0.5, 0.4, 0.4]]])
    selected = torch.ones(2, 1, dtype=torch.bool)
    first = adapter.reader(queries, logits, boxes, selected, snapshot)
    reversed_batch = adapter.reader(queries.flip(0), logits.flip(0), boxes.flip(0), selected, snapshot)

    assert torch.equal(first[2], reversed_batch[2].flip(0))
    assert torch.allclose(first[0], reversed_batch[0].flip(0))


def test_old_checkpoint_missing_reliability_is_unknown_not_fabricated():
    bank = PrototypeBank([1], DIM, slots_per_class=1)
    legacy = {name: value.detach().clone() for name, value in bank.state_dict().items()}
    for name in (
        "conf", "iou", "write_quality", "reliability_known",
        "feature_layer", "match_source", "unknown_mask",
    ):
        legacy.pop(name)
    restored = PrototypeBank([1], DIM, slots_per_class=1)
    restored.load_state_dict(legacy, strict=True)

    assert torch.isnan(restored.conf).all()
    assert torch.isnan(restored.iou).all()
    assert torch.isnan(restored.write_quality).all()
    assert not bool(restored.reliability_known.any())
    expected = UNKNOWN_CONF | UNKNOWN_IOU | UNKNOWN_WRITE_QUALITY
    expected |= UNKNOWN_FEATURE_LAYER | UNKNOWN_MATCH_SOURCE
    assert torch.all(restored.unknown_mask == expected)


def test_quality_diag_is_not_a_maintenance_fallback():
    adapter = _adapter()
    candidate, key = _candidate()
    candidate = (*candidate[:1], float("nan"), *candidate[2:])
    adapter.pending[key] = candidate
    assert adapter._candidate_priority(candidate) == (0, 0.0)


def test_nonfinite_and_empty_slots_are_not_read():
    bank = PrototypeBank([1], 4, slots_per_class=3)
    bank.valid[:] = True
    bank.features[0, 0] = float("nan")
    bank.geometry[1, 0] = float("inf")
    bank.valid[2] = False
    reader = PrototypeReader(4, [1], RetrievalConfig(key_dim=2, top_k=2, min_cosine=-1.0))
    snapshot = bank.snapshot(0)
    queries = torch.zeros(1, 1, 4)
    queries[..., 0] = 1.0
    logits = torch.zeros(1, 1, 2)
    logits[..., 1] = 5.0
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
    memory, _summary, accepted = reader(queries, logits, boxes, torch.ones(1, 1, dtype=torch.bool), snapshot)

    assert not bool(accepted.any())
    assert torch.equal(memory, torch.zeros_like(memory))


def test_model_version_change_marks_old_prototypes_as_possible_drift():
    """Audit requirement currently unmet: bank has no model/projection version marker."""

    bank = PrototypeBank([1], DIM, slots_per_class=1)
    snapshot = bank.snapshot(0)
    assert hasattr(snapshot, "feature_version"), "no feature/model version metadata exists"
    assert hasattr(snapshot, "drift_possible"), "old prototypes are not explicitly marked as drift candidates"

