"""Dependency-free audit for the prototype drift-version contract."""

from pathlib import Path


def test_bank_exposes_model_or_projection_version_for_drift_marking():
    core = Path(__file__).parents[2] / "src" / "rfdetr" / "marn" / "core.py"
    source = core.read_text(encoding="utf-8")
    assert any(token in source for token in ("feature_version", "model_version", "drift_possible")), (
        "PrototypeBank has update_count only; it cannot mark prototypes stale after a model/projection version change"
    )


def test_bank_persists_provenance_and_unknown_mask():
    source = Path(__file__).parents[2].joinpath("src/rfdetr/marn/core.py").read_text(encoding="utf-8")
    for token in ("feature_layer", "match_source", "unknown_mask", "UNKNOWN_CONF", "UNKNOWN_IOU"):
        assert token in source, f"missing persisted metadata token: {token}"


def test_quality_diag_cannot_be_used_as_maintenance_fallback():
    source = Path(__file__).parents[2].joinpath("src/rfdetr/marn/integration.py").read_text(encoding="utf-8")
    assert "quality_fallback" not in source
    assert "_quality_diag, write_quality" not in source

