"""Build the A/B/C comparison report from the per-condition JSON artifacts.

    python experiments/ocpm_min/compare.py --runs /root/autodl-tmp/project_marn-detr/ocpm_runs

The report header always carries the single-seed caveat; this script refuses to
compare runs whose protocol invariants differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONDITIONS = ("A", "B", "C")

PREAMBLE = (
    "> **单 seed 短期筛选，不构成稳定 AP 改善结论。**\n"
    "> 本表只用于确认三组使用完全相同的初始 checkpoint、种子、数据顺序、训练步数与评估函数；\n"
    "> 任何 AP 差异都必须在多 seed 重复后才可作为结论。\n"
)


def _load(path: Path) -> dict:
    """Read one JSON artifact, failing loudly when it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 4) -> str:
    """Format an optional number for the markdown table."""
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _check_invariants(runs: dict[str, dict]) -> None:
    """Refuse to compare runs that do not share the frozen protocol."""
    reference = runs["A"]
    for name in ("weights_md5", "seed", "global_step", "train_config_sha256", "data_order_fingerprint"):
        values = {condition: runs[condition][name] for condition in CONDITIONS}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            raise AssertionError(f"conditions disagree on {name}: {values}")
    if reference["weights_md5"] != "fb37061c1af7bace359c91b723a8d5c1":
        raise AssertionError(f"unexpected initial checkpoint md5: {reference['weights_md5']}")


def main() -> int:
    """Render ``comparison_report.md`` next to the run directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="directory containing A/B/C subdirectories")
    parser.add_argument("--output", default=None, help="report path (default: <runs>/comparison_report.md)")
    args = parser.parse_args()

    root = Path(args.runs)
    metrics = {condition: _load(root / condition / f"{condition}_metrics.json") for condition in CONDITIONS}
    banks = {condition: _load(root / condition / f"{condition}_bank.json") for condition in CONDITIONS}
    resources = {condition: _load(root / condition / f"{condition}_resources.json") for condition in CONDITIONS}
    quality = {condition: _load(root / condition / f"{condition}_quality.json") for condition in CONDITIONS}
    _check_invariants(metrics)

    detection_rows = [
        ("AP", "AP"),
        ("AP50", "AP50"),
        ("AP75", "AP75"),
        ("APs", "APs"),
        ("APm", "APm"),
        ("APl", "APl"),
    ]
    bank_rows = [
        ("bank fill rate", "fill_rate"),
        ("class coverage", "class_coverage"),
        ("duplicate rate", "duplicate_rate"),
        ("age p0", ("age_quantiles", "p0")),
        ("age p50", ("age_quantiles", "p50")),
        ("age p100", ("age_quantiles", "p100")),
        ("read events", "read_events"),
    ]

    def bank_value(condition: str, key) -> object:
        payload = banks[condition]
        if not payload.get("applicable", True):
            return None
        if isinstance(key, tuple):
            return payload.get(key[0], {}).get(key[1])
        return payload.get(key)

    lines = ["# OCPM 最小可比较实验（A/B/C）", "", PREAMBLE, ""]
    lines.append("## 协议校验")
    lines.append("")
    lines.append("| 项 | A | B | C |")
    lines.append("|---|---|---|---|")
    for name in ("weights_md5", "seed", "global_step", "train_config_sha256", "data_order_fingerprint"):
        lines.append(
            "| " + name + " | " + " | ".join(_fmt(metrics[c][name], 0) for c in CONDITIONS) + " |"
        )
    lines.append("")

    lines.append("## 检测指标（COCO val2017, maxDets=100）")
    lines.append("")
    lines.append("| 指标 | A | B | C |")
    lines.append("|---|---:|---:|---:|")
    for label, key in detection_rows:
        lines.append(
            "| " + label + " | " + " | ".join(_fmt(metrics[c]["metrics"].get(key)) for c in CONDITIONS) + " |"
        )
    lines.append("")

    lines.append("## Bank / 读取")
    lines.append("")
    lines.append("| 指标 | A | B | C |")
    lines.append("|---|---:|---:|---:|")
    for label, key in bank_rows:
        lines.append("| " + label + " | " + " | ".join(_fmt(bank_value(c, key)) for c in CONDITIONS) + " |")
    for label, key in (("fill operations", "fill"), ("update operations", "update"), ("replace operations", "replace")):
        cells = []
        for condition in CONDITIONS:
            payload = banks[condition]
            cumulative = payload.get("cumulative_operations") or {}
            cells.append(_fmt(cumulative.get(key)))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 质量列")
    lines.append("")
    lines.append("| 指标 | A | B | C |")
    lines.append("|---|---:|---:|---:|")
    for label, key in (
        ("Q_write evaluated", ("components", "Q_write", "count")),
        ("Q_write unknown", ("components", "Q_write", "unknown")),
        ("conf known slots", "bank_conf_known"),
        ("IoU known slots", "bank_iou_known"),
        ("write_quality known slots", "bank_write_quality_known"),
    ):
        cells = []
        for condition in CONDITIONS:
            payload = quality[condition]
            if not payload.get("applicable", True):
                cells.append("NA")
                continue
            value = payload
            for part in key:
                value = value.get(part) if isinstance(value, dict) else None
            cells.append(_fmt(value, 0))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 资源")
    lines.append("")
    lines.append("| 指标 | A | B | C |")
    lines.append("|---|---:|---:|---:|")
    for label, key in (
        ("wall seconds", "wall_seconds"),
        ("batch seconds mean", "batch_seconds_mean"),
        ("peak allocated GiB", "peak_allocated_GiB"),
        ("peak reserved GiB", "peak_reserved_GiB"),
    ):
        lines.append(
            "| " + label + " | " + " | ".join(_fmt(resources[c].get(key), 2) for c in CONDITIONS) + " |"
        )
    lines.append("")

    lines.append("## 边界说明")
    lines.append("")
    lines.append("- B/C 的读取策略是**确定性随机槽位选择**，不使用 cosine/class/geometry 打分，因此三者差异只反映")
    lines.append("  「是否写入并维护原型库」，不反映「检索相关性」。")
    lines.append("- `quality_diag` 仅作诊断，不参与排序、更新或淘汰；`Q_write` 只记录，不参与检索。")
    lines.append("- 原型填充率、类别覆盖率、相似度都不等价于 AP 收益，不能单独作为结论。")
    lines.append("- 本报告不含 UAMR，也不含完整 A–H 矩阵。")
    lines.append("")

    output = Path(args.output) if args.output else root / "comparison_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
