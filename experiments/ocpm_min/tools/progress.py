#!/usr/bin/env python3
"""Live dashboard for the OCPM A/B/C run. Redraws in place every few seconds."""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/root/autodl-tmp/project_marn-detr/ocpm_runs")
STEPS_PER_EPOCH = 7393
TOTAL_STEPS = 3 * STEPS_PER_EPOCH
CHECKPOINT_EVERY = 3

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"


def read_status() -> str:
    try:
        return (OUT.parent / ".." / ".." / "abc.status").read_text().strip()
    except Exception:
        pass
    try:
        return Path("/root/abc.status").read_text().strip()
    except Exception:
        return "unknown"


def last_metric(condition: str):
    path = OUT / condition / "metrics.csv"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            return None
        header, last = rows[0], rows[-1]
        record = dict(zip(header, last))
        return {
            "epoch": int(float(record.get("epoch", 0) or 0)),
            "step": int(float(record.get("step", 0) or 0)),
            "loss": record.get("train/loss"),
        }
    except Exception:
        return None


def val_points(condition: str):
    """Per-epoch validation AP already written by the COCO eval callback."""
    points = {}
    folder = OUT / condition
    if not folder.is_dir():
        return points
    for name in sorted(folder.glob("coco_epoch_*_regular.json")):
        try:
            idx = int(name.stem.split("_")[2])
        except Exception:
            continue
        try:
            payload = json.loads(name.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        ap = payload.get("map") or payload.get("mAP") or payload.get("AP")
        points[idx] = ap
    return points


def final_ap(condition: str):
    """Final val5000 AP written after training finishes (works for condition A too)."""
    path = OUT / condition / f"{condition}_metrics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        value = metrics.get("AP")
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        util, used, total, power, temp = [v.strip() for v in out.split(",")]
        return util, used, total, power, temp
    except Exception:
        return "?", "?", "?", "?", "?"


def bar(fraction: float, width: int = 28) -> str:
    done = int(max(0.0, min(1.0, fraction)) * width)
    return "█" * done + "░" * (width - done)


def hhmm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def rate_for(condition: str):
    """Steps per second from the last two samples in monitor.csv for this condition."""
    path = OUT / "monitor.csv"
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[1:]
        samples = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1] == condition:
                try:
                    samples.append((parts[0], int(parts[2])))
                except Exception:
                    continue
        if len(samples) < 2:
            return None
        fmt = "%H:%M:%S"
        now = time.time()
        # only trust samples from the last 5 minutes so a pause is not averaged in
        recent = []
        for stamp, step in samples:
            try:
                moment = time.mktime(time.strptime(stamp, fmt))
            except Exception:
                continue
            if now - moment <= 300:
                recent.append((moment, step))
        if len(recent) < 2:
            recent = [(time.mktime(time.strptime(t, fmt)), s) for t, s in samples[-6:]]
        (t0, s0), (t1, s1) = recent[0], recent[-1]
        d = t1 - t0
        if d <= 0 or s1 < s0:
            return None
        return (s1 - s0) / d
    except Exception:
        return None


def render() -> str:
    status = read_status()
    util, used, total, power, temp = gpu()
    lines = []
    lines.append(f"{BOLD}{CYAN}══════════════ MARN-DETR · OCPM  A / B / C  监控 ══════════════{RESET}")
    colour = GREEN if status == "ALL_DONE" else (RED if "STOPPED" in status else YELLOW)
    lines.append(f"  状态 {BOLD}{colour}{status}{RESET}    时间 {time.strftime('%H:%M:%S')}")
    lines.append("")

    current = None
    for c in ("A", "B", "C"):
        m = last_metric(c)
        if m is not None:
            current = c
    for c in ("A", "B", "C"):
        m = last_metric(c)
        if m is None:
            lines.append(f"  {DIM}{c}  等待中{RESET}")
            continue
        total_step = min(TOTAL_STEPS, m["epoch"] * STEPS_PER_EPOCH + m["step"])
        frac = min(1.0, total_step / TOTAL_STEPS)
        epoch_no = m["epoch"] + 1
        rate = rate_for(c)
        if c == current and rate:
            remain = (TOTAL_STEPS - total_step) / rate if rate > 0 else 0
            eta = f"  速度 {rate:.1f} 步/秒  剩余 {hhmm(remain)}"
        else:
            eta = ""
        mark = f" {GREEN}◀ 运行中{RESET}" if c == current and status != "ALL_DONE" else ""
        lines.append(f"  {BOLD}{c}{RESET}  [{bar(frac)}] {frac * 100:5.1f}%   "
                     f"epoch {epoch_no}/3  step {total_step}/{TOTAL_STEPS}{mark}")
        loss = m.get("loss")
        loss_txt = f"loss {float(loss):.3f}" if loss not in (None, "") else ""
        lines.append(f"      {DIM}{loss_txt}{eta}{RESET}")
    lines.append("")
    lines.append(f"  GPU {util}%   {int(used) if used.isdigit() else used}/{total} MiB   功耗 {power}W   温度 {temp}°C")
    lines.append("")
    lines.append(f"  {BOLD}验证点（val5000 AP）{RESET}")
    for c in ("A", "B", "C"):
        pts = val_points(c)
        final = final_ap(c)
        if not pts and final is None:
            lines.append(f"    {DIM}{c}: 尚未产生{RESET}")
            continue
        cells = "  ".join(f"ep{k} {v:.4f}" if isinstance(v, (int, float)) else f"ep{k} -" for k, v in pts.items())
        head = f"    {BOLD}{c}{RESET}: "
        if final is not None:
            head += f"{GREEN}最终 AP {final:.4f}{RESET}"
        if cells:
            head += f"   {DIM}({cells}){RESET}"
        lines.append(head)
    lines.append("")
    lines.append(f"{DIM}  Ctrl+C 退出（不影响训练）{RESET}")
    return "\n".join(lines)


def main() -> None:
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    try:
        while True:
            frame = render()
            sys.stdout.write(CLEAR + frame + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
