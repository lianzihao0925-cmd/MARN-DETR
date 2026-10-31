#!/usr/bin/env bash
# Controlled comparison: condition A then B then C, one full protocol run each.
#
# The runner is skip-aware (a condition whose {C}_metrics.json already exists is
# not repeated) and retries a failed condition once, so an interrupted session can
# be resumed by simply starting this script again.
#
# Usage:
#   bash tools/run_abc.sh                 # 3 epochs per condition (the frozen protocol)
#   STEPS=2000 bash tools/run_abc.sh      # exactly 2000 steps per condition
#
# Environment:
#   PROJECT_ROOT, PYTHON_BIN   see tools/preflight.sh
#   OUT_ROOT                   output root (default $PROJECT_ROOT/ocpm_runs)
#   EPOCHS / STEPS             protocol size; STEPS>0 switches to
#                              "--epochs 1 --limit-batches STEPS"

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/project_marn-detr}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/marn-detr/bin/python}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/ocpm_runs}
EPOCHS=${EPOCHS:-3}
STEPS=${STEPS:-0}
cd "$PROJECT_ROOT" || exit 1

step_args() {
  if [ "$STEPS" -gt 0 ] 2>/dev/null; then
    printf '%s\n' --epochs 1 --limit-batches "$STEPS"
  else
    printf '%s\n' --epochs "$EPOCHS"
  fi
}

# mapfile keeps the two switch words together regardless of spacing
mapfile -t EXTRA < <(step_args)

bash "$PROJECT_ROOT/experiments/ocpm_min/tools/preflight.sh" || {
  echo "PREFLIGHT_FAILED"; exit 10;
}

for C in A B C; do
  attempt=0
  ok=0
  while [ "$attempt" -lt 2 ] && [ "$ok" -eq 0 ]; do
    attempt=$((attempt + 1))
    if ls "$OUT_ROOT/$C"/*_metrics.json >/dev/null 2>&1; then
      echo "[$(date +%T)] SKIP $C (already finished)"; ok=1; break
    fi
    echo "[$(date +%T)] ===== CONDITION $C attempt $attempt START ====="
    "$PYTHON_BIN" experiments/ocpm_min/run_condition.py \
      --condition "$C" --output "$OUT_ROOT/$C" "${EXTRA[@]}"
    rc=$?
    echo "[$(date +%T)] ===== CONDITION $C attempt $attempt rc=$rc ====="
    [ "$rc" -eq 0 ] && ok=1
  done
  if [ "$ok" -eq 0 ]; then
    echo "STOPPED_CONDITION_$C"; exit "$rc"
  fi
done

echo "[$(date +%T)] ===== COMPARE ====="
"$PYTHON_BIN" experiments/ocpm_min/compare.py --runs "$OUT_ROOT"
echo "[$(date +%T)] ===== DONE ====="
