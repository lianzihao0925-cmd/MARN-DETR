#!/usr/bin/env bash
# Low-data protocol: A / B / D over a per-class image subset.
#
# D is the method's real read path and is produced by running condition B with
# --read-policy similarity; B uses the vectorised deterministic random draw. Both
# are compared against A, which has no memory module at all.
#
# Usage:
#   bash tools/run_lowdata.sh                 # lr 1e-5 (the frozen protocol lr)
#   LR=1e-4 bash tools/run_lowdata.sh         # lr 1e-4, which creates ~3.6 AP of headroom
#
# Environment:
#   PROJECT_ROOT, PYTHON_BIN   see tools/preflight.sh
#   SUBSET_DIR                 image subset root (default /root/autodl-tmp/datasets/coco_fewshot)
#   OUT_ROOT                   output root (default $PROJECT_ROOT/ocpm_lowdata)
#   SEEDS                      default "42 43"
#   EPOCHS / SUBSET_IMAGES     default 25 / 958
#   LR / LR_ENCODER            optional learning-rate overrides
#   RHO / BETA_MAX             optional MGQR bound overrides (e.g. 0.5 and 1.0)
#   QPG                        optional queries-per-group override (300 = every query)

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/project_marn-detr}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/marn-detr/bin/python}
SUBSET_DIR=${SUBSET_DIR:-/root/autodl-tmp/datasets/coco_fewshot}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/ocpm_lowdata}
SEEDS=${SEEDS:-"42 43"}
EPOCHS=${EPOCHS:-25}
SUBSET_IMAGES=${SUBSET_IMAGES:-958}
LR=${LR:-}
LR_ENCODER=${LR_ENCODER:-}
RHO=${RHO:-}
BETA_MAX=${BETA_MAX:-}
QPG=${QPG:-}
cd "$PROJECT_ROOT" || exit 1

extra=()
[ -n "$LR" ] && extra+=(--lr "$LR")
[ -n "$LR_ENCODER" ] && extra+=(--lr-encoder "$LR_ENCODER")
[ -n "$RHO" ] && extra+=(--rho "$RHO")
[ -n "$BETA_MAX" ] && extra+=(--beta-max "$BETA_MAX")
[ -n "$QPG" ] && extra+=(--queries-per-group "$QPG")
tag=${LR:-protocol_lr}

run_one() {
  local arm=$1 seed=$2 cond=$3 policy=$4
  local dir="$OUT_ROOT/${arm}_s${seed}"
  if ls "$dir"/*_metrics.json >/dev/null 2>&1; then
    echo "[$(date +%T)] SKIP $arm s$seed"; return 0
  fi
  echo "[$(date +%T)] ===== $arm seed=$seed cond=$cond policy=$policy lr=$tag ====="
  "$PYTHON_BIN" experiments/ocpm_min/run_condition.py \
    --condition "$cond" --output "$dir" --seed "$seed" \
    --dataset-dir "$SUBSET_DIR" --expect-train-images "$SUBSET_IMAGES" \
    --epochs "$EPOCHS" --read-policy "$policy" "${extra[@]}"
  echo "[$(date +%T)] ===== $arm seed=$seed rc=$? ====="
}

for seed in $SEEDS; do
  run_one A "$seed" A random
  run_one B "$seed" B random
  run_one D "$seed" B similarity
done

echo "[$(date +%T)] LOWDATA_DONE"
