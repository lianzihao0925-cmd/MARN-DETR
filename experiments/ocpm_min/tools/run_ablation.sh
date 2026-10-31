#!/usr/bin/env bash
# Ablations that were used to rule out alternative explanations for the null result.
#
#   wide    the MGQR residual budget is widened 10x (rho 0.1 -> 0.5, beta_max 0.5 -> 1.0)
#   cover   every query is refined instead of 30 of 300 (queries_per_group 30 -> 300)
#   lr      the low-data protocol is rerun at lr 1e-4 to create measurable headroom
#
# Usage:
#   bash tools/run_ablation.sh wide
#   bash tools/run_ablation.sh cover
#   bash tools/run_ablation.sh lr
#
# All three build on tools/run_lowdata.sh so the only differences are the flags.

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-}

case "$MODE" in
  wide)
    echo "== wide MGQR budget: rho 0.5, beta_max 1.0, lr 1e-4 =="
    LR=1e-4 LR_ENCODER=1.5e-4 RHO=0.5 BETA_MAX=1.0 SEEDS="42" \
      bash "$HERE/run_lowdata.sh"
    ;;
  cover)
    echo "== full query coverage: queries_per_group 300, wide budget, random read =="
    LR=1e-4 LR_ENCODER=1.5e-4 RHO=0.5 BETA_MAX=1.0 QPG=300 SEEDS="42" \
      bash "$HERE/run_lowdata.sh"
    ;;
  lr)
    echo "== low-data protocol at lr 1e-4 =="
    LR=1e-4 LR_ENCODER=1.5e-4 SEEDS="42 43" \
      bash "$HERE/run_lowdata.sh"
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 2
    ;;
esac

echo "[$(date +%T)] ABLATION_${MODE}_DONE"
