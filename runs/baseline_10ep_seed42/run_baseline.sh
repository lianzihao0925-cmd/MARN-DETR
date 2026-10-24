#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=/root/autodl-tmp/project_marn-detr/rf-detr
PYTHON_BIN=/root/miniconda3/envs/marn-detr/bin/python

cd "$REPO_DIR" || exit 1
test -x "$PYTHON_BIN" || exit 1
test -f "$RUN_DIR/train_baseline.py" || exit 1
command -v timeout >/dev/null || exit 1

export RF_HOME=/root/autodl-tmp/models/roboflow
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Reserve time for graceful shutdown before the four-hour ceiling.
timeout --signal=INT --kill-after=60s 14280s \
    "$PYTHON_BIN" -u "$RUN_DIR/train_baseline.py"
run_status=$?
printf '%s\n' "$run_status" > "$RUN_DIR/exit_code.txt"
exit "$run_status"
