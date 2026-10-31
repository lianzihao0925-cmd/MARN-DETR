#!/usr/bin/env bash
# Preflight gate: refuse to start a run unless the host is in the expected state.
#
# Every check prints OK or FAIL and the script exits non-zero on the first
# failure, so a half-prepared environment can never launch a training run.
#
# Overridable through the environment:
#   PROJECT_ROOT  repository checkout        (default /root/autodl-tmp/project_marn-detr)
#   PYTHON_BIN    interpreter with rfdetr    (default /root/miniconda3/envs/marn-detr/bin/python)
#   WEIGHTS       pretrained checkpoint      (default /root/autodl-tmp/models/roboflow/rf-detr-small.pth)

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/project_marn-detr}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/marn-detr/bin/python}
WEIGHTS=${WEIGHTS:-/root/autodl-tmp/models/roboflow/rf-detr-small.pth}
WEIGHT_MD5=fb37061c1af7bace359c91b723a8d5c1
TRAIN_IMAGES=118287
VAL_IMAGES=5000
fail=0

say() { printf '%s\n' "$*"; }

say "===== PREFLIGHT ====="

say "-- 1. no leftover training process"
if pgrep -f "run_condition.py --condition" > /dev/null; then
  say "   FAIL: a run_condition.py process is already alive"; fail=1
else
  say "   OK"
fi

say "-- 2. GPU is idle"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
if [ -n "$used" ] && [ "$used" -lt 2000 ]; then
  say "   OK (used=${used}MiB)"
else
  say "   FAIL: GPU memory in use (${used:-unknown}MiB)"; fail=1
fi

say "-- 3. free disk"
free_g=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${free_g:-0}" -gt 80 ]; then
  say "   OK (${free_g}G free)"
else
  say "   FAIL: only ${free_g:-0}G free"; fail=1
fi

say "-- 4. pretrained weights md5"
if [ -f "$WEIGHTS" ]; then
  md5=$(md5sum "$WEIGHTS" | cut -d' ' -f1)
  if [ "$md5" = "$WEIGHT_MD5" ]; then say "   OK"; else say "   FAIL: md5=$md5"; fail=1; fi
else
  say "   FAIL: $WEIGHTS missing"; fail=1
fi

say "-- 5. dataset image counts"
tr=$(ls "$PROJECT_ROOT/datasets/coco/train2017" 2>/dev/null | wc -l)
va=$(ls "$PROJECT_ROOT/datasets/coco/val2017" 2>/dev/null | wc -l)
if [ "$tr" = "$TRAIN_IMAGES" ] && [ "$va" = "$VAL_IMAGES" ]; then
  say "   OK (train=$tr val=$va)"
else
  say "   FAIL: train=$tr val=$va"; fail=1
fi

say "-- 6. imports and CUDA"
if "$PYTHON_BIN" - <<'PY'
import torch, torchvision, pytorch_lightning, transformers, supervision, timm
import pycocotools, cv2, scipy, matplotlib, torchmetrics, rfdetr, triton
assert torch.cuda.is_available(), "CUDA is not available"
print("   OK", torch.__version__, torch.cuda.get_device_name(0))
PY
then :; else say "   FAIL: import or CUDA check failed"; fail=1; fi

say "-- 7. git status (only untracked experiment output is expected)"
( cd "$PROJECT_ROOT" && git status --short | head -5 )

say "-- 8. frozen protocol"
if "$PYTHON_BIN" - <<'PY'
import sys
root = "/root/autodl-tmp/project_marn-detr"
sys.path.insert(0, root); sys.path.insert(0, root + "/rf-detr/src")
from experiments.ocpm_min import common
tc = common.build_train_config("/tmp/_preflight")
assert tc.batch_size == 16 and tc.grad_accum_steps == 1 and tc.amp_dtype == "bf16"
assert tc.use_ema and tc.warmup_epochs == 0.5 and tc.lr_scheduler == "cosine"
assert tc.eval_interval == 1 and tc.eval_max_dets == 100
print(f"   OK epochs={tc.epochs} batch={tc.batch_size} amp={tc.amp_dtype} ema={tc.use_ema}")
PY
then :; else say "   FAIL: frozen protocol check failed"; fail=1; fi

say "===== PREFLIGHT RESULT: fail=$fail ====="
exit "$fail"
