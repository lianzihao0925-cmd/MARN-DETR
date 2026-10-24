#!/bin/bash
set +e
cd /root/autodl-tmp/project_marn-detr/experiments/marn_c4
/root/miniconda3/envs/marn-detr/bin/python -u run_pair.py --gpu-hours 4
status=$?
printf '%s\n' "$status" > "/root/autodl-tmp/project_marn-detr/runs/codex_operations/pair_4h_20261016_225833/exit_code"
exit "$status"
