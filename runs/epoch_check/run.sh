#!/usr/bin/env bash
cd /root/autodl-tmp/project_marn-detr/rf-detr || exit 1
/root/miniconda3/envs/marn-detr/bin/python -u \
  ../runs/epoch_check/check_epoch.py
run_status=$?
printf '%s\n' "$run_status" > ../runs/epoch_check/exit_code.txt
exit "$run_status"
