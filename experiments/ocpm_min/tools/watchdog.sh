#!/usr/bin/env bash
# Read-only health monitor for a long run.
#
# Every INTERVAL seconds it samples the running condition, its step counter, GPU
# utilisation and free disk, writes one JSON status file and appends events to a
# log. If the step counter stops advancing for STALL_SECONDS while the status file
# says a run is active, it kills the runner and restarts it through
# $RESTART_CMD, at most twice; after that it marks the status file STOPPED_STALL_LIMIT
# and leaves the evidence in place.
#
# The monitor never touches training output: it only writes health.json and
# events.log next to the run roots.
#
# Usage:
#   STATUS_FILE=/root/abc.status RUNS=/root/autodl-tmp/project_marn-detr/ocpm_runs \
#     RESTART_CMD=/root/run_abc.sh bash tools/watchdog.sh

set -uo pipefail

BASE=${BASE:-/root/autodl-tmp/project_marn-detr}
HEALTH_DIR=${HEALTH_DIR:-$BASE/ocpm_runs}
RUNS=${RUNS:-$HEALTH_DIR}
STATUS_FILE=${STATUS_FILE:-/root/abc.status}
RESTART_CMD=${RESTART_CMD:-}
INTERVAL=${INTERVAL:-60}
STALL_SECONDS=${STALL_SECONDS:-600}
STATE_FILE=${STATE_FILE:-/root/.watchdog_state}
RESTART_COUNTER=${RESTART_COUNTER:-/root/.watchdog_restarts}

mkdir -p "$HEALTH_DIR"
HEALTH=$HEALTH_DIR/health.json
EVENTS=$HEALTH_DIR/events.log

log_event() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$EVENTS"; }

current_condition() {
  ps -eo args | grep -o 'run_condition.py --condition [ABC]' | head -1 | awk '{print $NF}'
}

last_step() {
  local f="$RUNS/$1/metrics.csv"
  [ -f "$f" ] && tail -1 "$f" | cut -d, -f2
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

while true; do
  cond=$(current_condition)
  status=$(cat "$STATUS_FILE" 2>/dev/null || echo unknown)
  gpu=$(gpu_util)
  disk=$(df -BG "$HEALTH_DIR" | awk 'NR==2{gsub("G","",$4); print $4}')
  now=$(date +%s)
  step=""
  stalled=0
  action=none

  if [ -n "$cond" ]; then
    step=$(last_step "$cond")
    prev_step=$(grep -s "^$cond " "$STATE_FILE" | awk '{print $2}')
    prev_time=$(grep -s "^$cond " "$STATE_FILE" | awk '{print $3}')
    if [ -z "$prev_step" ] || [ -z "$prev_time" ] || [ "$step" != "$prev_step" ]; then
      sed -i "/^$cond /d" "$STATE_FILE" 2>/dev/null || true
      echo "$cond ${step:-0} $now" >> "$STATE_FILE"
      prev_time=$now
    fi
    idle=$(( now - prev_time ))
    if [ "$idle" -ge "$STALL_SECONDS" ]; then
      stalled=1
      count=$(cat "$RESTART_COUNTER" 2>/dev/null || echo 0)
      log_event "STALL cond=$cond step=$step idle=${idle}s restarts=$count"
      if [ "$count" -lt 2 ] && [ -n "$RESTART_CMD" ]; then
        echo $(( count + 1 )) > "$RESTART_COUNTER"
        action=restart
        pkill -9 -f "run_condition.py --condition" 2>/dev/null
        sleep 3
        setsid nohup "$RESTART_CMD" > /root/watchdog_restart.log 2>&1 < /dev/null &
        log_event "RESTARTED $RESTART_CMD"
      else
        action=stop
        echo "STOPPED_STALL_LIMIT" > "$STATUS_FILE"
        log_event "STALL_LIMIT_REACHED stopping"
      fi
    fi
  fi

  cat > "$HEALTH" <<JSON
{"ts":"$(date '+%F %T')","status":"$status","condition":"${cond:-none}","step":"${step}","gpu_util":"$gpu","disk_free_gb":"$disk","stalled":$stalled,"action":"$action","restarts":$(cat "$RESTART_COUNTER" 2>/dev/null || echo 0)}
JSON

  sleep "$INTERVAL"
done
