#!/bin/bash
set -uo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage02_registry
LOG=$ROOT/logs/smoke_gb24_retry2.log
EXIT_FILE=$ROOT/smoke/retry2/smoke.exit
mkdir -p "$ROOT/smoke/retry2"

bash "$ROOT/configs/run_smoke_gb24_retry2.sh" >"$LOG" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" >"$EXIT_FILE"
{
  date -Is
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} >"$ROOT/logs/gpu_after_smoke_retry2.txt" 2>&1
exit "$STATUS"
