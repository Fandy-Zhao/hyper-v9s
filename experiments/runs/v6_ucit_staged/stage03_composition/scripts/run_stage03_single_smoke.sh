#!/usr/bin/env bash
set -uo pipefail
cd /home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage03_composition
mkdir -p "$ROOT/smoke/single" "$ROOT/logs"
bash "$ROOT/configs/run_single_smoke.sh" > "$ROOT/logs/single_smoke_train.log" 2>&1
status=$?
printf '%s\n' "$status" > "$ROOT/smoke/single/train.exit"
{
  date -Is
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$ROOT/logs/gpu_after_single_smoke.txt" 2>&1
exit "$status"
