#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
STUDY="$REPO/experiments/runs/compose_ucit_rank_study_seed42"
cd "$REPO"

phase="${1:?usage: run_rank_study_seed42.sh A|B}"

run_one() {
  local root="$1" task="$2" gpus="$3"
  local log="$root/driver.log"
  for gpu in ${gpus//,/ }; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    [[ "$used" -lt 1000 ]] || { echo "GPU $gpu busy (${used} MiB)" >&2; return 1; }
  done
  "$PY" -m compose.experiments.task_run \
    --config "$root/config.yaml" --root "$root" \
    --first-task "$task" --last-task "$task" --gpus "$gpus" \
    >"$log" 2>&1
}

if [[ "$phase" == A ]]; then
  runs=(
    "$STUDY/A/task0_rank8:0" "$STUDY/A/task0_rank16:0"
    "$STUDY/A/task0_rank32:0" "$STUDY/A/task1_rank8:1"
    "$STUDY/A/task1_rank16:1" "$STUDY/A/task1_rank32:1"
    "$STUDY/A/task4_rank8:4" "$STUDY/A/task4_rank16:4"
  )
  for item in "${runs[@]}"; do
    IFS=: read -r root task <<<"$item"
    run_one "$root" "$task" "4,5,6,7"
  done
  run_one "$STUDY/A/task4_rank32" 4 "4,5,6,7"
elif [[ "$phase" == B ]]; then
  run_one "$STUDY/B/task0_single_rank8" 0 "4,5,6,7"
  run_one "$STUDY/B/task0_single_rank16" 0 "4,5,6,7"
  run_one "$STUDY/B/task0_single_rank32" 0 "4,5,6,7"
else
  echo "unknown phase: $phase" >&2; exit 2
fi
