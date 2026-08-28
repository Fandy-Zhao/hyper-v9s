#!/usr/bin/env bash
set -euo pipefail
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
STUDY="$REPO/experiments/runs/compose_ucit_rank_study_seed42"
cd "$REPO"
runs=(
  "A/task0_rank8:0" "A/task0_rank16:0" "A/task0_rank32:0"
  "A/task1_rank8:1" "A/task1_rank16:1" "A/task1_rank32:1"
  "A/task4_rank8:4" "A/task4_rank16:4" "A/task4_rank32:4"
  "B/task0_single_rank8:0" "B/task0_single_rank16:0" "B/task0_single_rank32:0"
)

for item in "${runs[@]}"; do
  IFS=: read -r relative task <<<"$item"
  root="$STUDY/$relative"
  "$PY" -m compose.experiments.rank_study_metrics score-test --run-root "$root" --task-id "$task" >"$root/analysis_score_test.log" 2>&1
done

run_nll() {
  local item="$1" gpu="$2"
  IFS=: read -r relative task <<<"$item"
  local root="$STUDY/$relative"
  mkdir -p "$root/analysis"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m compose.experiments.rank_study_eval nll --run-root "$root" --task-id "$task" --split train --device cuda:0 --output "$root/analysis/nll_train.json" >"$root/analysis/nll_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m compose.experiments.rank_study_eval nll --run-root "$root" --task-id "$task" --split val --device cuda:0 --output "$root/analysis/nll_val.json" >"$root/analysis/nll_val.log" 2>&1
}
for ((start=0; start<${#runs[@]}; start+=4)); do
  pids=()
  for ((offset=0; offset<4 && start+offset<${#runs[@]}; offset++)); do
    run_nll "${runs[$((start+offset))]}" "$((4+offset))" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

run_generation() {
  local item="$1" gpus="$2"
  IFS=: read -r relative task <<<"$item"
  local root="$STUDY/$relative"
  "$PY" -m compose.experiments.rank_study_metrics eval-split --run-root "$root" --task-id "$task" --split train --gpus "$gpus" >"$root/analysis_train_generation.log" 2>&1
  "$PY" -m compose.experiments.rank_study_metrics eval-split --run-root "$root" --task-id "$task" --split val --gpus "$gpus" >"$root/analysis_val_generation.log" 2>&1
}
for item in "${runs[@]}"; do
  run_generation "$item" "4,5,6,7"
done

"$PY" -m compose.experiments.rank_study_metrics summarize --study-root "$STUDY" >"$STUDY/AB_summarize.log" 2>&1
touch "$STUDY/AB_metrics.complete"
