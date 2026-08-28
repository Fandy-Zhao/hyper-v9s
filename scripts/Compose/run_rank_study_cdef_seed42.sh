#!/usr/bin/env bash
set -euo pipefail
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
STUDY="$REPO/experiments/runs/compose_ucit_rank_study_seed42"
FORMAL="$REPO/experiments/runs/compose_ucit_formal_seed42"
cd "$REPO"

# C: deterministic 512-sample diagonal stage/task decomposition.
CROOT="$STUDY/C_oracle"
"$PY" -m compose.experiments.oracle_decomposition_seed42 prepare --output-root "$CROOT" >"$STUDY/C_prepare.log" 2>&1
for start in 0 4; do
  pids=()
  for ((offset=0; offset<4 && start+offset<6; offset++)); do
    stage=$((start+offset))
    CUDA_VISIBLE_DEVICES="$((4+offset))" "$PY" -m compose.experiments.oracle_decomposition_seed42 run-stage --formal-root "$FORMAL" --output-root "$CROOT" --stage "$stage" --device cuda:0 >"$STUDY/C_stage${stage}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
"$PY" -m compose.experiments.oracle_decomposition_seed42 summarize --output-root "$CROOT" >"$STUDY/C_summarize.log" 2>&1
cp "$CROOT/C_oracle_decomposition.csv" "$STUDY/C_oracle_decomposition.csv"
cp "$CROOT/C_oracle_decomposition_report.md" "$STUDY/C_oracle_decomposition_report.md"

# D: change only the effective RMS calibration values; reuse formal weights,
# keys, routes, clusters, data, generation configuration, and original scorer.
DROOT="$STUDY/D_rms"
"$PY" -m compose.experiments.rms_ablation_seed42 prepare --formal-root "$FORMAL" --output-root "$DROOT" >"$STUDY/D_prepare.log" 2>&1
run_d_stage() {
  local stage="$1" gpus="$2"
  "$PY" -m compose.eval.formal_ucit_eval --root "$DROOT/frozen_run" --stage-task "$stage" --config configs/compose_ucit.yaml --gpus "$gpus" --no-reuse-s11 >"$STUDY/D_stage${stage}.log" 2>&1
}
for stage in 0 1 2 3 4 5; do
  run_d_stage "$stage" 4,5,6,7
done
"$PY" -m compose.experiments.rms_ablation_seed42 summarize --formal-root "$FORMAL" --output-root "$DROOT" >"$STUDY/D_summarize.log" 2>&1
cp "$DROOT/D_rms_ablation_matrix.csv" "$STUDY/D_rms_ablation_matrix.csv"
cp "$DROOT/D_rms_ablation_metrics.json" "$STUDY/D_rms_ablation_metrics.json"
cp "$DROOT/D_rms_ablation_report.md" "$STUDY/D_rms_ablation_report.md"

# E: offline query clustering only; no expert training.
EROOT="$STUDY/E_clustering"
"$PY" -m compose.experiments.clustering_diagnostic_seed42 --formal-root "$FORMAL" --output-root "$EROOT" >"$STUDY/E_clustering.log" 2>&1
cp "$EROOT/E_clustering_algorithm_ablation.csv" "$STUDY/E_clustering_algorithm_ablation.csv"
cp "$EROOT/E_clustering_stability_report.md" "$STUDY/E_clustering_stability_report.md"

# F: teacher-forced offline analysis over A checkpoints only.
FROOT="$STUDY/F_specialization"
mkdir -p "$FROOT/details"
specs=("0:8" "0:16" "0:32" "1:8" "1:16" "1:32" "4:8" "4:16" "4:32")
for ((start=0; start<${#specs[@]}; start+=4)); do
  pids=()
  for ((offset=0; offset<4 && start+offset<${#specs[@]}; offset++)); do
    IFS=: read -r task rank <<<"${specs[$((start+offset))]}"
    run_root="$STUDY/A/task${task}_rank${rank}"
    CUDA_VISIBLE_DEVICES="$((4+offset))" "$PY" -m compose.experiments.rank_specialization_seed42 run --run-root "$run_root" --task-id "$task" --rank "$rank" --device cuda:0 --output "$FROOT/details/task${task}_rank${rank}.json" >"$STUDY/F_task${task}_rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
"$PY" -m compose.experiments.rank_specialization_seed42 summarize --study-root "$STUDY" --output-root "$FROOT" >"$STUDY/F_summarize.log" 2>&1
cp "$FROOT/F_rank_specialization.csv" "$STUDY/F_rank_specialization.csv"
cp "$FROOT/F_rank_specialization_report.md" "$STUDY/F_rank_specialization_report.md"
touch "$STUDY/CDEF.complete"
