#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/zhaozhuofan/Hyper-LlaVA"
STAGE="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2"
EVAL_VARIANT="evaluation_v2"
export EVAL_VARIANT
until [[ -f "${STAGE}/training/matrix_complete_utc.txt" ]]; do sleep 20; done
if [[ ! -f "${STAGE}/diagnostics/complete_utc.txt" ]]; then
  bash "${STAGE}/run_diagnostics.sh"
fi
for seed in 42 43 44; do bash "${STAGE}/assemble_seed.sh" "${seed}"; done

jobs=()
add_jobs() {
  local seed="$1" dataset="$2"
  shift 2
  local model
  for model in "$@"; do jobs+=("${model} ${seed} ${dataset}"); done
}
for seed in 42 43 44; do
  add_jobs "${seed}" A_only base expert_a residual_b a_residual_b
  add_jobs "${seed}" B_only base independent_b residual_b
  add_jobs "${seed}" C_only base expert_c residual_b residual_b_c
  add_jobs "${seed}" A_plus_B base expert_a independent_b residual_b a_independent_b a_residual_b rank16_ab task_ab
  add_jobs "${seed}" B_plus_C base independent_b residual_b expert_c independent_b_c residual_b_c rank16_ab upper_bc
done
printf '%s\n' "${jobs[@]}" > "${STAGE}/${EVAL_VARIANT}_jobs.txt"

worker() {
  local gpu="$1" index model seed dataset
  for ((index=gpu; index<${#jobs[@]}; index+=8)); do
    read -r model seed dataset <<< "${jobs[$index]}"
    echo "GPU ${gpu} START ${model} seed${seed} ${dataset}"
    bash "${STAGE}/run_eval_one.sh" "${model}" "${seed}" "${dataset}" "${gpu}"
    echo "GPU ${gpu} DONE ${model} seed${seed} ${dataset}"
  done
}
for gpu in 0 1 2 3 4 5 6 7; do worker "${gpu}" & pids[$gpu]="$!"; done
for pid in "${pids[@]}"; do wait "${pid}"; done
python_module="/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
"${python_module}" -m compose.eval.summarize_controlled \
  --evaluation-root "${STAGE}/${EVAL_VARIANT}" \
  --output-file "${STAGE}/final_summary.json" \
  --failure-file "${STAGE}/failure_cases.json" \
  --marginal-file "${STAGE}/per_sample_marginal_contributions.jsonl" \
  > "${STAGE}/summarize.log" 2>&1
sha256sum "${STAGE}/final_summary.json" "${STAGE}/failure_cases.json" \
  "${STAGE}/per_sample_marginal_contributions.jsonl" \
  > "${STAGE}/evaluation_SHA256SUMS"
date -u +%FT%TZ > "${STAGE}/${EVAL_VARIANT}/matrix_complete_utc.txt"
