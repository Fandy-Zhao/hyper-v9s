#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/zhaozhuofan/Hyper-LlaVA"
STAGE="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2"
PRIMARY="evaluation_v2"
RETRY="evaluation_retry_v1"
MERGED="evaluation_merged"
RUNNER="${STAGE}/run_eval_one.sh"
read -r -a gpu_ids <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
[[ "${#gpu_ids[@]}" -gt 0 ]] || { echo "GPU_IDS must name at least one GPU" >&2; exit 2; }

is_complete() {
  local root="$1" model="$2" seed="$3" dataset="$4"
  local dir="${STAGE}/${root}/seed${seed}/${dataset}/${model}"
  [[ -s "${dir}/summary.json" && -s "${dir}/per_sample.jsonl" \
    && -f "${dir}/status.txt" ]] && grep -qx 'exit_code=0' "${dir}/status.txt"
}

jobs=()
while read -r model seed dataset; do
  if ! is_complete "${PRIMARY}" "${model}" "${seed}" "${dataset}"; then
    jobs+=("${model} ${seed} ${dataset}")
  fi
done < "${STAGE}/${PRIMARY}_jobs.txt"
printf '%s\n' "${jobs[@]}" > "${STAGE}/${RETRY}_jobs.txt"

worker() {
  local gpu="$1" slot="$2" index model seed dataset
  for ((index=slot; index<${#jobs[@]}; index+=${#gpu_ids[@]})); do
    read -r model seed dataset <<< "${jobs[$index]}"
    echo "GPU ${gpu} RETRY ${model} seed${seed} ${dataset}"
    EVAL_VARIANT="${RETRY}" BATCH_SIZE=4 bash "${RUNNER}" \
      "${model}" "${seed}" "${dataset}" "${gpu}"
  done
}
for slot in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$slot]}"
  worker "${gpu}" "${slot}" & pids[$slot]="$!"
done
for pid in "${pids[@]}"; do wait "${pid}"; done

[[ ! -e "${STAGE}/${MERGED}" ]] || {
  echo "Refusing existing merged evaluation root" >&2
  exit 1
}
while read -r model seed dataset; do
  source_root="${PRIMARY}"
  if is_complete "${RETRY}" "${model}" "${seed}" "${dataset}"; then
    source_root="${RETRY}"
  elif ! is_complete "${PRIMARY}" "${model}" "${seed}" "${dataset}"; then
    echo "No successful result for ${model} seed${seed} ${dataset}" >&2
    exit 1
  fi
  target_parent="${STAGE}/${MERGED}/seed${seed}/${dataset}"
  mkdir -p "${target_parent}"
  ln -s "${STAGE}/${source_root}/seed${seed}/${dataset}/${model}" \
    "${target_parent}/${model}"
done < "${STAGE}/${PRIMARY}_jobs.txt"

python_module="/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
"${python_module}" -m compose.eval.summarize_controlled \
  --evaluation-root "${STAGE}/${MERGED}" \
  --output-file "${STAGE}/final_summary.json" \
  --failure-file "${STAGE}/failure_cases.json" \
  --marginal-file "${STAGE}/per_sample_marginal_contributions.jsonl" \
  > "${STAGE}/summarize.log" 2>&1
sha256sum "${STAGE}/final_summary.json" "${STAGE}/failure_cases.json" \
  "${STAGE}/per_sample_marginal_contributions.jsonl" \
  > "${STAGE}/evaluation_SHA256SUMS"
date -u +%FT%TZ > "${STAGE}/${MERGED}/matrix_complete_utc.txt"
