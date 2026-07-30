#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhaozhuofan/Hyper-LlaVA"
PYTHON="/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
RUN_DIR="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2"
DATA_ROOT="/data/dataset/zhaozhuofan/Hyper-LLaVA/controlled_functional_v1"
mkdir -p "${RUN_DIR}"
if [[ -e "${DATA_ROOT}" ]]; then
  echo "Refusing existing controlled dataset root: ${DATA_ROOT}" >&2
  exit 1
fi
args=(-m compose.data.controlled_functional --output-root "${DATA_ROOT}" --seed 730 --train-size 1600 --val-size 200 --test-size 400)
{
  printf '%q' "${PYTHON}"
  printf ' %q' "${args[@]}"
  printf '\n'
} > "${RUN_DIR}/generate_command.txt"
set +e
"${PYTHON}" "${args[@]}" > "${RUN_DIR}/generate.log" 2>&1
status="$?"
set -e
printf 'exit_code=%s\nfinished_utc=%s\n' "${status}" "$(date -u +%FT%TZ)" > "${RUN_DIR}/generate.status"
if [[ "${status}" -eq 0 ]]; then
  cp "${DATA_ROOT}/manifest.json" "${RUN_DIR}/dataset_manifest.json"
  sha256sum "${DATA_ROOT}/manifest.json" > "${RUN_DIR}/dataset_manifest.sha256"
fi
exit "${status}"
