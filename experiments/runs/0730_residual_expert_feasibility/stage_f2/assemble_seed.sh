#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 1 ]] || { echo "Usage: $0 SEED" >&2; exit 2; }
seed="$1"
case "${seed}" in 42|43|44) ;; *) echo "Unsupported seed: ${seed}" >&2; exit 2 ;; esac

ROOT="/home/zhaozhuofan/Hyper-LlaVA"
SOURCE_ROOT="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/seed${seed}"
OUTPUT_ROOT="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/assembled/seed${seed}"
RECORD_ROOT="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/assembly/seed${seed}"
mkdir -p "${RECORD_ROOT}"

assemble() {
  local name="$1"
  shift
  local output="${OUTPUT_ROOT}/${name}"
  if [[ -s "${output}/compose_experts.bin" && -s "${output}/compose_experts.json" ]]; then
    echo "SKIP complete ${output}"
    return
  fi
  /home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m compose.experts.assemble \
    "$@" --output-dir "${output}" > "${RECORD_ROOT}/${name}.log" 2>&1
  sha256sum "${output}/compose_experts.bin" "${output}/compose_experts.json" \
    > "${RECORD_ROOT}/${name}_SHA256SUMS"
}

cd "${ROOT}"
assemble a_independent_b \
  --source "${SOURCE_ROOT}/expert_a" 0 \
  --source "${SOURCE_ROOT}/independent_b" 1
assemble independent_b_c \
  --source "${SOURCE_ROOT}/independent_b" 1 \
  --source "${SOURCE_ROOT}/expert_c" 2
assemble residual_b_c \
  --source "${SOURCE_ROOT}/residual_b" 1 \
  --source "${SOURCE_ROOT}/expert_c" 2
date -u +%FT%TZ > "${RECORD_ROOT}/complete_utc.txt"
