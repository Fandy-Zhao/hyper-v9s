#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhaozhuofan/Hyper-LlaVA"
RUNNER="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/run_train_one.sh"
RUN_ROOT="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/training"

is_complete() {
  local model_name="$1" seed="$2" status checkpoint_root
  status="${RUN_ROOT}/seed${seed}/${model_name}/status.txt"
  checkpoint_root="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/seed${seed}/${model_name}"
  [[ -f "${status}" ]] && grep -qx 'exit_code=0' "${status}" \
    && [[ -s "${checkpoint_root}/compose_experts.bin" ]] \
    && [[ -s "${checkpoint_root}/compose_experts.json" ]]
}

wait_for_existing() {
  local model_name="$1" seed="$2"
  until [[ -f "${RUN_ROOT}/seed${seed}/${model_name}/status.txt" ]]; do
    sleep 10
  done
  is_complete "${model_name}" "${seed}" || {
    echo "Existing run failed validation: seed=${seed} model=${model_name}" >&2
    exit 1
  }
}

run_one() {
  local model_name="$1" seed="$2" gpu_ids="$3" port="$4"
  if is_complete "${model_name}" "${seed}"; then
    echo "SKIP complete seed=${seed} model=${model_name}"
    return
  fi
  echo "START seed=${seed} model=${model_name} gpu=${gpu_ids} port=${port}"
  bash "${RUNNER}" "${model_name}" "${seed}" "${gpu_ids}" "${port}"
  is_complete "${model_name}" "${seed}"
  echo "DONE seed=${seed} model=${model_name}"
}

queue_left() {
  wait_for_existing expert_a 42
  run_one expert_c 42 0,1,2,3 29721
  run_one task_ab 42 0,1,2,3 29722
  run_one residual_b 42 0,1,2,3 29723
  for seed in 43 44; do
    run_one expert_a "${seed}" 0,1,2,3 "$((29800 + seed * 10 + 1))"
    run_one expert_c "${seed}" 0,1,2,3 "$((29800 + seed * 10 + 2))"
    run_one task_ab "${seed}" 0,1,2,3 "$((29800 + seed * 10 + 3))"
    run_one residual_b "${seed}" 0,1,2,3 "$((29800 + seed * 10 + 4))"
  done
}

queue_right() {
  wait_for_existing independent_b 42
  run_one rank16_ab 42 4,5,6,7 29731
  run_one upper_bc 42 4,5,6,7 29732
  for seed in 43 44; do
    run_one independent_b "${seed}" 4,5,6,7 "$((29900 + seed * 10 + 1))"
    run_one rank16_ab "${seed}" 4,5,6,7 "$((29900 + seed * 10 + 2))"
    run_one upper_bc "${seed}" 4,5,6,7 "$((29900 + seed * 10 + 3))"
  done
}

queue_left &
left_pid="$!"
queue_right &
right_pid="$!"
wait "${left_pid}"
wait "${right_pid}"
date -u +%FT%TZ > "${RUN_ROOT}/matrix_complete_utc.txt"
