#!/usr/bin/env bash
# Formal V7, GPU-count-adaptive (spec 0903): sequential Task0..Task5 with one
# deterministic GPU plan for every stage.
#
# GPU entry is unified -- declare once, in one of:
#     ./v7_six_task_run_auto.sh 0,1,2,3     positional argument (CLI)
#     GPUS=0,1,2,3 ./v7_six_task_run_auto.sh
#     V7_GPUS=0,1,2,3 ./v7_six_task_run_auto.sh
#     CUDA_VISIBLE_DEVICES=0,1,2,3 ...
#     (none of the above) -> the orchestrator probes and uses ONLY idle GPUs
#
# The GPU count only changes scheduling.  strict (default) keeps the formal
# global batch exactly 64: world size is the largest recipe-exact divisor of
# the declared count, gradient accumulation = 64 / world size; extra GPUs
# stay idle during S3.  throughput is an explicit smoke/benchmark opt-in and
# is recorded RECIPE_EQUIVALENT=NO.
#
# The launcher never touches a GPU another user is using: explicit GPU lists
# go through an idle availability gate, and automatic detection only ever
# selects devices that are already idle (it never kills or resets anything).
set -euo pipefail

REPO="${REPO:-/home/zhaozhuofan/Hyper-LlaVA}"
PY="${PY:-/home/zhaozhuofan/miniconda3/envs/hyper/bin/python}"
FORMAL_CONFIG="${FORMAL_CONFIG:-${REPO}/configs/v7_ucit_formal.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-${REPO}/configs/v7_global_coevolution.yaml}"
RUN_ROOT="${RUN_ROOT:-/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_ucit_formal_auto_seed42}"
RECIPE_MODE="${RECIPE_MODE:-strict}"            # strict (formal) | throughput (smoke opt-in)
TRAINING_WORKERS="${TRAINING_WORKERS:-4}"       # dataloader workers; recipe math is untouched
GPU_ARG="${1:-${GPUS:-}}"                        # positional > $GPUS > (runner: V7_GPUS/CUDA_VISIBLE_DEVICES/probe)

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PATH="$(dirname "${PY}"):${PATH}"
cd "${REPO}"

# ---------------------------------------------------------------------------
# Resolve the execution plan once and print it before running anything.
# ---------------------------------------------------------------------------
plan_text="$(
  "${PY}" - "${GPU_ARG:-__AUTO__}" "${RECIPE_MODE}" <<'PY'
import json, subprocess, sys
from compose.v7.gpu_plan import V7GPUPlan, resolve_available_gpu_ids

declared = None if sys.argv[1] == "__AUTO__" else sys.argv[1]
try:
    available = resolve_available_gpu_ids(cli_ids=declared)
    source = "declared" if declared else "env/probe"
except Exception as error:  # fail closed with a readable message
    print("GPU plan error: {}".format(error), file=sys.stderr)
    raise SystemExit(2)
plan = V7GPUPlan.build(available, recipe_mode=sys.argv[2])
if plan.recipe_exact:
    print("GPU source: {} ({} devices, all idle-gated)".format(source, len(available)))
else:
    print("GPU source: {} ({} devices)".format(source, len(available)))
print(plan.render_plan_block())
print("RECIPE_MODE: {}".format(plan.recipe.recipe_mode))
print("RECIPE_EQUIVALENT: {}".format("YES" if plan.recipe_exact else "NO"))
PY
)"
echo "${plan_text}"

# ---------------------------------------------------------------------------
# Explicit GPU lists must not be busy (nothing is ever stolen).  Automatic
# detection inside the orchestrator only selects already-idle devices, so it
# needs no separate gate.
# ---------------------------------------------------------------------------
if [[ -n "${GPU_ARG}" ]]; then
  requested="${GPU_ARG}"
  "${PY}" - "${requested}" <<'PY'
import subprocess, sys
requested = [int(value) for value in sys.argv[1].split(",")]
rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
state = {}
for row in rows.splitlines():
    index, memory, utilization = [int(value.strip()) for value in row.split(",")]
    state[index] = (memory, utilization)
missing = [gpu for gpu in requested if gpu not in state]
busy = {gpu: state[gpu] for gpu in requested if gpu in state and (state[gpu][0] >= 1024 or state[gpu][1] != 0)}
if missing or busy:
    raise SystemExit("adaptive GPU availability gate failed: missing={} busy={}".format(missing, busy))
print("adaptive GPU availability gate passed: {}".format({gpu: state[gpu] for gpu in requested}))
PY
fi

# ---------------------------------------------------------------------------
# Formal task table (unchanged from the legacy launchers).
# ---------------------------------------------------------------------------
mapfile -t rows < <("${PY}" - "${FORMAL_CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
d = cfg["data"]
print("META\t" + "\t".join([cfg["method_config"], d["model_path"], d["vision_tower"], d["projector_path"], d["image_folder"]]))
for task in cfg["tasks"]:
    print("TASK\t" + "\t".join(str(task[key]) for key in (
        "task_index", "name", "train_file", "validation_file", "test_file",
        "validation_metric", "validation_annotation_file"
    )))
PY
)

IFS=$'\t' read -r kind method_config model_path vision_tower projector_path image_folder <<<"${rows[0]}"
for required in "${method_config}" "${model_path}" "${vision_tower}" "${projector_path}" "${image_folder}"; do
  [[ -e "${required}" ]] || { echo "Missing required formal path: ${required}" >&2; exit 1; }
done

echo "Run root: ${RUN_ROOT}"
echo "Execution plan:"
previous=""
for row in "${rows[@]:1}"; do
  IFS=$'\t' read -r kind task_index task_name train_file val_file test_file validation_metric validation_annotation <<<"${row}"
  echo "  Task${task_index} ${task_name}: ${RUN_ROOT}/task${task_index}"
done
echo "After the six tasks: ${PY} -m compose.eval.v7_formal_ucit_eval --root ${RUN_ROOT}"

# ---------------------------------------------------------------------------
# Run the tasks.  Only --gpus (plus the dataloader-worker knob) is passed:
# the runner derives training world size / gradient accumulation from the
# strict recipe of the declared GPU count.
# ---------------------------------------------------------------------------
previous=""
for row in "${rows[@]:1}"; do
  IFS=$'\t' read -r kind task_index task_name train_file val_file test_file validation_metric validation_annotation <<<"${row}"
  task_root="${RUN_ROOT}/task${task_index}"
  command=(
    "${PY}" -m compose.experiments.v7_task_run
    --config "${method_config}" --root "${task_root}" --task-index "${task_index}"
    --task-name "${task_name}" --train-file "${train_file}" --val-file "${val_file}"
    --test-file "${test_file}" --validation-metric "${validation_metric}"
    --validation-annotation-file "${validation_annotation}"
    --model-path "${model_path}" --vision-tower "${vision_tower}"
    --projector-path "${projector_path}" --image-folder "${image_folder}"
    --recipe-mode "${RECIPE_MODE}"
    --training-dataloader-num-workers "${TRAINING_WORKERS}"
    --skip-eval
  )
  # Unified GPU entry: only when a GPU list was actually declared; otherwise
  # the orchestrator resolves V7_GPUS / CUDA_VISIBLE_DEVICES / idle probe.
  if [[ -n "${GPU_ARG}" ]]; then
    command+=(--gpus "${GPU_ARG}")
  fi
  [[ -z "${previous}" ]] || command+=(--previous-checkpoint "${previous}")
  [[ ! -d "${task_root}" || -z "$(find "${task_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || command+=(--resume)
  echo "Task${task_index} ${task_name} -> ${task_root}"
  "${command[@]}"
  previous="${task_root}/committed"
done

eval_command=("${PY}" -m compose.eval.v7_formal_ucit_eval \
  --root "${RUN_ROOT}" --formal-config "${FORMAL_CONFIG}" \
  --method-config "${METHOD_CONFIG}" --python "${PY}")
if [[ -n "${GPU_ARG}" ]]; then
  eval_command+=(--gpus "${GPU_ARG}")
fi
"${eval_command[@]}"
"${PY}" -m compose.eval.formal_ucit_summary --root "${RUN_ROOT}"
