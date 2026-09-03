#!/usr/bin/env bash
# Formal V7 GPU0/1 cached-query run (0903 spec §29-35, §42, §47 Phase C).
#
# The six tasks run sequentially on physical GPU0+GPU1 ONLY.  Every V7 query
# (S1 train/val payloads, S2/S3/S4/S5 downstream consumption, and the final
# 21-cell evaluation) is served from the precomputed fixed-query cache:
#   --query-cache-manifest  per task (S1 emits payloads, encoder_calls=0)
#   v7_formal_ucit_eval --cache-manifest  for the 21 cells (selection
#       manifests over cache test rows, no CLIP query encoder)
# The V7 method, recipe, LR/loss/seed/epochs are untouched; the strict recipe
# is world 2 x per-device batch 1 x accumulation 32 = global batch exactly 64.
#
# The launcher never starts while another user occupies GPU0/1 (availability
# gate, fail closed -> wait/report blockers; nothing is ever stolen).
#
# Usage:
#     ./v7_gpu01_cached_query_formal.sh
#   env overrides: REPO PY RUN_ROOT CACHE_MANIFEST RECIPE_MODE TRAINING_WORKERS

set -euo pipefail

REPO="${REPO:-/home/zhaozhuofan/Hyper-LlaVA}"
PY="${PY:-/home/zhaozhuofan/miniconda3/envs/hyper/bin/python}"
FORMAL_CONFIG="${FORMAL_CONFIG:-${REPO}/configs/v7_ucit_formal.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-${REPO}/configs/v7_global_coevolution.yaml}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO}/artifacts/v7_query_cache/query_cache_manifest.json}"
RUN_ROOT="${RUN_ROOT:-/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_gpu01_cached_query_formal_20260903}"
RECIPE_MODE="${RECIPE_MODE:-strict}"            # strict is the only formal mode
TRAINING_WORKERS="${TRAINING_WORKERS:-4}"
GPUS="0,1"

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PATH="$(dirname "${PY}"):${PATH}"
cd "${REPO}"

banner() {
  echo
  echo "================================================================"
  echo "  V7 GPU0/1 CACHED QUERY FORMAL RUN"
  echo "================================================================"
  echo "  git HEAD      : $(git -C "${REPO}" rev-parse HEAD)"
  echo "  branch        : $(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
  echo "  run root      : ${RUN_ROOT}"
  echo "  cache manifest: ${CACHE_MANIFEST}"
  echo "  GPUs          : ${GPUS} (physical only; 2-7 are off-limits)"
  echo
}
banner

# ---------------------------------------------------------------------------
# Preflight: FORMAL_START_SHA (0903 spec §42), manifest kind, config files.
# ---------------------------------------------------------------------------
[[ -f "${CACHE_MANIFEST}" ]] || { echo "cache manifest missing: ${CACHE_MANIFEST}" >&2; exit 2; }
for required in "${FORMAL_CONFIG}" "${METHOD_CONFIG}"; do
  [[ -f "${required}" ]] || { echo "missing config: ${required}" >&2; exit 2; }
done

FORMAL_START_SHA="$(git -C "${REPO}" rev-parse HEAD)"
mkdir -p "${RUN_ROOT}"
printf '%s\n' "${FORMAL_START_SHA}" > "${RUN_ROOT}/formal_start_sha.txt"
echo "FORMAL_START_SHA: ${FORMAL_START_SHA}"

# ---------------------------------------------------------------------------
# GPU availability gate: GPU0 and GPU1 must both be idle (never stolen).
# ---------------------------------------------------------------------------
"${PY}" - "${GPUS}" <<'PY'
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
    raise SystemExit(
        "GPU availability gate failed; refusing to preempt other users' jobs.\n"
        "busy={} missing={}\n"
        "Wait for GPU0/1 to go idle, then relaunch (the run resumes from markers).".format(busy, missing)
    )
print("GPU availability gate passed: GPU0/GPU1 idle -> {}".format(state))
PY

# ---------------------------------------------------------------------------
# Query-cache runtime contract precheck (CPU; content binding, fail closed).
# ---------------------------------------------------------------------------
CLIP_PATH="$("${PY}" - "${METHOD_CONFIG}" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["query"]["path"])
PY
)"
"${PY}" - "${CACHE_MANIFEST}" "${METHOD_CONFIG}" <<'PY'
import json, sys, yaml
from compose.v7 import query_cache as qc
from compose.v7.query import FixedQueryProvenance
from compose.eval.query_features import query_backbone_provenance

manifest_path = sys.argv[1]
method = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
clip_path = method["query"]["path"]
manifest = qc.V7CacheManifest.locate(manifest_path)
manifest.validate_runtime_contract(
    required=("train", "val", "test"),
    backbone_hash=query_backbone_provenance(clip_path)["backbone_hash"],
    impl_hash=FixedQueryProvenance().module_hash,
)
print("QUERY_CACHE_READY: YES")
print("QUERY_CONTRACT_MATCH: YES (train/val/test splits, backbone+impl content bound)")
print("QUERY_CACHE_MANIFEST_SHA256: {}".format(manifest.manifest_sha256()))
PY

# ---------------------------------------------------------------------------
# Strict-recipe plan (world 2 x batch 1 x GA 32 = global batch 64).
# ---------------------------------------------------------------------------
"${PY}" - <<'PY'
import sys
from compose.v7.gpu_plan import V7GPUPlan

plan = V7GPUPlan.build([0, 1], recipe_mode="strict")
print(plan.render_plan_block())
recipe = plan.recipe
assert recipe.per_device_batch == 1, recipe
assert recipe.training_world_size == 2, recipe
assert recipe.gradient_accumulation_steps == 32, recipe
assert recipe.effective_global_batch == 64, recipe
print("RECIPE_EXACT: YES (per-device batch 1, world 2, GA 32, global batch exactly 64)")
PY

# ---------------------------------------------------------------------------
# Formal task table (unchanged from the legacy launchers: full data, no
# slicing; validation files and test_3000 files are the audited ones).
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
echo "After the six tasks: cached 21-cell evaluation on GPU0/1"

# ---------------------------------------------------------------------------
# Run the tasks: cache-mode S1, strict two-GPU recipe, full data, no eval in
# the task loop (the 21-cell evaluator generates every cell afterwards).
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
    --gpus "${GPUS}" --recipe-mode "${RECIPE_MODE}"
    --query-cache-manifest "${CACHE_MANIFEST}"
    --training-dataloader-num-workers "${TRAINING_WORKERS}"
    --skip-eval
  )
  [[ -z "${previous}" ]] || command+=(--previous-checkpoint "${previous}")
  [[ ! -d "${task_root}" || -z "$(find "${task_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || command+=(--resume)
  echo "Task${task_index} ${task_name} -> ${task_root}"
  "${command[@]}"
  previous="${task_root}/committed"
  echo "Task${task_index} complete: $(git -C "${REPO}" rev-parse HEAD)"
done

# ---------------------------------------------------------------------------
# Final evaluation: 21 lower-triangle cells from cache rows on GPU0/1.
# ---------------------------------------------------------------------------
"${PY}" -m compose.eval.v7_formal_ucit_eval \
  --root "${RUN_ROOT}" --formal-config "${FORMAL_CONFIG}" \
  --method-config "${METHOD_CONFIG}" --python "${PY}" \
  --gpus "${GPUS}" \
  --cache-manifest "${CACHE_MANIFEST}" \
  --backbone-path "${CLIP_PATH}"

echo
echo "V7 GPU0/1 CACHED QUERY FORMAL RUN: six tasks + 21-cell evaluation complete"
