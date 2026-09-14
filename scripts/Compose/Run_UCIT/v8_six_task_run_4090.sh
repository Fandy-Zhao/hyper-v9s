#!/usr/bin/env bash
# The only formal V8 launcher.  V7 names below are configuration/kernel names.
set -euo pipefail
REPO="${REPO:-/home/xukunlun/CODE/CIT/zzf}"
PY="${PY:-/home/xukunlun/anaconda3/envs/hyper/bin/python}"
FORMAL_CONFIG="${FORMAL_CONFIG:-${REPO}/configs/target_4090/v7_ucit_formal_4090.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-${REPO}/configs/target_4090/v7_global_coevolution_4090.yaml}"
RUN_ROOT="${RUN_ROOT:-/data/ckpt/xukunlun/CIT/zzf_runs/v8_formal}"
V8_CONFIG="${V8_CONFIG:-${REPO}/configs/v8_exact_accelerated.yaml}"
V8_QUERY_CACHE_ROOT="${V8_QUERY_CACHE_ROOT:-${RUN_ROOT}/query_cache}"
QUERY_CACHE_MANIFEST="${QUERY_CACHE_MANIFEST:-${V8_QUERY_CACHE_ROOT}/query_cache_manifest.json}"
GPUS="${1:-${GPUS:-}}"
TASKS="${TASKS:-0,1,2,3,4,5}"
EVAL_GPUS="${EVAL_GPUS:-${GPUS}}"
TEACHER_NUM_SAMPLES="${TEACHER_NUM_SAMPLES:-256}"
TEACHER_SAMPLE_RATIO="${TEACHER_SAMPLE_RATIO:-}"
TEACHER_SEED="${TEACHER_SEED:-42}"
MIN_TEACHER_SUPPORT="${MIN_TEACHER_SUPPORT:-4}"
MIN_TEACHER_USAGE_RATE="${MIN_TEACHER_USAGE_RATE:-0.02}"
RECIPE_MODE="${RECIPE_MODE:-strict}"
TRAINING_WORKERS="${TRAINING_WORKERS:-4}"
[[ -z "${TEACHER_NUM_SAMPLES}" || -z "${TEACHER_SAMPLE_RATIO}" ]] || { echo 'Teacher size and ratio are mutually exclusive' >&2; exit 2; }
cd "${REPO}"
"${PY}" -m compose.experiments.v8_task_run --check-teacher-support --tasks "${TASKS}"
mapfile -t rows < <("${PY}" - "${FORMAL_CONFIG}" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1])); d=c['data']
print('\t'.join([c['method_config'],d['model_path'],d['vision_tower'],d['projector_path'],d['image_folder']]))
for t in c['tasks']:
 print('\t'.join(str(t[k]) for k in ('task_index','name','train_file','validation_file','test_file','validation_metric','validation_annotation_file')))
PY
)
IFS=$'\t' read -r cfg model vision projector images <<<"${rows[0]}"
previous=""
for row in "${rows[@]:1}"; do
 IFS=$'\t' read -r task name train val test metric annotation <<<"${row}"
 [[ ",${TASKS}," == *",${task},"* ]] || continue
 args=("${PY}" -m compose.experiments.v8_task_run --config "${METHOD_CONFIG}" --root "${RUN_ROOT}/task${task}" --task-index "${task}" --task-name "${name}" --train-file "${train}" --val-file "${val}" --test-file "${test}" --validation-metric "${metric}" --validation-annotation-file "${annotation}" --model-path "${model}" --vision-tower "${vision}" --projector-path "${projector}" --image-folder "${images}" --query-cache-manifest "${QUERY_CACHE_MANIFEST}" --compose-v8-config "${V8_CONFIG}" --compose-v8-query-cache-root "${V8_QUERY_CACHE_ROOT}" --v8-teacher-seed "${TEACHER_SEED}" --v8-min-teacher-support "${MIN_TEACHER_SUPPORT}" --v8-min-teacher-usage-rate "${MIN_TEACHER_USAGE_RATE}" --recipe-mode "${RECIPE_MODE}" --training-dataloader-num-workers "${TRAINING_WORKERS}" --skip-eval)
 [[ "${task}" == 0 ]] || args+=( ${TEACHER_NUM_SAMPLES:+--v8-teacher-num-samples "${TEACHER_NUM_SAMPLES}"} ${TEACHER_SAMPLE_RATIO:+--v8-teacher-sample-ratio "${TEACHER_SAMPLE_RATIO}"} )
 [[ -z "${GPUS}" ]] || args+=(--gpus "${GPUS}")
 [[ -z "${previous}" ]] || args+=(--previous-checkpoint "${previous}")
 [[ ! -d "${RUN_ROOT}/task${task}" ]] || args+=(--resume)
 echo "V8 Task${task} ${name}"; "${args[@]}"; previous="${RUN_ROOT}/task${task}/committed"
done
