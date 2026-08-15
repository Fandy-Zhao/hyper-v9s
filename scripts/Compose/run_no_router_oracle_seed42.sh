#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
FORMAL="$REPO/experiments/runs/compose_ucit_v62_formal_seed42"
OUTPUT="$REPO/experiments/runs/compose_v62_seed42_no_router_oracle"
GPUS=4,5,6,7
export PATH=/home/zhaozhuofan/miniconda3/envs/hyper/lib/jvm/bin:/home/zhaozhuofan/miniconda3/envs/hyper/bin:"$PATH"

cd "$REPO"

if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ (^|,)(0|1|2|3)(,|$) ]]; then
  echo "Refusing to run with physical GPU 0-3 visible" >&2
  exit 2
fi

"$PY" -m compose.eval.no_router_oracle prepare \
  --formal-root "$FORMAL" --output-root "$OUTPUT"

# Phase A/B: finish all validation/test fixed combinations before starting
# the target-answer sample oracle. Every worker resumes from JSONL prefixes.
for task in 0 1 2 3 4 5; do
  "$PY" -m compose.eval.no_router_oracle run-fixed \
    --formal-root "$FORMAL" --output-root "$OUTPUT" \
    --task "$task" --split val --gpus "$GPUS"
  "$PY" -m compose.eval.no_router_oracle run-fixed \
    --formal-root "$FORMAL" --output-root "$OUTPUT" \
    --task "$task" --split test --gpus "$GPUS"
done

# Phase C: target-answer teacher-forcing selection plus one official
# generation per test sample using the selected 0/1/2-expert set.
for task in 0 1 2 3 4 5; do
  "$PY" -m compose.eval.no_router_oracle run-sample \
    --formal-root "$FORMAL" --output-root "$OUTPUT" \
    --task "$task" --gpus "$GPUS" --batch-size 8
done

# Phase D is derived from exact stage/final tensor+kappa equivalence. The
# aggregator refuses reuse if any then-available state differs.
"$PY" -m compose.eval.no_router_report \
  --formal-root "$FORMAL" --output-root "$OUTPUT"
