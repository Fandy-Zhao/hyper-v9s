#!/usr/bin/env bash
# Formal UCIT six-task seed42 pilot (spec §23).
# Main continual training on physical GPU 4 (logical cuda:0 under
# CUDA_VISIBLE_DEVICES=4,5,6,7); the runner itself is single-GPU serial
# (registry mutation only on the main process); GPUs 5-7 are used by the
# per-stage formal evaluator for sharded historical-task generation.
set -euo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hyper

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONHASHSEED=42

ROOT=experiments/runs/compose_ucit_formal_seed42

# Main continual training: Task0 -> Task5 strictly sequential on the first
# device (physical GPU 4 -> logical cuda:0). `--gpus 4` makes every
# subprocess run with CUDA_VISIBLE_DEVICES=4, cuda:0 = physical GPU4.
python -m compose.experiments.task_run \
  --config configs/compose_ucit.yaml \
  --root "$ROOT" \
  --first-task 0 \
  --last-task 5 \
  --gpus 4

# After each task, score every learned task j <= t against the task-t
# snapshot with the ORIGINAL Hyper-LLaVA scorers; historical generations are
# sharded across physical GPUs 4-7 (same snapshot, deterministic protocol).
for t in 0 1 2 3 4 5; do
  if [ -f "$ROOT/task$t/stages/s12_report.done" ]; then
    echo "== continual evaluation for stage $t =="
    python -m compose.eval.formal_ucit_eval \
      --root "$ROOT" \
      --stage-task "$t" \
      --config configs/compose_ucit.yaml \
      --gpus 4,5,6,7
  else
    echo "stage $t not complete; skipping its continual evaluation" >&2
  fi
done

# Final aggregation: matrix, original-script metrics, self-consistency,
# final table, method diagnostics.
python -m compose.eval.formal_ucit_summary --root "$ROOT"

echo "formal seed42 run complete"
