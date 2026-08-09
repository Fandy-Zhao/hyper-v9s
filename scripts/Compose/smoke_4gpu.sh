#!/usr/bin/env bash
# One-shot 4-GPU smoke (spec §24) — run exactly once, then start the
# formal seed42 resume (run_ucit_formal_seed42_4gpu.sh). Every phase
# FAILS the whole smoke (nonzero exit) on any violation.
#
#   Phase 1  §9/§21  DDP gradient audit + init-hash identity (smoke_ddp.py)
#   Phase 2  §24     mini task0 pipeline on GPUs 4,5,6,7 (contract, commit,
#                    pool_version, snapshot, RMS, distributed inference)
#   Phase 3  §24     mini task1 pipeline (historical experts through DDP)
#   Phase 4  §15     feature parity: single-GPU vs 4-GPU merged, <=1e-5
#   Phase 5  §17     RMS kappa parity: single-GPU vs 4-GPU torchrun, <=1e-5
#   Phase 6  §19     eval parity: 64 test samples, single vs 4-GPU,
#                    byte-identical answers
#   Phase 7  §25     scaling report: 256 samples, 1 vs 4 GPUs; the
#                    single-GPU reference runs the identical smoke_ddp
#                    path as plain python (torchrun world1 hits a cuDNN
#                    8.8-vs-8.9 engine-selection quirk on this host:
#                    CUDNN_STATUS_NOT_INITIALIZED / "unable to find an
#                    engine"; verified plain python world1 passes the
#                    full forward+backward path)
#   Phase 8  invariants: distributed training contract, registry
#                    pool_version deltas, RMS execution mode, S11 merge,
#                    snapshots
#
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
cd "$REPO"

GPUS="4,5,6,7"
SINGLE_GPU="4"
CONFIG="$REPO/configs/compose_ucit_smoke_4gpu.yaml"
SMOKE_ROOT="$REPO/experiments/runs/compose_ucit_smoke_4gpu"
SCRATCH=/tmp/compose_smoke_4gpu
mkdir -p "$SCRATCH"
rm -rf "$SMOKE_ROOT"   # one-shot smoke; a fresh root is part of the test

BASE=/data/ckpt/zhaozhuofan/models/llava-v1.5-7b
VISION=/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336
PROJECTOR="$BASE/mm_projector.bin"
IMAGES=/data/dataset/zhaozhuofan/UCIT/datasets
DATA="$REPO/data/smoke_4gpu"

PASS=0
FAIL=0
declare -a VERDICTS

verdict() { # verdict <ok|fail> <label>
  if [ "$1" = ok ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
  VERDICTS+=("$1 $2")
  echo "[$1] $2"
}

fail_hard() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
echo "=== Phase 0: preflight (GPUs 4-7 free) ==="
for g in 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -lt 1000 ] || fail_hard "GPU $g busy (${used} MiB used)"
done
echo "GPUs 4-7 idle"

# ---------------------------------------------------------------------------
echo "=== Phase 1: DDP gradient audit (§9/§21) ==="
AUDIT_ROOT="$SCRATCH/audit"
rm -rf "$AUDIT_ROOT"
mkdir -p "$AUDIT_ROOT"
if env CUDA_VISIBLE_DEVICES="$GPUS" $PY -m torch.distributed.run \
    --standalone --max-restarts=0 --nproc_per_node=4 \
    -m compose.experiments.smoke_ddp \
    --model_name_or_path "$BASE" \
    --vision_tower "$VISION" \
    --pretrain_mm_mlp_adapter "$PROJECTOR" \
    --version v1 \
    --data_path "$DATA/audit_train.json" \
    --image_folder "$IMAGES" \
    --compose-mode cluster_expert \
    --compose-selection-manifest "$DATA/audit_manifest.json" \
    --compose-cluster-expert-ids 77 \
    --output_dir "$AUDIT_ROOT" \
    --bf16 True --tf32 True --gradient_checkpointing True \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --num_train_epochs 1 --learning_rate 2.0e-4 --warmup_ratio 0.03 \
    --lr_scheduler_type cosine --logging_steps 1 --save_steps 999999 \
    --model_max_length 2048 --dataloader_num_workers 0 \
    --cache_dir /tmp/compose_hf_cache --report_to none --seed 42 \
    > "$SCRATCH/audit.log" 2>&1; then
  AUDIT_PASSES=$(grep -c '"audit_passed": true' "$SCRATCH/audit.log" || true)
  if [ "$AUDIT_PASSES" = "4" ]; then
    verdict ok "phase1 DDP gradient audit (all 4 ranks PASS, §9/§21)"
  else
    grep -E 'audit_passed|FAILED|Error' "$SCRATCH/audit.log" | tail -20
    verdict fail "phase1 DDP gradient audit ($AUDIT_PASSES/4 ranks passed)"
  fi
else
  tail -60 "$SCRATCH/audit.log" >&2
  verdict fail "phase1 DDP gradient audit (torchrun failed)"
fi

# ---------------------------------------------------------------------------
echo "=== Phase 2: mini task0 pipeline, 4-GPU (§24) ==="
# A pipeline failure is BLOCKING (spec: bug -> STOP): no point running
# later phases against a broken root.
if $PY -m compose.experiments.task_run \
    --config "$CONFIG" --root "$SMOKE_ROOT" \
    --first-task 0 --last-task 0 --gpus "$GPUS" \
    > "$SCRATCH/task0.log" 2>&1; then
  verdict ok "phase2 task0 pipeline completed"
else
  tail -80 "$SCRATCH/task0.log" >&2
  fail_hard "task0 pipeline failed"
fi

# ---------------------------------------------------------------------------
echo "=== Phase 3: mini task1 pipeline, 4-GPU (§24, historical experts) ==="
if $PY -m compose.experiments.task_run \
    --config "$CONFIG" --root "$SMOKE_ROOT" \
    --first-task 1 --last-task 1 --gpus "$GPUS" \
    > "$SCRATCH/task1.log" 2>&1; then
  verdict ok "phase3 task1 pipeline completed"
else
  tail -80 "$SCRATCH/task1.log" >&2
  fail_hard "task1 pipeline failed"
fi

# ---------------------------------------------------------------------------
echo "=== Phase 4: feature parity single vs 4-GPU (§15) ==="
FEAT_A="$SCRATCH/features_single.json"
if CUDA_VISIBLE_DEVICES=$SINGLE_GPU $PY -m compose.eval.query_features \
    --questions "$SMOKE_ROOT/task0/data/teacher_train.json" \
    --images "$IMAGES" \
    --output "$FEAT_A" \
    --seed 42 --device cuda:0 \
    > "$SCRATCH/feat_single.log" 2>&1; then
  if $PY -m scripts.Compose.check_smoke_parity \
      --features-a "$FEAT_A" \
      --features-b "$SMOKE_ROOT/task0/features/train_features.json" \
      > "$SCRATCH/parity_features.log" 2>&1; then
    verdict ok "phase4 feature parity (single vs merged-4, <=1e-5)"
  else
    verdict fail "phase4 feature parity: $(cat "$SCRATCH/parity_features.log")"
  fi
else
  fail_hard "single-GPU query_features failed: $(tail -20 "$SCRATCH/feat_single.log")"
fi

# ---------------------------------------------------------------------------
echo "=== Phase 5: RMS kappa parity single vs 4-GPU (§17) ==="
# Task1's S9 calibrated against the task1-committed pool; re-run it
# single-GPU with identical inputs and compare the calibration.
SNAP_MANIFEST="$SMOKE_ROOT/task1/snapshots/task1/manifest.json"
[ -f "$SNAP_MANIFEST" ] || fail_hard "task1 snapshot manifest missing"
POOL_DIR=$($PY - <<EOF
import json
print(json.load(open("$SNAP_MANIFEST"))["pool_checkpoint_dir"])
EOF
)
[ -n "$POOL_DIR" ] || fail_hard "no pool_checkpoint_dir in task1 snapshot"
[ -f "$POOL_DIR/compose_experts.bin" ] || fail_hard "pool bin missing: $POOL_DIR"
BIN_SHA=$($PY - <<EOF
import hashlib
print(hashlib.sha256(open("$POOL_DIR/compose_experts.bin", "rb").read()).hexdigest())
EOF
)
RMS_SCRATCH="$SCRATCH/rms_single"
rm -rf "$RMS_SCRATCH"; mkdir -p "$RMS_SCRATCH"
# --batch-size 4 replicates the 4-GPU per-rank batch shape (16 samples
# sharded 4x4 -> one batch of 4 per rank). bf16 GEMM rounding depends on
# the matmul shape, so §17 parity requires identical batch shapes, not
# just identical fp64 aggregation (batch-size 8 gave kappa diffs ~1e-3;
# batch-size 4 is bit-identical).
if CUDA_VISIBLE_DEVICES=$SINGLE_GPU $PY -m compose.eval.rms_stats \
    --model-path "$BASE" --vision-tower "$VISION" --projector-path "$PROJECTOR" \
    --checkpoint-dir "$POOL_DIR" \
    --question-file "$SMOKE_ROOT/task1/data/teacher_val.json" \
    --image-folder "$IMAGES" \
    --checkpoint-hash "$BIN_SHA" \
    --batch-size 4 \
    --output-dir "$RMS_SCRATCH" --device cuda:0 \
    > "$SCRATCH/rms_single.log" 2>&1; then
  if $PY -m scripts.Compose.check_smoke_parity \
      --rms-a "$RMS_SCRATCH/rms_calibration.json" \
      --rms-b "$SMOKE_ROOT/task1/rms/rms_calibration.json" \
      > "$SCRATCH/parity_rms.log" 2>&1; then
    verdict ok "phase5 RMS parity (single vs 4-GPU, <=1e-5)"
  else
    verdict fail "phase5 RMS parity: $(cat "$SCRATCH/parity_rms.log")"
  fi
else
  fail_hard "single-GPU rms_stats failed: $(tail -20 "$SCRATCH/rms_single.log")"
fi

# ---------------------------------------------------------------------------
echo "=== Phase 6: eval parity 64 samples, single vs 4-GPU (§19) ==="
# Preserve the pipeline's own 4-GPU S11 answers first (formal_ucit_eval
# --no-reuse-s11 overwrites eval_output/answers.jsonl).
cp "$SMOKE_ROOT/task0/eval_output/answers.jsonl" "$SCRATCH/answers_s11_4gpu.jsonl"
EVAL_SINGLE="$SCRATCH/answers_eval_single.jsonl"
EVAL_FOUR="$SCRATCH/answers_eval_4gpu.jsonl"
# Same snapshot, config, seed, prompts: only the execution strategy differs.
if $PY -m compose.eval.formal_ucit_eval \
    --root "$SMOKE_ROOT" --stage-task 0 --config "$CONFIG" \
    --gpus "$SINGLE_GPU" --no-reuse-s11 \
    > "$SCRATCH/eval_single.log" 2>&1; then
  cp "$SMOKE_ROOT/task0/eval_output/answers.jsonl" "$EVAL_SINGLE"
else
  fail_hard "single-GPU formal eval failed: $(tail -20 "$SCRATCH/eval_single.log")"
fi
if $PY -m compose.eval.formal_ucit_eval \
    --root "$SMOKE_ROOT" --stage-task 0 --config "$CONFIG" \
    --gpus "$GPUS" --no-reuse-s11 \
    > "$SCRATCH/eval_four.log" 2>&1; then
  cp "$SMOKE_ROOT/task0/eval_output/answers.jsonl" "$EVAL_FOUR"
else
  fail_hard "4-GPU formal eval failed: $(tail -20 "$SCRATCH/eval_four.log")"
fi
if $PY -m scripts.Compose.check_smoke_parity \
    --answers-a "$EVAL_SINGLE" --answers-b "$EVAL_FOUR" \
    > "$SCRATCH/parity_eval.log" 2>&1; then
  verdict ok "phase6 eval parity (64 samples byte-identical, §19)"
else
  verdict fail "phase6 eval parity: $(cat "$SCRATCH/parity_eval.log")"
fi

# ---------------------------------------------------------------------------
echo "=== Phase 7: scaling report, 256 samples, 1 vs 4 GPUs (§25) ==="
SCALE_STEPS=8
SCALE_COMMON=(
  --model_name_or_path "$BASE"
  --vision_tower "$VISION"
  --pretrain_mm_mlp_adapter "$PROJECTOR"
  --version v1
  --data_path "$DATA/scaling256.json"
  --image_folder "$IMAGES"
  --compose-mode cluster_expert
  --compose-selection-manifest "$DATA/scaling_manifest.json"
  --compose-cluster-expert-ids 77
  --bf16 True --tf32 True --gradient_checkpointing True
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2
  --num_train_epochs 1 --learning_rate 2.0e-4 --warmup_ratio 0.03
  --lr_scheduler_type cosine --logging_steps 1 --save_steps 999999
  --model_max_length 2048 --dataloader_num_workers 0
  --cache_dir /tmp/compose_hf_cache --report_to none --seed 42
)
if CUDA_VISIBLE_DEVICES=$SINGLE_GPU SMOKE_DDP_STEPS=$SCALE_STEPS \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29599 \
    $PY -m compose.experiments.smoke_ddp \
    --output_dir "$SCRATCH/scale_single" \
    "${SCALE_COMMON[@]}" \
    > "$SCRATCH/scale_single.log" 2>&1; then
  grep '"rank": 0' "$SCRATCH/scale_single.log" > "$SCRATCH/scaling_single.json" || true
else
  fail_hard "single-GPU scaling run failed: $(tail -20 "$SCRATCH/scale_single.log")"
fi
if CUDA_VISIBLE_DEVICES="$GPUS" SMOKE_DDP_STEPS=$SCALE_STEPS \
    $PY -m torch.distributed.run --standalone --max-restarts=0 --nproc_per_node=4 \
    -m compose.experiments.smoke_ddp \
    --output_dir "$SCRATCH/scale_four" \
    "${SCALE_COMMON[@]}" \
    > "$SCRATCH/scale_four.log" 2>&1; then
  grep '"rank": 0' "$SCRATCH/scale_four.log" > "$SCRATCH/scaling_four_rank0.json" || true
else
  fail_hard "4-GPU scaling run failed: $(tail -20 "$SCRATCH/scale_four.log")"
fi
$PY - "$SCRATCH/scaling_single.json" "$SCRATCH/scaling_four_rank0.json" \
    "$SMOKE_ROOT/scaling_report.json" <<'EOF'
import json, sys
single = json.load(open(sys.argv[1]))
four = json.load(open(sys.argv[2]))
report = {
    "samples": 256,
    "steps": single["steps_run"],
    "single_gpu": {
        "mean_step_time_s": single["mean_step_time_s"],
        "wall_time_steps_s": single["wall_time_steps_s"],
        "samples_per_second": single["samples_per_second"],
        "peak_vram_bytes": single["peak_memory_allocated_bytes"],
    },
    "four_gpu": {
        "mean_step_time_s": four["mean_step_time_s"],
        "wall_time_steps_s": four["wall_time_steps_s"],
        "samples_per_second": four["samples_per_second"],
        "peak_vram_bytes": four["peak_memory_allocated_bytes"],
    },
}
report["speedup"] = single["samples_per_second"] / four["samples_per_second"]
report["parallel_efficiency"] = report["speedup"] / 4.0
json.dump(report, open(sys.argv[3], "w"), indent=2)
print(json.dumps(report, indent=2))
EOF
verdict ok "phase7 scaling report ($SMOKE_ROOT/scaling_report.json)"

# ---------------------------------------------------------------------------
echo "=== Phase 8: pipeline-level invariants ==="
if $PY - "$SMOKE_ROOT" > "$SCRATCH/invariants.log" 2>&1 <<'EOF'
import json, os, sys
root = sys.argv[1]

def load(rel):
    with open(os.path.join(root, rel)) as f:
        return json.load(f)

problems = []
# distributed_training_contract.json per task: assertions true,
# global batches equal, per-device batch 1 with grad_accum 2.
for task in (0, 1):
    rel = "task{}/distributed_training_contract.json".format(task)
    if not os.path.exists(os.path.join(root, rel)):
        problems.append("task{}: missing distributed_training_contract.json".format(task))
        continue
    contract = load(rel)
    for key, value in contract["assertions"].items():
        if not value:
            problems.append("task{}: contract assertion {} = {}".format(task, key, value))
    if contract["single_gpu_reference"]["global_batch"] != contract["four_gpu"]["global_batch"]:
        problems.append("task{}: global batch mismatch".format(task))
    if contract["four_gpu"]["gradient_accumulation_steps"] != 2:
        problems.append("task{}: grad_accum != 2".format(task))
# registry: pool_version == total committed experts, per-task deltas exact.
reg0 = load("task0/state/expert_registry.json")
reg1 = load("task1/state/expert_registry.json")
n0 = len(reg0.get("experts", {}))
n1 = len(reg1.get("experts", {}))
if n0 < 1:
    problems.append("task0: no experts committed")
if n1 <= n0:
    problems.append("task1: expert count {} not > task0 {}".format(n1, n0))
if reg0.get("pool_version") != n0:
    problems.append("task0: pool_version {} != expert count {}".format(reg0.get("pool_version"), n0))
if reg1.get("pool_version") != n1:
    problems.append("task1: pool_version {} != expert count {}".format(reg1.get("pool_version"), n1))
# RMS executed distributed.
summary = load("task1/rms/rms_summary.json")
if summary.get("execution", {}).get("mode") != "4gpu_torchrun":
    problems.append("rms: execution.mode != 4gpu_torchrun")
# S11 answers merged with zero missing/duplicates.
summary = load("task0/eval_output/run_summary.json")
merge = summary.get("execution", {}).get("merge_verified", {})
if merge.get("missing", -1) != 0 or merge.get("duplicates", -1) != 0:
    problems.append("eval: merge_verified = {}".format(merge))
n_answers = sum(1 for _ in open(os.path.join(root, "task0/eval_output/answers.jsonl")))
if n_answers != 64:
    problems.append("eval: task0 answers = {} (expected 64)".format(n_answers))
# snapshots
for task in (0, 1):
    if not os.path.isdir(os.path.join(root, "task{}/snapshots/task{}".format(task, task))):
        problems.append("task{}: snapshot missing".format(task))
for problem in problems:
    print("PROBLEM:", problem)
sys.exit(1 if problems else 0)
EOF
then
  verdict ok "phase8 pipeline invariants (contract/registry/RMS/eval/snapshots)"
else
  cat "$SCRATCH/invariants.log"
  verdict fail "phase8 pipeline invariants"
fi

# ---------------------------------------------------------------------------
echo
echo "===== SMOKE SUMMARY: $PASS passed, $FAIL failed ====="
for v in "${VERDICTS[@]}"; do echo "  $v"; done
[ "$FAIL" -eq 0 ] || exit 1
echo "ALL SMOKE PHASES PASSED — ready for formal seed42 resume"
