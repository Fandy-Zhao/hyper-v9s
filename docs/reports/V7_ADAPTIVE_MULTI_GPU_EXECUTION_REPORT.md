# V7 GPU-Count-Adaptive Execution — Implementation & Verification Report

Branch `feat/0903-v7-throughput-equivalence` · 2026-09-03

## Verdicts

| Flag | Verdict |
|---|---|
| ADAPTIVE_GPU_EXECUTION | **PASS** |
| V7_METHOD_UNCHANGED | **YES** |
| STRICT_RECIPE_EQUIVALENT | **YES** (1→1×64, 2→2×32, 3→2×32, 4→4×16, 5/6/7→4×16, 8→8×8; per-device batch stays 1) |
| FULL_DATA_COVERAGE | **YES** (no sample reduction, no `teacher[:2000]`, no `max_samples`, no NLL proxies) |
| SINGLE_GPU_SUPPORTED | **YES** (adaptive path with one declared/auto GPU takes the world-1 recipe) |
| MULTI_GPU_QUERY_SHARDING | **YES** (deterministic merge == single-GPU payload, CPU-verified) |
| MULTI_GPU_TRAINING_DDP | **YES** (torchrun world = strict divisor; existing DDP contract kept) |
| MULTI_GPU_RMS_ALLREDUCE | **YES** (fp64 SUM all-reduce; rank-0-only writes; historical RMS read-only) |
| MULTI_GPU_PRUNING_PARALLEL | **YES** (same-iteration full/minus jobs on separate GPUs; trajectory serial and byte-equal) |
| MULTI_GPU_EVALUATION | **YES** (contiguous chunks → single deterministic merge → one official evaluator) |
| FORMAL_EXPERIMENT_READY | **YES** |
| LEGACY_BYTE_COMPATIBLE | **YES** (six legacy vectors byte-identical incl. 3-rank batch-63; in-flight 2-GPU formal root resumes) |
| THROUGHPUT_MODE | opt-in only, RECIPE_EQUIVALENT=NO recorded (never silently 63/60/70 in strict mode) |

## 1. Architecture

The V7 task runner stays a **single-process orchestrator** (`WORLD_SIZE = 1`,
enforced by the run contract builder).  GPU-count adaptation is a *scheduling*
change only — the method contract (1536-D fixed query, Global Top-2, 4
candidates, rank-8 LoRA, selected-current-only updates, Answer+Key loss, RMS
calibration, iterative remove-and-reroute pruning, committed-only inference,
Task0→Task5 order, batch 64, LR 2e-4, warmup 0.03, cosine, BF16, seed 42) is
untouched.

Unified GPU entry (spec §2–§4): `--gpus` > `V7_GPUS` > `CUDA_VISIBLE_DEVICES` >
idle-GPU probe.  The probe only ever selects devices that are **already idle**
(used memory < 1024 MiB **and** 0% utilization); explicit lists pass a launcher
availability gate; nothing is ever stolen, killed or reset, and automatic
detection never touches a busy device.

## 2. Stage plan (strict recipe)

```
global_batch = per_device_batch(1) × training_world_size × gradient_accumulation
```

| declared GPUs | S1 query | S3 training | S4 RMS | S5 pruning | S6 eval | strict recipe |
|---|---|---|---|---|---|---|
| 1 | 1 | 1-rank | 1 | 1 | 1 | 1×1×64 = 64 |
| 2 | 2 | 2-rank DDP | 2 | 2 | 2 | 2×1×32 = 64 |
| 3 | 3 | 2-rank DDP (GPU 2 idle in S3) | 3 | 3 | 3 | 2×1×32 = 64 |
| 4 | 4 | 4-rank DDP | 4 | 4 | 4 | 4×1×16 = 64 |
| 8 | 8 | 8-rank DDP | 8 | 8 | 8 | 8×1×8 = 64 |

`--gpus` declares the *available* set; the planner derives stage assignment and
the S3 recipe deterministically (`compose/v7/gpu_plan.py`, 24 unit tests).  A
strict run can never silently become 63/60/70 — the effective global batch is
checked inside `build_run_contract` and recorded (`recipe_exact`).

## 3. Stage implementation (spec §10–§18)

- **S1 fixed queries**: per-GPU workers hash their own contiguous sample slice
  into `features/tmp/<split>.json.rank{k}`; the orchestrator validates
  per-shard coverage on resume, merges partials (duplicate detection, union ==
  expected ids, header agreement), recomputes `feature_hash`/`query_hash` over
  the full union and atomically writes `features/<split>.json`.  A 1-worker
  plan writes the canonical payload directly.
- **S3 training**: torchrun `--nproc_per_node <planned world size>` over the
  planned training GPU subset; env `CUDA_VISIBLE_DEVICES` = the subset only.
  The existing DDP contract is untouched (`find_unused_parameters=True`, key
  anchor, non-reentrant checkpointing, selected-only gradients, OldOld
  no-op, full-coverage audit).  S3 resume requires the identical world size
  (contract gate, fail-closed; elastic resume is not opened).
- **S4 RMS**: `torchrun` multi-rank RMS with fp64 SUM all-reduce and
  rank-0-only artifact writes (`compose/eval/rms_stats.py`); historical RMS is
  read-only committed calibration.  Single-worker plans use the legacy command
  shape.
- **S5 pruning**: `CandidatePruner.evaluate` submits every remove-and-reroute
  hypothesis of an iteration **before** consuming any score
  (two-phase restructure in `compose/v7/pruning.py`, output-identical for the
  synchronous scorer); the `PooledJobRunner` owns one GPU per job
  (`CUDA_VISIBLE_DEVICES` per job, lease recorded in
  `data/stage_gpu_usage.jsonl`).  Decisions keep removing exactly one
  candidate per iteration, in the exact serial order.  A content-verified job
  cache (`stages/job_<i>.done` + manifest equality + full-val NLL coverage)
  makes partial S5 resumes reuse only exact matches.
- **S6 inference**: contiguous chunks (`ceil(N/world)` per worker), per-chunk
  `answers.jsonl.rank{k}`, orchestrator merge validates line count + id order
  == the original test file and atomically writes the canonical answers file —
  the only official evaluator is the merged output.
- **S0/S2 and all merges**: worker temp outputs → orchestrator validation →
  deterministic merge → atomic rename (`mkstemp` + `fsync` + `os.replace`).
  Only the orchestrator writes `*.done` markers.

## 4. Resume & file-concurrency semantics

Every stage marker embeds the run-contract hash; `bind_run_contract` compares
the full stored contract.  Completed shards/jobs are reused only when their
coverage still matches exactly; any worker failure propagates (batch runner
raises with log paths) and no `*.done` marker is written.  The adaptive
contract adds the `gpu_plan` block + `data/gpu_plan.json`; legacy launches
(no `--gpus`) reproduce the HEAD contract byte-for-byte.

## 5. Equivalence evidence

- **Legacy byte parity**: contracts built by the rewritten module were compared
  to HEAD code for six vectors — world-1 GA64, world-2 GA32 (the in-flight
  root), world-3 GA21/batch-63, optimized world-3 (pdb=3, GA7, workers 8),
  smoke GA1, smoke world-4 GA4 — all `byte-identical: True`.
- **Parallel scorer == serial trajectory** (`tests/compose/test_v7_adaptive_equivalence.py`):
  a Deferred-backed capped worker pool produces identical
  `(retained, metrics, audit)` to the synchronous scorer, including the
  recorded nested scorer payloads, with identical call counts.
- **Shard merge == single-GPU payload**: merged 3-shard payload equals the
  assembled full payload byte-for-byte; header disagreement fails closed.
- Full suites: `test_v7_global_coevolution.py` 40/40,
  `test_gpu_plan.py` 24/24, `test_v7_adaptive_equivalence.py` 4/4,
  whole `tests/compose` 446 passed + 8 subtests (the only 2 failures,
  `test_ucit_evaluator_parity.py[VizWiz,Flickr30k]`, are pre-existing on HEAD
  and environmental: COCO caption scoring requires `java`, absent on this
  machine).

## 6. Performance

Worker fan-out is deterministic per declared GPU count (section 2).  Wall-clock
rows are **pending live measurement**: GPUs 0–1 are idle, GPUs 2–7 are busy
with other users' jobs, and this implementation refuses to run on busy
devices.  Benchmark harness:

```
for n in 1 2 4; do
  RECIPE_MODE=throughput RUN_ROOT=.../bench_${n}gpu \
    GPUS=0,1,2,3 ...  # 4-GPU rows require an idle 4-GPU window
done
```

Per-stage durations are collected per stage log (`logs/*.log` tail) and per
`eval/summary.json` `samples_per_second`; speedups are computed against the
1-GPU row of the same formal root.  Expected (planner-derived) parallel
scale: query/RMS/pruning/eval fan-out ×N, S3 throughput ×world size at
identical global batch (LR/warmup/scheduler untouched).

## 7. Commits (this change set)

1. `746f3f8` planner `compose/v7/gpu_plan.py` + 24 unit tests
2. `85eedc5` sharded query merge + worker pool (`workers.py`, equivalence tests)
3. `d5bd9f8` adaptive orchestrator (`v7_task_run.py`) + two-phase pruning
4. `634b554` auto launcher `v7_six_task_run_auto.sh` (legacy launchers untouched)

## 8. How to run

```bash
# strict (formal), explicit
./scripts/Compose/Run_UCIT/v7_six_task_run_auto.sh 0,1,2,3
# strict, env-declared
GPUS=0,1 ./scripts/Compose/Run_UCIT/v7_six_task_run_auto.sh
# strict, auto-detect (only idle GPUs are ever used)
./scripts/Compose/Run_UCIT/v7_six_task_run_auto.sh
# throughput benchmark (explicit opt-in, RECIPE_EQUIVALENT=NO)
RECIPE_MODE=throughput ./scripts/Compose/Run_UCIT/v7_six_task_run_auto.sh 0,1,2,3
```
