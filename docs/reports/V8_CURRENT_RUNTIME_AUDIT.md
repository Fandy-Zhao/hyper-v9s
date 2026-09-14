# V8 Current Runtime Audit

**Date:** 2026-09-11
**Branch:** `exp/v8-answer-supervised-multikey` @ `81ca61d`
**Method:** read the code and the artefacts of the runs that actually happened. Nothing
below is inferred from a design document; where a number is derived rather than measured,
it says so.

---

## 0. Executive summary (read this first)

The brief assumes V8 is a training loop with a per-step dynamic teacher whose historical
NLL is recomputed inside the step. **That is not what is on disk.** The audit found:

1. **V8 today = V8-A, an inference-only offline teacher run.** `compose/experiments/v8_task_run.py`
   generates answers and answer-NLLs over the *validation* split of one task with the pool
   frozen, and writes a teacher verdict per sample. It trains nothing
   (`v8_task_run.py:1139` — `"trainable": "none (V8-A is inference only)"`).
2. **The V8 training loop exists but has never run.** `compose/v8/trainer.py`
   (`V8TaskTrainer`) is assembled and unit-tested; its only callers are tests. It has no
   dataloader and no entry point (`docs/reports/V8_IMPLEMENTATION_AND_ACCEPTANCE_REPORT.md` §19.4).
3. **The teacher is *not* dynamic.** It runs once per task, before training, over the frozen
   pool. The current task's candidate experts are not in the search universe. The brief's
   rules 3/4 ("do not freeze a dynamic teacher; do not lower the refresh frequency") are
   therefore satisfied **by construction** — there is no dynamic teacher to preserve. This is
   the single most important audit result, because it removes the largest class of forbidden
   optimizations from the table: nothing about the Current Expert's NLL is cached anywhere,
   and nothing can be, because nothing recomputes it.
4. **The wall-clock that actually needs to shrink is the V7 six-task pipeline**, which V8-B
   reuses unchanged (report §19.2 derives V8-B's training cost directly from it).
   Measured over the 2026-09-03→06 formal run:

   | Stage | Per task, all six (measured) | Six tasks |
   | --- | --- | --- |
   | S3 training (`compose.train.train_compose`, `v7_global_coevolution`) | 4 h 00 m 37 s – 5 h 57 m 34 s | **32.10 h** |
   | S4 RMS calibration (`compose.eval.rms_stats`) | 18 m 12 s – **7 h 44 m 07 s** | **20.31 h** |
   | S5 pruning + commit | 20 m 32 s – **4 h 58 m 13 s** | **7.59 h** |
   | S0/S1/S2 (data, fixed queries, candidates) | ≈ 2 min | ≈ 12 min |
   | V8-A teacher over 256 val samples (one task, one scope) | 1 h 05 m – 2 h 27 m | (V8 adds this per task) |

   The three totals are generated, not hand-added:
   `python -m compose.experiments.stage_budget <run-root>` reproduces every cell
   from the run's own stage markers and logs, archived at
   `docs/reports/data/stage_budget_formal_20260903.json`, and it names the
   boundary it used for each stage. **An earlier revision of this table was
   wrong in a way worth recording**: it aggregated only tasks 1-4 — the four
   whose stages had finished when this audit was written — and estimated the
   six-task totals from them. The estimates were low, S4 badly so (7.2 h against
   the measured 20.31 h), and the per-task maxima hid the two tasks that matter
   most for the RMS stage: task 5's RMS alone is 7 h 44 m and its pruning stage
   4 h 58 m, both outside the ranges the earlier revision printed. The
   conclusion the table supports is unchanged and in fact stronger — see
   `V8_ACCELERATION_IMPLEMENTATION.md` §3.2.

   **The training step, not the data pipeline, is the bottleneck**, and it is bottlenecked by
   an execution choice, not by a semantic one: `per_device_train_batch_size = 1`.

5. **Four pure-execution defects were found that cost time without doing any work**, all of
   them safe to remove because they do not touch the loss, the routing or the optimizer:
   a 1.3 GB JSON query cache parsed at startup, one full 7B forward per (sample, route) in the
   teacher with the vision tower re-run each time, `batch_size=1` in the RMS stage that its own
   docstring says is "unconstrained", and ≈ 1 800 GPU synchronisations per micro-step in the
   gradient audit hooks.

The rest of this document answers A–R of the brief, one by one, then lists the measured
evidence, then states the acceleration plan the evidence supports.

---

## 1. What "V8" is, precisely

There are two layers. Confusing them is the main hazard of this task, so they are separated
here once.

| Layer | Entry point | Status | Trains? |
| --- | --- | --- | --- |
| **V8-A** — Answer-Supervised Expert Teacher | `compose/experiments/v8_task_run.py` → `compose/v8/teacher.py` | **run** (4 runs, 2026-09-11) | no |
| **V8-B** — current-task gated training | `compose/v8/trainer.py` (`V8TaskTrainer`) | **code complete, never run** — no driver | yes (if driven) |

V8-B, when driven, reuses the V7 training recipe verbatim: the report's cost table derives it
as "≈ V7's own ≈ 6 h on the same recipe". So *accelerating V8-B's training means accelerating
`compose.train.train_compose` in `v7_global_coevolution` mode*, and that is the largest single
block of wall-clock in the programme.

---

## 2. A–R: the audit questions

### A. The real training data flow of one task

`compose/experiments/v7_task_run.py` runs six stages per task; each stage is a subprocess and
each is idempotent behind a `.done` marker (`stages/s{0..5}_*.done`):

```
s0_full_data      materialise task{N}/data/train_full.json + val_full.json
s1_fixed_queries  query cache -> features/{train,val}.json   (encoder_calls = 0)
s2_candidates     initialise 4 candidate experts + keys from the task centre
s3_training       torchrun -m compose.train.train_compose --compose_mode v7_global_coevolution
s4_rms            torchrun -m compose.eval.rms_stats   (LoRA delta RMS -> kappa calibration)
s5_pruning_commit prune candidates, commit the pool, write task{N}/committed
```

For **V8** the flow prepends one stage, and it is the expensive one: the teacher sweep
(`v8_task_run.py:run_teacher`) over the frozen pool, writing `teacher_result.json`, which the
training stage of V8-B consumes through `compose/v8/cache.py:read_teacher_result`.

### B. The real data flow of one optimizer step

`V7ComposeTrainer.training_step` (`compose/v7/hf_trainer.py:335-395`), once per micro-batch:

1. pull `fixed_queries` out of the batch and move to the key device;
2. `self.v7_router(queries)` (`compose/v7/routing.py`) — cosine over every key, max per
   expert, Top-2 distinct experts. **This is dynamic: it is recomputed every micro-batch
   against the *current* key values, so the route co-evolves with training.** (Brief rule 7 —
   no long-lived route freeze — is respected by the baseline and must stay respected.)
3. `padded_compose_selections` → `inputs["compose_selections"]`;
4. `with model.no_sync(): super().training_step(...)` → `compute_loss`:
   - one model forward over the frozen backbone with per-sample LoRA deltas composed by
     `ComposeLinear` (`compose/adapters/lora.py:181-260`);
   - answer loss = sum of per-sample mean answer-token NLL (`v7_sum_per_sample_loss`);
   - key loss = per-sample mean `1 - cos(query.detach(), selected current key)`, weighted
     `lambda_key = 0.1`;
   - Old+Old routes add a zero-valued key anchor so `backward()` still runs and the step is a
     true no-op;
5. backward; `assert_historical_key_gradients_frozen` / `assert_historical_lora_frozen`;
6. at the accumulation boundary only: `mean_sync_accumulated_gradients` — one flattened
   all-reduce over all optimizer parameters, then `grad.div_(world_size)` (`hf_trainer.py:112-162`);
7. clip, `optimizer.step()`, scheduler step; write one JSONL row.

### C. What is trainable

Exactly two groups, enforced by an optimizer-time audit that refuses anything else
(`hf_trainer.py:265-275`):

| Group | Contents | LR | WD |
| --- | --- | --- | --- |
| candidate LoRA | the 4 current candidate experts' A/B matrices, 224 layers × 4 experts × 2 | 2.0e-4 | 0.0 |
| keys | the 4 current task's V7 keys, 1536-D each | 3.0e-4 | 0.0 |

### D. What is frozen

The whole 7B base, `embed_tokens`, the mm_projector (V7 mode refuses `tune_mm_mlp_adapter`),
the CLIP vision tower, every historical expert's LoRA, every historical key. The end-of-task
audit (`assert_task_freeze_integrity`) re-verifies by checksum and by `runtime_kappa_calibration`.

### E. Are queries already cached offline?

**Yes — twice, and the training path reads the worse of the two copies.**

* Efficient copy: `query_cache/task{N}/{split}/queries.pt` — one contiguous fp32 tensor
  `[N, 1536]` plus `sample_ids`, `query_tensor_hash`, `contract_hash`
  (writer `compose/eval/precompute_v7_queries.py:525-560`). task4 train = **245 MB**.
* Copy the trainer actually reads: `<root>/task4/features/train.json` — the same vectors as
  **JSON text, 1.30 GB**, loaded with a single `json.load` at
  `compose/train/train_compose.py:507-510` and materialised into a Python dict of 39 743
  lists of 1 536 floats.

Verified numerically on the val split: `json` record vs `queries.pt` row → **max abs diff 0.0,
0 differing elements**. The two copies are the same data.

### F. Where historical-expert answer NLL is computed, and how often

Only in the teacher, and only offline — never inside a training step.

`AnswerNLLScorer._forward` (`v8_task_run.py:342-385`) runs one full teacher-forced forward per
**(sample, route)** pair, then memoises it in `nll_cache.jsonl` keyed `(sample_id, route)`.

Measured counts on the finished formal runs:

| Run | samples | generations | live NLL forwards | NLL cache hits | wall clock |
| --- | --- | --- | --- | --- | --- |
| task 4, all experts | 256 | 2 612 | 2 396 | 216 | 3 917 s |
| task 4, history only | 256 | 2 826 | 1 968 | 858 | 3 982 s |
| task 3, all experts | 256 | — | — | — | 8 828 s |
| task 3, history only | 256 | — | — | — | 8 528 s |

Two structural redundancies are visible here and are quantified in §3:
(a) the same sample is scored under many routes, and each route re-runs the **vision tower**;
(b) the same sample is scored under `single_k` for every visible expert `k` — between 12 and
23 forwards that differ only in which LoRA delta is added.

### G. How Current Expert NLL is computed

**It is not.** No V8 code path computes the current candidate expert's answer NLL. The teacher's
search universe is the visible *historical* pool (`teacher.run(visible_experts=...)`,
`teacher.py:417`), asserted total by `assert_full_history_coverage`. This is DECISION-1 of the
2026-09-11 semantics round.

### H. Does teacher / responsibility change as the Current Expert updates?

**No, and it cannot.** The teacher runs once per task, before training, over a frozen pool.
The consequence for this task is important and negative-in-a-good-way: the brief's prohibitions
3 and 4 — do not pin a dynamic teacher to its task-start value, do not lower its refresh
frequency — describe a risk that **does not exist in this codebase**. There is no per-step
teacher, so no cache can illegally freeze one.

What *is* dynamic, and must stay dynamic, is the **V7 router inside training**: `GlobalTop2Router`
is re-run every micro-batch against the current keys (§B step 2). Route bucketing, route
caching, or "route once per epoch" would break it. It is on the forbidden list.

### I. Full 7B forwards per step

Per optimizer step: **64** (one per micro-batch — `per_device 1 × accum 16 × world 4`),
plus one recompute pass per micro-batch from gradient checkpointing, so **128** forward
passes of the 7B trunk per optimizer step, over 224 LoRA-instrumented decoder projections.

For the V8 teacher: **1 forward per (sample, route)**, plus one per teacher-forced NLL
(same count) — i.e. ≈ (1 + |visible| + |pairs|) per unsolved sample.

### J. Full 7B backwards per step

**64** per optimizer step (one per micro-batch). No extra backward from checkpointing beyond
the recompute forward in I.

### K. Duplicate vision-tower forwards

* **Training: none.** One forward per micro-batch, so one `encode_images` per sample
  (`llava/model/llava_arch.py:134-135`). LoRA deltas are added inside the decoder only.
* **V8 teacher: yes, and this is the single largest algorithmic redundancy in the V8-specific
  path.** `GenerationEngine._encode` caches the tokenised prompt and the *raw 336×336 pixel
  tensor* (`generate.py:207-225`), but `model.generate` re-runs `encode_images` on every call.
  A sample scored under `base`, then `single_07`, then `pair_07_11` pays the CLIP tower three
  times for one image. `AnswerNLLScorer._forward` has the same shape
  (`v8_task_run.py:349-356`). The CLIP tower is frozen and its output depends only on the
  image, so recomputing it is pure waste — **this is exactly the "delete repeated computation
  of a frozen module" the brief asks for, and it changes no number.**

### L. Duplicate tokenization

* **Training: once per sample per epoch.** `LazySupervisedDataset.__getitem__`
  (`compose/train/data.py:165-193`) re-opens the image (`PIL.Image.open`), re-pads, re-runs the
  image processor and re-tokenises on every access. With `num_train_epochs = 1` that is one
  pass, not a repeat — but it is *not* cached, so any resume re-does it and any epoch > 1
  would repeat it. It is a legitimate token/label cache target (brief P2), with the caveat that
  a cache must be fingerprint-bound or it will silently drift from the tokenizer.
* **V8 teacher: bounded.** `_encode` caches `(input_ids, image_tensor)` per sample id for the
  life of the engine (`generate.py:202`), so tokenisation happens once per sample.
  `AnswerNLLScorer` does **not** cache: it rebuilds the batch through
  `LazySupervisedDataset.__getitem__` + collator on **every** `_forward` call
  (`v8_task_run.py:343-344`), so it re-decodes the image and re-tokenises for every single
  (sample, route). Measured: 2 396 such rebuilds in the task-4 run.

### M. Global / effective batch

**64 samples per optimizer step.** `DEFAULT_TARGET_GLOBAL_BATCH = 64`
(`compose/v7/gpu_plan.py:40`), asserted by the launcher in strict mode. The decomposition
depends on how many GPUs were free:

| Task | world size | grad accum | per-device | global |
| --- | --- | --- | --- | --- |
| task 0 | 2 | 32 | 1 | **64** |
| task 1–5 | 4 | 16 | 1 | **64** |

This is the invariant that must not move. It is also what makes P5 (raise micro-batch, lower
accumulation proportionally) the highest-value legal change: 1×16 → 2×8 or 4×4 keeps the
global batch at exactly 64.

The *data stream* is part of that invariant, and it is worth stating where it comes from.
`Trainer` builds the dataloader at `per_device_train_batch_size` and hands it to
`accelerator.prepare`, whose `BatchSamplerShard` deals batches round-robin; rank *r* ends a
step after its `GA`-th batch. The sampler's megabatch is therefore
`per_device x world_size x GA` — **the same 64 samples at world 1, 2 and 4 alike** — and
`TrainingArguments.n_gpu` does not enter it, being 1 per process under `torchrun`. A step
of 64 samples over 4 GPUs costs less wall clock than over 2; it does not train on different
data. This was measured, not read off the source: `V8_RECIPE_EQUIVALENCE_REPORT.md` §1.1
has the sample counts per rank and the two real runs behind them.

### N. Micro-batch

**1 sample.** `--per_device_train_batch_size 1`, and `compose/v7/config.py:56` defaults to 1.

### O. Gradient accumulation

16 (world 4) or 32 (world 2), derived by the plan so that `per_device × accum × world = 64`.

### P. Sequence length and padding ratio

Not yet instrumented — this is what the profiler is for. What is known structurally:
`image_aspect_ratio = pad` makes every image exactly 576 tokens, and `model_max_length = 2048`,
so the only length variation is the question text. **At micro-batch 1 the padding ratio is
exactly 0**; padding only becomes a cost once the micro-batch rises, at which point
`--group_by_modality_length True` (already on) groups similar lengths together.

### Q. DataLoader wait time

Not yet instrumented. Structurally it should be small: 4 workers, batch 1, ~10-20 ms of
PIL + preprocess per sample against a ~2.16 s micro-step budget. The `inter_step_wait_sec`
field already logged in `train_steps.rank*.jsonl` is a related but different quantity
(rank-skew before the step). This is measured in the baseline profile, not assumed.

### R. Time share of forward / backward / communication / optimizer / validation

Not yet instrumented at this granularity. What is measured today is the *total*: 34.55 s per
optimizer step (task 4, 621 steps, 5 h 57 m 34 s) and ≈ 2.16 s per micro-step. Splitting that
number is the purpose of the profiler described in §4.

---

## 3. Measured evidence

### 3.1 The per-task stage budget (from `.done` marker timestamps)

| Task | S3 training | S4 RMS | S5 pruning+commit |
| --- | --- | --- | --- |
| 1 | 5 h 22 m 40 s | 18 m 13 s | 20 m 33 s |
| 2 | 5 h 50 m 44 s | 56 m 38 s | 51 m 50 s |
| 3 | 4 h 04 m 41 s | 2 h 12 m 50 s | 25 m 02 s |
| 4 | 6 h 02 m 29 s | 4 h 07 m 15 s | 23 m 34 s |

RMS is not a rounding error: on task 4 it is **68 % of the training stage's duration** for
256 validation samples. Its own command line says `--batch-size 1`, and its own docstring says
"the batch size is unconstrained" (`compose/lora/rms.py:184-186`) — the hook path recomputes
each expert's delta per `ComposeLinear` layer (224 layers × up to 20 experts) per sample.

### 3.2 The per-step budget

`task4/logs/training.log`: 621 steps, `train_runtime 21 454.65 s`, `train_samples_per_second
1.852`, `train_steps_per_second 0.029`, i.e. **34.55 s per optimizer step ≈ 0.54 s per sample**.

### 3.3 The query cache

`features/train.json` for task 4 is **1 304 390 670 bytes**, parsed whole with `json.load`.
`query_cache/task4/train/queries.pt` is **245 243 992 bytes** and holds the identical vectors
(§E). The `.pt` is already produced by the pipeline; nothing reads it in the training path.

### 3.4 The GPU fleet

8 × RTX 4090 (24 GiB). At audit time GPUs 3 and 7 are idle; 0, 1, 2, 4, 5, 6 are held by other
users. The repo's own convention (and the V7 launcher's gate) is to use only idle devices, which
sets the experiment plan: a 2-rank DDP run on GPUs 3,7 reproduces the **task-0 strict recipe
exactly** (world 2, accum 32, global 64).

---

## 4. The profiler to be added (P0)

A single low-overhead `TrainingProfiler` (`torch.cuda.synchronize()` only at the boundaries it
times, one JSONL row per optimizer step, no per-parameter synchronisation), enabled by
`--profile_training`, recording:

`data_wait_time`, `query_load_time`, `token_load_time`, `routing_time`,
`vision_forward_time`, `llm_forward_time`, `backward_time`, `allreduce_time`,
`optimizer_time`, `validation_time`, `checkpoint_time`, plus
`samples/sec`, `tokens/sec`, `steps/sec`, `peak_vram`, `allocated_vram`,
`padding_ratio`, `mean_seq_len`, `model_forwards_per_step`, `tokens_per_second_effective`.

It must be **off by default** and must not change any number it does not measure: every timer
is a `perf_counter()` pair around an existing call, and the only device synchronisation is at
the outermost boundary of each phase.

---

## 5. Acceleration plan the evidence supports

Ordered by measured leverage. Every item is an *execution* change; none touches the loss, the
router, the teacher semantics, the optimizer, the LR, the scheduler or the global batch.

**The `A` column is this audit's own leverage ordering, not the brief's ablation.** The brief's
numbered S0–S11 steps are a *different cut* of overlapping work — its S6 is the micro-batch
split below, its S5 is the sync removal, its S7 is checkpointing — and the two must not be read
against each other. The brief's ordering, with the measurement that decided each step, is in
`V8_ACCELERATION_IMPLEMENTATION.md` §7.

| # | Change | Expected effect | Why it is recipe-safe |
| --- | --- | --- | --- |
| A1 | Read `queries.pt` instead of the 1.3 GB `train.json` | minutes of startup + GBs of RAM | vectors verified bit-identical |
| A2 | Micro-batch 1 → 2/4 with accumulation 32/16 → 16/8 | the big one: batch-1 kernels are launch-bound | global batch stays exactly 64 |
| A3 | Cache the frozen vision tower output per sample in the teacher | removes (routes − 1) × CLIP forwards per sample | frozen module, image-only input |
| A4 | Batch the teacher's per-route forwards | same, for the LLM side | needs a numeric equivalence gate on teacher labels |
| A5 | RMS stage: raise `--batch-size` from 1 | 4 h → tens of minutes on task 4 | accumulation is a sum over samples; gate on kappa delta |
| A6 | Replace ≈1 800 per-step GPU syncs in the gradient hooks with one | few % | identical numbers, fewer synchronisations |
| A7 | Attention: eager → SDPA/flash, behind a flag | forward/backward kernel efficiency | **numerics change** — flag + 100–500 sample equivalence check, off by default |
| A8 | Gradient checkpointing off at the micro-batch that fits | trades recompute for memory | pure execution; benchmark decides |
| A9 | Token/label cache for `LazySupervisedDataset` | only matters at epochs > 1 or heavy resume | fingerprint-bound cache |

None of A1–A9 changes the loss, the router, the teacher semantics, the optimizer, the LR, the
scheduler or the global batch. A3, A4, A5 and A7 nevertheless change the *numbers* the code
produces (A3/A4 through kernel nondeterminism, A5 through summation order, A7 genuinely), so
they ship **behind flags, off by default**, each with its own equivalence measurement — the
brief's rule that a V8-Fast must not be mistaken for a V8-Exact is enforced by the flags, not by
good intentions.

---

## 6. What this audit changes about the brief

Three of the brief's premises do not hold in this repository, and the plan is adjusted rather
than the facts:

1. **"Teacher / responsibility changes as the Current Expert updates"** — it does not. The
   teacher is offline and one-shot per task. Nothing is at risk here; nothing needs a
   refresh-frequency guarantee.
2. **"Historical Expert NLL is recomputed in the training loop"** — it is not. The NLL cache
   already exists (`nll_cache.jsonl`) and the training loop never computes a historical NLL.
   The real redundancy is *within* the teacher sweep: the same (sample, expert) is scored
   under multiple routes and each one re-encodes the image.
3. **"V8 single-task wall-clock"** — V8-B has never run, so there is no V8 training baseline to
   profile. The baseline that exists and can be profiled is the V7 training stage, which V8-B
   reuses unchanged and which the project's own cost model uses for V8's estimate.

Everything the brief forbids remains forbidden. In particular the audit found no reason —
speed or otherwise — to change: the router's per-micro-batch Top-2 (dynamic route),
`residual_weights` (never drop a sample, weight it to zero), the teacher's total-over-pool
search, the answer-NLL oracle, the Key loss, the optimizer, the LR, the scheduler, the global
batch, or the task-end candidate lifecycle.
