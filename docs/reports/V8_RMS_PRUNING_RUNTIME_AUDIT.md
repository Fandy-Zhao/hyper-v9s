# V8 RMS + Pruning Runtime Audit

**Date:** 2026-09-12
**Branch:** `exp/v8-answer-supervised-multikey`
**Method:** read the shipped code, then measure the shipped code. Every number below is
either quoted from a run artefact or produced by a timing harness that calls the production
functions directly. Where a number is derived rather than measured, it says so.

---

## 0. Executive summary

1. **"RMS + pruning" is stages S4 and S5 of `compose/experiments/v7_task_run.py`**, the
   six-stage per-task pipeline that V8-B reuses verbatim
   (`V8_CURRENT_RUNTIME_AUDIT.md` §1-§2). V8-A (`compose/experiments/v8_task_run.py`) is an
   inference-only teacher and has no RMS or pruning stage.

2. **The measured cost of those two stages is 27.90 h over six tasks** — RMS 20.31 h, pruning
   7.59 h — against 32.10 h of training. They are 47% of the run's non-training wall-clock.
   Source: `compose.experiments.stage_budget` over the 2026-09-03 formal run.

3. **Both stages are CPU-bound, not GPU-bound.** A clean single-GPU RMS run holds the GPU at
   ~20% utilisation while one CPU core saturates at ~99%. The RMS stage is a Python loop that
   synchronises the GPU three times per `OnlineMoments.update`, twice per (layer, expert), for
   224 layers × 24 experts per sample — about **32 000 CUDA synchronisations per sample**.

4. **Two thirds of the RMS statistics work is arithmetically redundant.** `module_output` is
   passed to `RMSStatistics.update` once per expert, unchanged across experts, and its moments
   are reduced over 2.46 M elements versus the delta's 4 800. The output moments are therefore
   ~99.8% of the element-work and 23/24 of it is the same number recomputed. The calibration
   itself (`build_kappa_calibration` → `delta_rms`) ignores the output moments entirely.

5. **The pruning stage re-scores the full pool once per remove-and-reroute iteration**, and the
   re-score is *numerically identical* to a job it already ran: task 5's iteration-1
   `metric_full = 55.91` is exactly iteration-0's `metric_minus` for the expert that was
   removed. That is one whole 256-sample generation job discarded and recomputed.

6. **Every pruning job is a fresh subprocess that reloads the 7B model — twice.** One job is
   `nll_eval` (teacher-forced NLL) + `eval_task` (autoregressive generation) + a metric step,
   and the first two are separate processes with separate model loads.

7. **Parallelism exists but was silently lost.** `PooledJobRunner` can run one configuration per
   GPU, but on this run the idle probe narrowed the pool to a single GPU:
   `S5 pruning pool narrowed by idle probe: (0, 1, 2, 3) -> [0] (planned GPUs busy)`
   (`tasks1_to5.log:323`). All nine of task 5's jobs ran one after another at ~28 min each.

The optimization targets that follow from this are listed in §5, ranked by (wall-clock saved)
÷ (risk to equivalence).

---

## 1. Scope: what the two stages actually are

| Stage | Entry point | Marker | What it writes |
| --- | --- | --- | --- |
| S4 RMS calibration | `compose.eval.rms_stats` (torchrun, 2-4 ranks) | `stages/s4_rms.done` | `rms/rms_statistics.json`, `rms/rms_calibration.json`, `rms/rms_report.json`, `rms/rms_summary.json`; patches `rms_calibration` into the checkpoint manifest |
| S5 pruning + commit | inline in `v7_task_run.py` (scorer subprocesses) | `stages/s5_pruning_commit.done` | `metrics/candidate_pruning.json`, `metrics/candidate_pruning_trajectory.json`, `committed/` |

Both are **task-end, frozen-model** evaluation. Nothing in either stage trains, takes a
gradient, or mutates an expert's weights. S5's only mutation is the pool membership it commits.

---

## A. RMS — the real flow

### A.1-A.5 Dataset, size, split, batch size, per-expert repetition

| Question | Answer | Evidence |
| --- | --- | --- |
| A1 dataset | The task's **validation** split, `task{N}/data/val_full.json` | `v7_task_run.py` S4 argv; `ComposeRMSConfig` rejects any split containing `test` |
| A2 samples | **256** for every task | `rms/rms_summary.json` → `"samples": 256`, all six tasks |
| A3 split | validation only | `CALIBRATION_SPLIT = "validation"` (`rms_stats.py:53`) |
| A4 batch size | **1** | `v7_task_run.py` S4 argv passes `--batch-size 1`; the module default is 8 and is unused |
| A5 one pass per expert? | **No.** One backbone forward collects **every** expert at **every** layer | `compute_expert_rms` (`compose/lora/rms.py:181-192`) registers one forward hook per `ComposeLinear`, and the hook loops over all `expert_ids` |

This is the single most important structural fact: the RMS stage is *already* "one forward,
all layers". It is **not** the "for layer: model forward" anti-pattern. The cost is in what
the hook then does with the activation, not in how many forwards are run.

### A.6 Which experts need RMS

All experts registered in the checkpoint submitted to S4 — historical **and** new candidates.
`rms_stats.py:261` takes `bundle.expert_pool.expert_ids()`. Pool size grows by 4 candidates per
task: 4, 8, 12, 16, 20, **24** for tasks 0-5 (`rms_summary.json` per task).

### A.7-A.9 The RMS definition

Per layer `l` and expert `k`, over the calibration samples `x`:

```
R_k_l = sqrt( sum_x sum_elements (B_k_l A_k_l h_l(x))^2  /  count )
```

`OnlineMoments.rms` (`statistics.py:76-78`) is `sqrt(sum_squares / count)` — a root-mean-square
over the flattened delta elements pooled across all samples and tokens of the batch sequence.
Collection point is the **post-activation LoRA delta** `module.experts[key](hidden)`, computed
inside a forward hook from the module's input `hidden`, then cast to `module_output.dtype`
(`rms.py:158`). It is **not** the hidden state and **not** the module output.

Aggregation across ranks is fp64 all-reduce of `(count, sum, sum_squares)` (`statistics.py:89-102`)
— additive moments, so the merge is exact and order-independent. kappa is then

```
R_bar_l = mean_k R_k_l          (arithmetic mean over ALL active experts in the layer)
kappa_k_l = clip(R_bar_l / (R_k_l + 1e-8), 0.25, 4.0)
```

(`build_kappa_calibration`, `rms.py:211-255`). Committed experts' kappa is frozen and the
runtime merge only adds values for `--new-expert-ids` (`merge_commit_frozen_calibration`).

### A.10 Accumulation precision

Per-batch reduction in **fp32** on device; the Welford merge and the cross-rank aggregation are
**fp64** (`OnlineMoments`, docstring `statistics.py:35-56`).

### A.11-A.14 Vision tower / projector / per-layer forwards / per-layer registration

| Question | Answer |
| --- | --- |
| A11 vision tower repeated? | **No.** Once per batch, inside the single model forward |
| A12 projector repeated? | **No.** Same forward |
| A13 one forward per layer? | **No.** One forward per batch; 224 hooks fire inside it |
| A14 per-layer registration? | 224 `register_forward_hook` calls, once per run (`rms.py:181-182`) — negligible |

### A.15-A.19 Model reload, grad, dropout, eval mode

| Question | Answer |
| --- | --- |
| A15/A16 reload base model or expert per expert? | **No.** One load per rank for all 24 experts |
| A17 grad enabled? | **No** — `torch.inference_mode()` (`rms.py:189`) |
| A18 `model.train()`? | **No** — `model.eval()` (`rms.py:187`) |
| A19 dropout / stochastic preprocessing? | No. Adapter `dropout = 0.0` (checkpoint manifest). Images built deterministically by `_build_batches` |

So the "obvious" wins from the brief's §4 and §7 are **already in place**. The RMS stage is
correctly inference-only, correctly single-forward, correctly single-load. What is left is the
hook's inner loop.

### A.20 Measured RMS cost

Harness: `compose/experiments/profile_rms.py`, which calls the production
`_build_batches` / `load_compose_model` / `compute_expert_rms` on the task-5 checkpoint
(24 experts, 224 `ComposeLinear` layers, LoRA rank 8, alpha 16), 8 samples, batch size 1,
one RTX 4090, `torch.cuda.synchronize()` around each phase.

| Phase | 8 samples | Per sample | Share |
| --- | --- | --- | --- |
| Model load (base 7B + 24 experts) | 92.4 s | — | one-off |
| Batch build (I/O, tokenise, image preprocess) | 0.26 s | 0.033 s | 1.1% |
| Backbone forward, **no hooks** | 1.59 s | 0.199 s | 6.8% |
| Hooks + statistics (`compute_expert_rms`) | 23.35 s | 2.919 s | 100% |
| **hook overhead** (row 4 − row 3) | **21.76 s** | **2.720 s** | **93.2%** |

Event counts per sample: 224 layers × 24 experts = **5 376** `RMSStatistics.update` calls,
each running **2** `OnlineMoments.update` calls (delta and output) = **10 752** moment updates,
each doing **3** `.item()` GPU→CPU synchronisations = **32 256 CUDA synchronisations per
sample**. That is the whole story of the 93%: the GPU finishes each 4 800-element reduction
instantly and then waits for Python to collect the scalar.

Element-work split (derived, not measured): the delta is `(1, seq, 8)` ≈ 4 800 elements, the
module output is `(1, seq, 4096)` ≈ 2.46 M. The output moments are therefore **~99.8%** of the
reduction element-work, and `RMSStatistics.update` recomputes them **once per expert** even
though the tensor is identical for all 24.

Extrapolation to the real 256-sample stage on one GPU: 256 × 2.92 s ≈ **12.5 min** plus 92 s
load. The 2026-09-03 formal run reported **4.99 h (task 0) to 7.74 h (task 5)** for the same
stage. The gap is contention, not code: the same run's own idle probe found GPUs 1-3 held by
other tenants. §C reports both the clean measurement and the production figure, and the
speedup claim is made against the clean one.

### A.21 What the calibration actually depends on

`build_kappa_calibration` reads **only** `stats.delta_rms(expert_id, layer)`, which reads only
`OnlineMoments.rms` = `sqrt(sum_squares / count)`. The `output` moments feed
`rms_statistics.json` and the `output_rms` column of `rms_report.json` — diagnostics, never
the kappa. `mean` and `m2` feed the reported mean/variance, never the kappa.

This matters for §5: the largest single redundancy in the stage is in a term the calibration
does not consume, so removing it is exact by construction *and* verifiable against the
artefacts.

---

## B. Pruning — the real flow

The shipped implementation is **`compose.v7.pruning.CandidatePruner.evaluate`**, called at
`compose/experiments/v7_task_run.py:1337`. (`compose/v8/pruning.py` is a different thing: pool
bookkeeping and key/expert *planning* helpers used by the V8 trainer, not the remove-and-reroute
audit.)

### B.1-B.2 Metrics used

Two scalars per configuration, both from the same scorer call:

| Field | Meaning | Source |
| --- | --- | --- |
| `metric` | the **task-specific official UCIT metric** (e.g. Flickr30k captioning Average) | `compose.eval.v7_validation_metric` over generated answers |
| `loss` / `answer_nll` | mean teacher-forced answer NLL over the 256 validation samples | `compose.eval.nll_eval` |

`removal_gain_metric = full.metric − minus.metric` and
`removal_gain_loss = minus.loss − full.loss` (`pruning.py:108-109`). Usage, key cosine and
redundancy are additional *diagnostics*; the delete rule is
`gain_metric <= 0 and gain_loss <= 0`, or membership of a redundant pair
(`pruning.py:145-159`). On task 5 the official metric was active, not the NLL fallback:
`performance_metric = task_specific_official_ucit` (`metrics/candidate_pruning.json`).

### B.3-B.9 The remove-and-reroute loop

```
iteration 0:  score(pool)                 ← "full"
              score(pool \ {E})  for each surviving E
              commit the single best removal, or stop
iteration 1:  score(pool \ {E_removed})   ← recomputed, identical to the job above
              score(pool \ {E_removed} \ {E})  for each surviving E
              ...
```

| Question | Answer |
| --- | --- |
| B3 full-pool evaluations | **one per iteration**, `pruning.py:70-71` |
| B4 removal evaluations | one per surviving expert per iteration, `pruning.py:92-102` |
| B5 reroute after removal? | **Yes** — `self.router(val_queries, excluded=excluded \| {expert_id})` (`pruning.py:92`) |
| B6 code location | `compose/v7/pruning.py:68-200`; `GlobalTop2Router` in `compose/v7/routing.py` |
| B7 pruning metric | official generation metric **and** answer NLL, both |
| B8 full-pool recomputed per candidate? | Full pool is scored **once per iteration**, not once per candidate — but iteration *k+1*'s full-pool job is a byte-identical re-run of iteration *k*'s removal job |

**B8 is the confirmed duplicate.** Task 5's trajectory (`metrics/candidate_pruning_trajectory.json`):

```
iteration 0, candidate 22, decision=remove,   metric_full=55.40, metric_minus=55.91
iteration 1,                 metric_full=55.91  ← same number, recomputed from scratch
```

The routes are the same tensor: iteration *k+1*'s `router(val, excluded=E ∪ {X})` is exactly
iteration *k*'s `router(val, excluded=E ∪ {X})` for the removed `X`. Same routes → same
selections manifest → same NLL, same generations, same metric.

### B.10-B.11 Usage and redundancy

| Question | Answer |
| --- | --- |
| B10 does Usage re-run the model? | **No.** `route_usage` (`pruning.py:29-36`) counts from the route tensor; `train_usage`/`val_usage` come from `full_train`/`full_val` route tensors only |
| B11 does parameter redundancy call the LLM? | **No.** `pairwise_key_cosine` over key vectors; `key_norm` and `key_to_task_center_cosine` are tensor ops (`pruning.py:52-59`, `216-222`) |

So the brief's §12 and §13 are **already satisfied**. No optimization is available there.

### B.12-B.15 Tokenisation, vision tower, model load, serial execution

| Question | Answer |
| --- | --- |
| B12 repeated tokenisation? | Yes across jobs — each `nll_eval` process rebuilds `LazySupervisedDataset` and re-tokenises all 256 records |
| B13 repeated vision tower? | Yes — `nll_eval` and `eval_task` each run the vision tower over the same 256 images, in **separate processes**, for the same job |
| B14 repeated model load? | **Yes, twice per job.** `execute_scoring_job` shells out to `compose.eval.nll_eval` and then `compose.eval.eval_task`; each loads the 7B base + 24 experts (~92 s measured) |
| B15 configurations serial? | **Yes on this run** — `PooledJobRunner` was given a 1-GPU pool by the idle probe, so 9 jobs × ~28 min ran back to back |

### B.16-B.18 Candidate count, validation size, cost

| Question | Answer |
| --- | --- |
| B16 candidates | **4** per task (`retained_candidate_ids` + pruned = 4) |
| B17 validation size | **256** samples |
| B18 total cost | task 5: **4 h 58 m** wall-clock for 9 jobs on 1 GPU, ≈ 28 min/job |

Job inventory for task 5 (9 jobs = iterations 1×(1+4) + 1×(1+3), one of them a duplicate):

```
job 0  full pool            (iteration 0)
job 1  remove {20}          job 2  remove {21}     job 3  remove {22}    job 4  remove {23}
job 5  full pool minus 22   ← DUPLICATE of job 3
job 6  remove {20,22}       job 7  remove {21,22}  job 8  remove {23,22}
```

### B.19 How much of a job is genuinely new work

Routing on task 5's validation split collapses to **8 distinct routes** over 256 samples
(`audit.val_routes`, all of them 2-subsets of {20,21,22,23}; the 20 historical experts never win
a Top-2 slot on this split). Removing one expert leaves every sample that did **not** route to
it on the same route, with the same image, the same prompt and the same expert set — so its
generation and its NLL are the same computation. Per-sample counts in the full-pool routes:
expert 20 in 157/256, 21 in 125/256, 22 in 131/256, 23 in 99/256. A removal job therefore
re-uses ≈ 40-60% of the previous job's per-sample evidence and only has to evaluate the
remainder.

---

## C. Wall-clock decomposition

Production figures are from `compose.experiments.stage_budget` over
`/data/ckpt/.../v7_gpu01_cached_query_formal_20260903` (stage-marker boundaries, guarded
against the two known marker re-stamps). Clean figures are this audit's own measurements.

### C.1 Production, per task (hours)

| task | RMS | prune | candidates | pool at RMS | RMS GPUs | prune GPUs |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 4.99 | 0.60 | 4 | 4 | 2 | 1 |
| 1 | 0.30 | 0.34 | 4 | 8 | 4 | 1 |
| 2 | 0.94 | 0.86 | 4 | 12 | 4 | 1 |
| 3 | 2.21 | 0.42 | 4 | 16 | 4 | 1 |
| 4 | **4.12** | 0.39 | 4 | 20 | 4 | 1 |
| 5 | **7.74** | **4.97** | 4 | 24 | 4 | 1 |
| **total** | **20.31** | **7.59** | | | | |

Tasks 4 and 5 sum to **4.51 h** — the figure in the brief — and task 5 alone is 12.71 h.

### C.2 RMS, requested decomposition

Only the clean measurement can be split into phases; the production clock is a single
subprocess boundary and its internal split was never logged. Clean, per sample, one GPU:

```
RMS:
    model load                     92.4 s   (one-off, not per sample)
    data load / I-O                 0.033 s   (1%)
    vision forward (inside B)        ~0.045 s (derived: CLIP-L/336 on 1 image)
    LLM forward                      ~0.154 s (derived: remainder of the 0.199 s B)
    RMS statistics (hooks)           2.720 s  (93%)
    ------------------------------------------
    total                            2.919 s/sample
```

Production, six tasks, requested split (the resolution the boundary supports):

```
RMS total            20.31 h
  of which task 5     7.74 h   ← 2-4 GPUs, contended
  of which tasks 0-4 12.57 h
```

### C.3 Pruning, requested decomposition (task 5, the worst case)

```
Pruning total               4 h 58 m      9 jobs, 1 GPU, ~28 min/job

per job (approximate split, from the two subprocess logs):
    model load  ×2            ~3 m 04 s   (2 × 92 s; nll_eval and eval_task are separate processes)
    nll_eval (256 fwd)        ~2 m
    eval_task (256 generate) ~20 m        ← dominates
    metric (CIDEr/METEOR)     ~1 m
    --------------------------------------
    total                     ~28 m

Full-Pool evaluation        1 job  (recomputed every iteration: 1 duplicate in 9)
Removal E_j                 4 jobs iteration 0
Removal E_j (iteration 1)   3 jobs
Usage                       0 s — computed from route tensors
Redundancy                  0 s — pairwise key cosine, CPU
Metric aggregation          ~1 m/job
Model loading               ~11% of the stage
I/O                         < 1%
```

### C.4 The GPU is idle for most of both stages

Measured during the clean RMS baseline: GPU utilisation **19-20%** while both worker processes
sit at ~99% CPU. For pruning, the same shape applies per job: the generation loop is
`for record in tqdm(records)` with one `model.generate` per record at batch size 1, and the
NLL loop is `for record_index in indices` with one forward per record — despite `nll_eval`
accepting a `--batch-size` argument, **that argument is parsed and never used**
(`nll_eval.py:60` declares it, the loop at `nll_eval.py:113` indexes one record at a time).

---

## D. What is already optimal (do not "optimize" these)

Optimizing these would risk equivalence for no gain, and the brief's §12/§13 ask for them
explicitly. They are already true:

| Brief item | Status |
| --- | --- |
| Training loop untouched | N/A — neither stage trains |
| `model.eval()` / `torch.inference_mode()` | already in `compute_expert_rms` |
| No backward / optimizer / grad accumulation | already absent |
| Usage from route records, no extra forward | `route_usage` is pure counting |
| Parameter redundancy without the LLM | `pairwise_key_cosine`, CPU only |
| One forward for all layers (RMS) | already one forward; 224 hooks inside it |
| Model loaded once per RMS rank | already once per rank |
| Deterministic sample order | `shard_records` is contiguous and order-preserving; `mean_nll` averages in record order |
| Reroute after removal | implemented correctly; §11's red line is respected by the baseline and must stay so |

---

## E. What the numbers say to change, ranked

| # | Change | Saves | Risk to equivalence |
| --- | --- | --- | --- |
| 1 | RMS: compute the `output` moments **once per layer per batch**, merge into every expert's entry | ~24× on the dominant term of the stage; ≈ 60% of RMS | none — same tensor, same kernels, same merge sequence per entry |
| 2 | RMS: stop calling `.item()` per moment update; accumulate on device and sync once per batch | the ~32 k syncs/sample; the remaining ~40% of RMS | none — identical kernels, only the sync *timing* moves |
| 3 | Pruning: reuse iteration *k*'s removal job as iteration *k+1*'s full-pool score | 1 of 9 jobs on task 5 | none — provably the same routes |
| 4 | Pruning: per-(sample, route) evidence cache; only evaluate samples whose route actually changed | ~40-60% of each removal job | none — same deterministic function of the same inputs |
| 5 | Pruning: one persistent worker per GPU, model resident, evaluating many configurations | ~11% (the model reloads) plus it makes #4's cache actually hit | adapter/route switching must be shown equivalent |
| 6 | Pruning: run configurations concurrently on the free GPU pool | up to (pool size)× | none — independent configurations |
| 7 | Both: batch size | large, but kernel shapes change | **excluded from round 1** per brief §18 |

Rungs R1-R8 of the implementation ladder, and which of these each carries, are in
`V8_RMS_PRUNING_ACCELERATION_IMPLEMENTATION.md`.
