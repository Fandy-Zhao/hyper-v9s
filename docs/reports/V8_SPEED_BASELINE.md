# V8 speed baseline: where the time actually goes

The P0 deliverable. The brief asks for the real bottleneck, "not guesses", so
this document is a trace that was taken and a table that was computed, and it
names the bottleneck the trace points at — which is not the one the phase
columns make look largest.

**Run:** the frozen pre-V8 command line, unmodified — no V8 flags, no
`--compose_v8_config`. Task 4 (`train_full.json`, 39 743 samples), seed 42,
world 2 on GPUs 3 and 7, micro-batch 1 x 32 accumulation, 200 optimizer steps.
Output: `/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs/v8_baseline_task4_200step_20260911/`.
Profiling: `--profile_training True` with the default sync-paired boundaries.

---

## 1. The headline

Measured over the run's **197 analysed steps** (steps 4-200; the first three are
model-adaptation warm-up and are dropped):

| | value |
| --- | --- |
| **wall clock per optimizer step** | **64.36 s** (both ranks, agree to 0.1 ms) |
| samples per second (global) | **0.994** |
| tokens per second (global) | **645** |
| optimizer steps per second | **0.0155** |
| **GPU utilization** | **~30 %** (1 Hz `nvidia-smi` samples, 388 per GPU) |
| peak allocated VRAM | 15.13 GiB per rank |
| peak reserved VRAM | **17.43 GiB** |

**GPU utilization of 30 % is the finding.** A step that is compute-bound would
sit at 95-100 %. At 30 %, the GPU is idle for roughly 70 % of every step, and the
phase table below should be read with that in mind: the largest phase is not the
largest *cost*.

(The allocated and reserved figures are both correct and differ for a reason that
is not a memory difference: allocated peaks at 15.1310 GiB on both ranks, while
reserved peaks at 17.4297 GiB. The 2.3 GiB gap is the caching allocator holding
freed blocks rather than any extra requirement — the world-1 arms of the S5 and
S7 A/Bs, which carry the same model on one rank, report the same 15.128 GiB
allocated at 15.527 GiB reserved. An earlier revision quoted 15.53 GiB here,
which was the single-rank figure.)

### 1.1 What this baseline is a baseline *of*

The production task-4 training run
(`.../v7_gpu01_cached_query_formal_20260903/task4/logs/training.log`) ran a
different **allocation** of the same recipe:

| | production | this baseline |
| --- | --- | --- |
| GPUs (`nproc_per_node`) | **4** | 2 |
| `per_device_train_batch_size` | 1 | 1 |
| `gradient_accumulation_steps` | **16** | **32** |
| **effective batch** | **64** | **64** |
| s/optimizer step | 34.55 | 64.36 |
| steps × 621 | 5 h 58 m | 11 h 06 m |
| **samples/s per GPU** | **0.463** | **0.497** |

The effective batch is 64 in both. The production run is faster per *step*
only because it spreads the same 64 samples over twice as many GPUs — and the
per-GPU rates agree to **6.9 %**, which is the real measure of whether these two
runs cost the same thing. They do.

This matters for reading every number that follows, and for the speedup report:

- **Step time is not comparable across world sizes.** A 2-GPU measurement of
  64.36 s/step is not "the baseline is slow", it is "the baseline was given half
  the GPUs". Any speedup claim must be **per-GPU**, and every S6 arm in this
  exercise was run single-GPU against a single-GPU base for exactly that reason.
- **The production reference for the ETA table is 34.55 s/step, not 64.36.**
  `V8_SPEEDUP_REPORT.md` §3 extrapolates from the production run.

The audit's §3.4 note ("a 2-rank DDP run on GPUs 3,7 reproduces the task-0
strict recipe exactly") is about reproducing the *recipe*, which world size does
not affect — and `V8_RECIPE_EQUIVALENCE_REPORT.md` §1.1 now shows that at a fixed
effective batch it does not affect the *data stream* either. The sampler's
megabatch is `per_device x world_size x GA`, which **is** the effective batch,
because `TrainingArguments.n_gpu` is 1 per process under `torchrun`. One
optimizer step therefore consumes the same 64 samples at world 1, 2 and 4 alike;
only how they are dealt out changes.

An earlier revision of this document asserted the opposite — that the megabatch
was `n_gpu x effective_batch` and moved with the world size — on the strength of
an `n_gpu` factor that is 1 in every launch in this repository. That error is
corrected here and in §1.1 of the equivalence report, and this run's own metrics
are the evidence: the world-2 baseline logs 32 samples per rank per step, so a
step is 64 samples, and 39 743 / 64 = 621 steps — the step count production
actually took.

What world size *does* change is the cost of a step: the same 64 samples spread
over more devices, so a 4-GPU step is faster while the per-GPU rate is unchanged.
That is what the table above measures. It is not a claim that 2 GPUs and 4 GPUs
cost the same.

---

## 2. The phase table

**Steps 4-200 of the run** (197 of the 200; the first three are
model-adaptation warm-up and the run is complete, so there is no flush lag left
in these figures — an earlier revision of this table covered steps 4-72 only,
because that was all the running trace had flushed). Means, in seconds per
optimizer step, per rank:

| phase | rank 0 | rank 1 |
| --- | --- | --- |
| `window_wall_time` | 64.359 | 64.359 |
| `step_body_time` | 56.325 | 56.009 |
| `current_expert_forward_time` | 21.777 | 21.985 |
| `llm_forward_time` | 21.088 | 21.324 |
| `vision_forward_time` | 0.690 | 0.661 |
| `backward_time` | 34.547 | 34.024 |
| `routing_time` | 0.077 | 0.075 |
| `audit_time` | 2.523 | 2.509 |
| `allreduce_time` | 0.970 | 1.334 |
| `optimizer_time` | 0.028 | 0.027 |
| `metrics_time` | 0.014 | 0.014 |
| `data_wait_time` | 0.007 | 0.008 |
| `compose_linear_cpu_time` | **34.009** | **34.188** |
| `lora_expert_cpu_time` | 5.860 | 6.060 |
| `accounted_step_time` | 56.353 | 56.036 |
| `unaccounted_step_time` | 8.006 | 8.323 |
| `samples` / `micro_steps` | 32 / 32 | 32 / 32 |
| `tokens` (post-expansion) | 20 755 | 20 746 |
| `mean_seq_len` | 649 | 648 |
| `text_padding_ratio` | 0.000 | 0.000 |
| `peak_allocated_bytes` | 15.13 GiB | 15.13 GiB |
| `peak_reserved_bytes` | 17.43 GiB | 17.43 GiB |

`allreduce_time`'s mean is not a usable summary of `allreduce_time` — the
distribution is skewed by rare multi-second stalls, and §4.1 uses the medians
(0.066 s on rank 0, 1.262 s on rank 1) and the maxima instead.

Reproduce with:

```bash
python -m compose.experiments.baseline_summary \
  --trace rank0=.../training/profile_steps.rank0.jsonl \
  --trace rank1=.../training/profile_steps.rank1.jsonl \
  --util-tsv ... --util-gpus 3,7 --json <out>.json
```

(`compose.experiments.compare_profiles` reads the same traces and reports the
same phase means; `baseline_summary` is the wrapper that adds the throughput,
schedule-shape, VRAM and cross-rank statistics this document quotes.)

### 2.1 The trap in this table

`compose_linear_cpu_time` is the largest annotated phase — 34.0 s of a 64.4 s
step, **53 %**. The natural reading is "the ComposeLinear routing path is the
bottleneck". That reading is wrong, and the correction is the single most
important thing in this document.

`ComposeLinear`'s baseline path calls `torch.unique` and one `int(...item())` per
distinct expert, per layer. `.item()` is a **device synchronisation**: the CPU
blocks until every kernel queued so far has completed. So this timer measures the
CPU's *waiting*, not its work — and what it waits for is the GPU finishing the
backward pass it was enqueued behind. The 34.1 s is a large fraction of the step
because the GPU really is busy for that long; the CPU is simply blocked on it
rather than doing anything useful or harmful.

The proof is the S5 A/B: removing those synchronisations entirely
(`compose_selection_plan`) changes the step time by **1.003x, an effect of 0.19
standard errors** — nothing. A 53 %-of-step phase that can be eliminated for a
0 % speedup was never a cost. See `V8_ACCELERATION_IMPLEMENTATION.md` §5.1.

### 2.2 The corroborating signal

`backward_time` (34.5 s on rank 0, 34.0 s on rank 1) is very nearly equal to
`compose_linear_cpu_time` (34.0 s and 34.2 s), and both are ~53 % of the step.
`backward_time` is measured with sync-paired boundaries, so it is a real
wall-clock interval; the CPU timer inside it is measuring the same interval from
the other side. Two timers agreeing on the same 34 s is what a
blocked-CPU-backed-by-a-busy-GPU looks like.

### 2.3 What the phase columns cannot show

The brief asks for "model forwards per batch" and "expert evaluations per batch",
which is exactly the right instinct, because the real cost driver is a *count*
that no duration column reports:

| counter | per optimizer step | per micro-batch |
| --- | --- | --- |
| model forwards | 32 | 1 |
| `ComposeLinear` forwards | 14 336 | **448** |
| LoRA expert evaluations | 28 672 | **896** |

448 `ComposeLinear` calls per micro-batch is 224 layers x 2, the second 224 being
gradient-checkpoint recompute. Each of those is a separate small kernel launch on
a batch of **one sample**. At micro-batch 1 the step is **launch-bound**: the GPU
spends most of the step waiting for the CPU to enqueue thousands of tiny kernels,
which is precisely what 30 % utilization means.

This is the number that predicts everything downstream: it is invariant to which
phase you instrument, and it is why the acceleration that worked was the one that
amortised the launches over more work each (S6), not the one that removed CPU
work (S5).

---

## 3. The brief's required timers, one by one

Every timer the brief names, and what this runtime does with it. The last three
are the interesting ones: they are **structurally zero here**, and that is a
property of the frozen recipe, not a gap in the instrumentation.

| required | status |
| --- | --- |
| `data_wait_time` | measured, **0.007-0.008 s** (0.01 % of the step) |
| `query_load_time` | measured at startup, `query_load` in `profile_steps.jsonl.startup.json` (**23.9 s**, once per run) |
| `token_load_time` | folded into `data_wait_time`: tokenisation runs inside the DataLoader workers, under the loader's prefetch |
| `routing_time` | measured, 0.077 s |
| `historical_expert_forward_time` | measured, **exactly 0.0** — the training loop never runs a historical expert |
| `current_expert_forward_time` | measured, 21.8-22.0 s |
| `vision_forward_time` | measured, 0.66-0.69 s (1.1 % of the step) |
| `llm_forward_time` | measured, 21.1-21.3 s |
| `backward_time` | measured, 34.0-34.5 s |
| `allreduce_time` | measured: medians 0.066 s (rank 0) / 1.262 s (rank 1), means 0.970 / 1.334, maxima 12.860 / 6.507 — §4.1 |
| `optimizer_time` | measured, 0.027-0.028 s |
| `validation_time` | **structurally zero** — `eval_dataset=None`; this trainer runs no in-loop validation |
| `checkpoint_time` | **structurally zero** — `save_steps 1000000`; no mid-run checkpoint is written |
| `samples/sec`, `tokens/sec`, `optimizer_steps/sec` | derived, §1 |
| GPU utilization | sampled externally, §1 |
| peak / allocated VRAM | `peak_allocated_bytes`, `peak_reserved_bytes` |
| padding ratio | `text_padding_ratio` (and `padding_ratio` on older traces) |
| mean sequence length | `mean_seq_len` |
| effective / padded token counts | `tokens` (post-expansion, what the LLM processes); `text_valid_tokens` / `text_padded_tokens` (what the collator sees) |
| model forwards per batch | `model_forwards` |
| expert evaluations per batch | `lora_expert_evaluations` |

Two notes on precision, both of which the profiler is careful about:

- **`tokens` is the post-expansion count.** A single `<image>` placeholder becomes
  576 patch tokens inside the model, so the collator's text count and the sequence
  length the LLM actually processes differ by a factor of **8.8** — measured on
  one trace at a fixed 64-sample step: 4 707 text tokens against 41 507
  post-expansion, i.e. 73.5 against 648.5 per sample
  (`v8_s6fix_bs1_20260911`). The post-expansion count is the honest denominator
  because it is the sequence the attention actually spans; a text-only count would
  be 8.8x smaller and would not describe the compute.
  An earlier revision of this bullet quoted "4 711 ... 20 754 ... a factor of
  ~4.4", which divided a world-1 64-sample collator count by a world-2 32-sample
  *per-rank* post-expansion count. Both numbers were right; the ratio between them
  was not a ratio of anything.
- **This trace predates two of the profiler's columns.** `text_valid_tokens` and
  `text_padded_tokens` were added to `compose/train/profiler.py` at 20:19 on the
  day of the run; the run started at 19:07 and so its rows do not carry them —
  `compare_profiles` prints `--` for both, which is why the figures just above
  come from the world-1 ladder trace rather than from the baseline itself. The
  run only got more steps afterwards, not a new row format: it completed with all
  **200** steps profiled (`profiled_optimizer_steps: 200` in the sidecar) and the
  two columns are still absent from every row. The addition does not touch the
  timings, and `padding_ratio`, which reports the same quantity as
  `text_padded_tokens` would, is present in both revisions.
- **The step time is confirmed by a clock that is not the profiler's.** The
  sidecar records `train_wall = 12 885.90 s` for the 200-step loop, which is
  **64.43 s/step** — 0.1 % above the 64.359 s the sync-paired step timer reports
  over steps 4-200. The two agree while measuring different things: one is the
  process's own elapsed time around the training loop, the other is a sum of
  per-step intervals that excludes the three warm-up steps.
- **`text_padding_ratio` is 0.000** at micro-batch 1: every micro-batch holds one
  sample, so there is nothing to pad. This is why length bucketing (S8) is a
  no-op in the baseline and becomes a *cost* at wider micro-batches.

---

## 4. What is not the bottleneck

Each of these was a candidate on the brief's list. Each is listed with the number
that closed it, because "measured and it is not worth it" and "we did not look"
are different claims.

| candidate | measured | verdict |
| --- | --- | --- |
| DataLoader / CPU / IO (`num_workers`, `pin_memory`, NVMe) | `data_wait_time` **0.007 s** of 64.36 s = 0.01 % | already hidden by prefetch |
| token/label cache | same | tokenisation is inside the workers, under that prefetch |
| vision tower output cache | `vision_forward_time` **0.69 s** = 1.1 % | tens of GB per task to remove one percent |
| DDP `no_sync` | `allreduce_time` median 0.066 s of 64.36 s on rank 0 (0.1 %); one flattened all-reduce per step already | nothing to win, and what variance there is is straggler time, not transfer |
| logging / metrics | `metrics_time` 0.014 s = 0.02 % | nothing to win |
| checkpointing | no mid-run checkpoint at all | not applicable |
| **micro-batch width** | see `V8_SPEEDUP_REPORT.md` | **the one that paid** |

### 4.1 `allreduce_time` is a straggler signal, not a bandwidth cost

The mean is the wrong statistic here, and saying why is the whole finding. Over
the 197 analysed steps:

| | rank 0 | rank 1 |
| --- | --- | --- |
| median | **0.066 s** | **1.262 s** |
| mean | 0.970 s | 1.334 s |
| max | **12.860 s** | 6.507 s |

Rank 1's *typical* step spends 1.26 s in the collective while rank 0's typical
step spends 0.066 s — a **19.6x** asymmetry that no bandwidth-bound reduction can
produce, since both ranks move the same bytes. Rank 0's mean is 15x its own
median, because a handful of steps stall for ten seconds and drag it there; the
maxima disagree in the other direction (rank 0 12.9 s, rank 1 6.5 s).

Both facts point the same way: this is one rank **arriving late** at the
synchronisation point, and the other rank's timer recording the wait. The
correlations computed across the two ranks say so directly —
`corr(allreduce, window_wall)` is **+0.900 on rank 0** and **−0.086 on rank 1**
(a long collective on rank 0 predicts a long *step* there, while rank 1's
collective time carries no information about its step), and the across-rank
correlation of the two allreduce durations is **−0.356**, the negative sign
being the diagnostic for a *blocking* collective: when one side waits longer, the
other waits less. `unaccounted_step_time` (8.006 s on rank 0, 8.323 s on rank 1)
absorbs the same imbalance.

An earlier revision of this section reported the means as 0.221 s and 1.634 s, a
"7.4x asymmetry", with a 6.4 s single-step maximum — all three from the partial
steps 4-72 trace. On the full 197 steps the asymmetry is larger in the median
(19.6x) and smaller in the mean (1.4x), which is the difference between a
straggler signal and a cost. It is worth knowing, it belongs to the trainer's
data pipeline rather than to the accelerations this brief permits, and `no_sync`
would not touch it.

---

## 5. The stage budget outside the training loop

The profiler covers the training loop. The audit (`V8_CURRENT_RUNTIME_AUDIT.md`
§3.1) covers the stages around it, and one of them dwarfs everything above:

| stage | task-4 wall clock |
| --- | --- |
| training | 5 h 57 m 35 s |
| **RMS calibration sweep** | **4 h 07 m 15 s** |
| pruning + commit | 23 m 35 s |

All three come from the production run's own markers, regenerated by
`compose/experiments/stage_budget.py` (see `V8_CURRENT_RUNTIME_AUDIT.md` §3).

The RMS stage runs at `--batch-size 1` while its own docstring says the batch
size is unconstrained. Raising it is a pure scheduling change, and by these
numbers it is a **69 %**-of-the-training-stage item sitting outside the training
loop entirely. **It was not touched**, because the brief's rule 10 freezes the
RMS calibration rule and a summation-order change to kappa is exactly the kind of
quiet recipe move this exercise exists to prevent. It is recorded here as
measured, unclaimed work. See `V8_ACCELERATION_IMPLEMENTATION.md` §3.2.

---

## 6. Reproducing

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m compose.train.train_compose ... # the frozen pre-V8 command line, verbatim
  --profile_training True
```

The profiler is off by default and a no-op when off. It writes
`training/profile_steps.rank{N}.jsonl` (one row per optimizer step, flushed every
25) plus a `...startup.json` sidecar with the startup phase timings. Its design
rule is that it must not change what it measures: sync-paired `perf_counter()`
boundaries only at the outer phase edges, no synchronisation at all inside the
CPU probes, and sample/token counts taken from the collator on the CPU rather
than by calling `.item()` in the trainer.
