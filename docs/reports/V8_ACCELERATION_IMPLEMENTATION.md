# V8-Exact-Accelerated: implementation report

Branch `exp/v8-answer-supervised-multikey`. This document is the companion to
`V8_CURRENT_RUNTIME_AUDIT.md` (what the runtime actually does) and
`V8_SPEED_BASELINE.md` (what the frozen baseline actually costs). It describes
what was built to make it faster, what was refused and why, and — the part that
matters — how the claim "the numbers did not change" is *checked by code*
rather than asserted.

---

## 0. The one-sentence version

Every acceleration in this repository is a **scheduling** change behind a flag,
each flag is checked against a deny-list and a recipe-invariant gate at startup,
and the baseline still runs when every flag is off. Where a proposed change
could not be shown to preserve the numbers, it was not shipped.

---

## 1. The promise, and the gate that enforces it

The brief's closing rule is *"delete the repeated computation of frozen parts,
not the Current Expert's opportunity to be trained."* That is a statement about
**which work is redundant**, and the risk is that an implementation quietly
slides from one to the other. So the mode is built as a gate, not as a
convention: `compose/train/v8_flags.py`.

### 1.1 Three checks, all fail-closed

1. **The prohibited-acceleration deny-list** (`PROHIBITED_KEYS`). Each of the
   brief's twelve forbidden accelerations is rejected *by name*, at any nesting
   depth, before anything else happens. A config file cannot smuggle in
   `freeze_teacher: true` or `candidate_top_m: 8`; the loader raises
   `ProhibitedFlagError` naming the brief clause it would violate.

2. **Unknown flags are rejected, not ignored.** An `EXECUTION_FLAGS` typo is a
   startup error. Silently ignoring an unrecognised key is how a config and its
   run drift apart. The rejection is *actionable*, because the brief names
   capabilities this repository deliberately does not implement — the token/label
   cache, the historical-NLL cache, the vectorised expert evaluation — and a
   reader who copies those names into `flags:` would otherwise be told only that
   they are unknown. Instead, a flag that has a `not_applicable:` entry in the
   same file is answered with that entry's measured reason, and the brief's two
   renamed capabilities — `use_flash_attention` and `microbatch_autotune` — are
   pointed at the flags that do the job, `attn_implementation` and
   `micro_batch_size`. The check still refuses; it just no longer leaves the
   reader unable to tell a declined capability from a forgotten one
   (`_unknown_flag_message`, pinned by `tests/compose/test_v8_flags.py`).

3. **The recipe is re-derived from the resolved arguments** and compared with
   the frozen baseline (`assert_recipe_invariants`): learning rate `2e-4`,
   1 epoch, warmup `0.03`, cosine, weight decay `0`, seed `42`, `bf16`, `tf32`,
   `model_max_length 2048`, `remove_unused_columns False`, gradient checkpointing
   on, and — the one that catches everything else — **effective batch
   `micro_batch x accumulation x world_size == 64`**.

   The effective-batch check runs *after* the config and command line are
   merged, so a flag that is not on the list at all still cannot move the
   optimisation recipe.

### 1.2 The world-size detail, and why it was a real bug

`assert_recipe_invariants` computes the effective batch from the world size.
`TrainingArguments.world_size` is a property over the accelerator's distributed
state, which reads as `1` before the Trainer is built — and this check
deliberately runs before the Trainer is built. Left alone, a correct two-process
launch would have been refused with a nonsensical "effective batch 32", or (with
the precedence the other way) a genuinely single-process launch would have been
waved through.

Fixed by making the explicit value authoritative and sourcing it from the
launch: `_launch_world_size()` in `compose/train/train_compose.py` reads the
process group if it is already initialised and otherwise `WORLD_SIZE` from the
environment, which `torchrun` exports for every rank. Pinned by
`test_explicit_world_size_wins_over_the_attribute`.

### 1.3 The config owns the split, deliberately

`micro_batch_size` and `gradient_accumulation_steps` are applied *after*
`argparse`, so a command-line value would be silently overwritten. That is a
sharp edge — an arm intended to run at `2 / 16` would quietly run at `1 / 32`
and produce a duplicate measurement that looked like a result. The convention is
therefore: **the config owns the split, the command line does not touch it**,
and the A/B driver writes the split into a frozen per-arm copy of the config
with a `sha256` recorded next to the run (`logs/arm.env`).

---

## 2. How to run

Baseline, unchanged, no flags:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m compose.train.train_compose ... # exactly the pre-V8 command line
```

Accelerated:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m compose.train.train_compose ... \
  --compose_v8_config configs/v8_exact_accelerated.yaml
```

The loader prints what it resolved, so the run's own log records the flags:

```
[v8-exact-accelerated] /path/configs/v8_exact_accelerated.yaml -> {"flags": {...}, "invariants": {...}}
```

Config cache handling is existence-detecting and fail-loud: a *missing* query
tensor logs and falls back to the JSON cache, while one that is *present* but
stale (contract hash), corrupted (value fingerprint) or short of the dataset's
sample ids raises. A cache that is wrong must never be consumed silently.

### 2.1 The flag surface

| Flag | Sets | Step |
| --- | --- | --- |
| `cache_queries` | `--compose_v7_query_tensor` | S1 |
| `compose_selection_plan` | shared per-micro-step selection decomposition | S5 |
| `micro_batch_size` | `--per_device_train_batch_size` | S6 |
| `gradient_accumulation_steps` | `--gradient_accumulation_steps` | S6 |
| `length_bucketing` | `--group_by_modality_length` | S8 |
| `attn_implementation` | `--compose_attn_implementation` | kernel |
| `profile_training` / `profile_sync` / `profile_flush_every` | profiler | P0 |

Profiling is deliberately **not** in the shipped config: it is a diagnostic, and
a config that forced it on or off would fight the command line.

---

## 3. Deviations from the brief, reported separately

The brief's rule 10 allows a frozen knob to move only as *a documented bug fix,
reported separately*. Two things are reported here rather than silently fixed.

### 3.1 The profiler's `accounted_step_time` was wrong, and traces are re-derived

`routing_time`, `allreduce_time` and `metrics_time` are measured *inside*
`step_body_time`, but the first version of `end_step` added all of them into
`accounted_step_time` anyway. The total over-counted by ~0.16 s/step and
`unaccounted_step_time` inherited the error.

The 200-step baseline had already started when this was found, and restarting a
3.5-hour measurement to fix a derived column is not worth it — so the fix is on
the *reader*: `compose.experiments.compare_profiles.normalize()` recomputes
`accounted_step_time` and `unaccounted_step_time` from the raw phase columns,
which were always correct, and back-fills `valid_tokens` for traces written
before that column existed. Traces from before and after the fix are therefore
comparable field for field, and every number in the reports is produced by that
one reader.

### 3.2 The RMS calibration stage is the largest remaining wall-clock item, and it is out of scope

The audit's stage budget (§3.1 there) shows that on task 4 the S4 RMS stage ran
**4 h 07 m** against the S3 training stage's 6 h 02 m. It is a validation-sweep
stage, not a training loop, and its own call site uses `--batch-size 1` while
its docstring says "the batch size is unconstrained" (`compose/lora/rms.py`).

By the brief's arithmetic that is ~68 % of the training stage's duration sitting
in a stage nobody profiled. Across the whole six-task run it is **20.31 h of
59.99 h — 34 %**, more than the pruning stage (7.59 h) and only exceeded by
training itself (32.10 h). On task 5 the RMS stage **outlasts that task's
training** outright: 7.74 h against 5.67 h on the trainer's clock (and more than
that once load and startup are counted, so the comparison holds either way).

Its cost also grows faster than the pool it calibrates. Every task scores the
same 256 samples over the same 224 `ComposeLinear` layers, so the expert-forward
count rises exactly linearly with the pool; the wall clock does not:

| task | experts | RMS wall clock | per sample | per sample per expert |
| --- | --- | --- | --- | --- |
| task1 | 8 | 0.30 h | 4.27 s | 0.53 s |
| task2 | 12 | 0.94 h | 13.27 s | 1.11 s |
| task3 | 16 | 2.21 h | 31.13 s | 1.95 s |
| task4 | 20 | 4.12 h | 57.95 s | 2.90 s |
| task5 | 24 | 7.74 h | 108.78 s | 4.53 s |

(The per-sample column is `rms_s / 256` on the same generated budget as §3 of
`V8_SPEEDUP_REPORT.md` — `docs/reports/data/stage_budget_formal_20260903.json` —
so it cannot drift from it.)

**A 3x rise in expert count costs 25x per sample** (3.00x to 25.48x). The hook materialises every
expert delta on a batch of one and holds them live per layer
(`compose/lora/rms.py:151-177`; its own comment at line 197 notes the allocator
pressure large pools create), so the per-forward cost itself grows with the pool.
The stage is *launch-bound and pool-superlinear at the same time* — the two
pathologies this exercise removed from training, still present here.

**It was not touched.** The brief's rule 10 freezes "RMS calibration rule", and
a summation-order change to kappa is exactly the kind of quiet recipe move this
whole exercise exists to prevent. It is reported here as measured, unclaimed
work with the prohibition named, which is what rule 10 asks for. The same
applies to the teacher sweep and the pruning stage.

That last column is the reason this is reported and not fixed. `per sample per
expert` rises monotonically, 0.53 s -> 4.53 s, so a component of the cost is
superlinear in the pool — and widening the batch does not remove a superlinear
term. Batching would attack the launch-bound half only; the other half would
survive it, and nothing here separates the two. The split is unknown, so no
speedup is claimed, and the prohibition stands on its own anyway.

### 3.3 The key loss was a sum and the answer loss a mean — a fix the brief's rule 10 permits, and the bug that made S6 look better than it is

**This is the one change in the exercise that moves the objective, so it is
separated from every speedup claim.** It is a bug fix under rule 10's carve-out,
and it changes what the trainer optimises at any micro-batch wider than one. It
does **not** change what the frozen baseline optimises, which is why the
frozen baseline remains a valid reference.

**What was wrong.** `V7ComposeTrainer.compute_loss` has always set
`inputs["v7_sum_per_sample_loss"] = True`, asking the model for a *sum* of
per-sample token-mean losses rather than a token-weighted batch mean. Two defects
composed to defeat it:

1. **The flag never reached a branch.** `ComposeLlavaForCausalLM` inherits
   `LlamaForCausalLM`, not `LlavaLlamaForCausalLM` — the class that owns the
   `v7_sum_per_sample_loss` branch. The compose forward ends in `**kwargs`, so
   the flag was accepted and dropped. The answer term stayed a token-weighted
   batch mean in *both* model classes' behaviour, because the one class V8
   instantiates had no branch to reach.
2. **The key term was a per-micro-batch sum against it.**
   `key_loss = per_sample.sum() if queries.shape[0] > 1 else mean_key_loss`
   (`compose/v7/hf_trainer.py` at HEAD, line 409). At width one the two branches
   coincide; wider, the key term is the un-normalised sum.

Each term is then divided by `gradient_accumulation_steps` inside
`Trainer.training_step`. Widening the micro-batch shrinks GA by the same factor,
so a mean survives that division and a sum does not: at micro 4 the key term
reached the optimizer **4x** too large, an effective `lambda_key` of 0.4 where
the config says 0.1.

**How it was found, and why it mattered here.** The S6 speedup is a *widening*,
so its two arms are precisely the arms this defect moves apart. The equivalence
gate (`compose.experiments.compare_recipe --mode window`) reported, on the real
traces of the pre-fix sweep:

| comparison | key objective ratio | row-width ratio | `scales_with_row_width` | answer ratio | routing |
| --- | --- | --- | --- | --- | --- |
| bs1_ga64 vs bs2_ga32 | **2.0014** | 2.00 | True | 1.004 | **differs on window 5** |
| bs2_ga32 vs bs4_ga16 | **2.0000** | 2.00 | True | 1.003 | identical |
| bs2_ga32 vs bs8_ga8 | **4.0000** | 4.00 | True | 1.003 | identical |

Sample sets identical in all three — the windows were right and the *objective*
moved, exactly in proportion to the width, in the key term only. The gate names
that signature explicitly rather than reporting a generic loss mismatch, because
"the objective scales with the row width" and "the two arms disagree" call for
different responses. Source: `data/prefix_keywidth_bs{1,2}_vs_bs{2,4,8}.json`.

Two details in that table are worth reading rather than skimming. The first row's
key ratio is **2.0014**, not 2.0000, and it is the only row whose routing is not
identical: on window 5 a single sample (`v7_t4_train_451`, route `NewNew`) has its
second current expert flip from 16 to 17. Both are consequences of the same
thing — a key gradient 2x too large is *applied*, so by the sixth window the
router has genuinely drifted, and one sample's near-tie resolves differently. The
other two rows keep identical routing because they compare against `bs2_ga32`
rather than the width-1 arm, so the divergence has already happened on both sides
before the comparison starts.

That divergence is the reason the post-fix result is stated as *exactly* identical
routing rather than merely a small loss difference. The fix does not shrink the
bug's effect; it removes it — 0 differing routing decisions and a key objective at
1e-9, where the pre-fix gate could not even hold the data stream together.

Had the gate summed the window instead of averaging it, it would have reported
the opposite of the truth in both directions: a summed key term is *invariant*
under windowing, and a meaned answer term *halves* when the window widens. That
is why `_window_objective` is the mean — see the equivalence report §1.2.

**The fix**, three edits, each a no-op at micro-batch one:

| file | change |
| --- | --- |
| `llava/model/language_model/llava_llama.py` | the inline per-sample branch becomes the module-level `sum_of_per_sample_token_means`, shared deliberately, because the compose class is a sibling and not a subclass |
| `compose/model/compose_llava.py` | `forward` reads the flag and applies the same helper, so the compose class now honours the contract its trainer asks for |
| `compose/v7/hf_trainer.py` | the model now returns a sum above width one, so the trainer divides by the width, and `key_loss` becomes `per_sample.mean()` — putting both terms on a per-micro-batch **mean** footing |

The objective becomes `Ā + 0.1·K̄` at every width, which is the width-1 objective.
At width one the model's branch does not fire (`labels.shape[0] > 1`) and the
division is by one, so a width-1 run evaluates the same quantity as before, up to
floating-point summation order (`mean of values` versus `sum, then divide` —
the same number to rounding, which is the standard this exercise holds every
comparison to).

**The consequence for this report, stated plainly: the pre-fix speedup was
measured on arms that carried the defect** — 2.24x at the time, and the figure
§7.1 and `V8_SPEEDUP_REPORT.md` now quote is the re-run on the fixed tree, 2.26x.
The pre-fix arms are preserved as evidence of the defect
(`v8_s6_*_20260911`, `v8_armC_prefix_buggy_20260911`) and are not shipped arms.

One traceability note about the second of those: its `config.sha256` records
`ba22f18e…` for `configs/v8_exact_accelerated.yaml`, the shipped arm C recorded
`b60429e7…`, and the file on disk now hashes to `6bac1253…`. Every difference is
comments — the briefing text under `not_applicable`, the brief-name
cross-references added to `micro_batch_size` and `attn_implementation`, and two
measurements in comments restated from the finished 200-step baseline rather than
the partial trace — and none touched anything under `flags:`. Checked against the
run's own copy, which the launcher writes beside it:

| flag | pre-fix arm's copy | shipped file |
| --- | --- | --- |
| `compose_selection_plan` | `true` | `true` |
| `micro_batch_size` / `effective_batch` | `4` / `64` | `4` / `64` |
| `length_bucketing` | `true` | `true` |
| `attn_implementation` | `""` | `""` |
| `cache_queries` | `…/query_cache/task4/train` | `null` |

The one difference is the per-arm fill-in the launcher performs by design — the
shipped file leaves `cache_queries` null and each arm's copy resolves it to that
run's cache directory, which is the whole reason the A/B driver writes a frozen
per-arm copy. It is recorded rather than left to be discovered, because a hash
that does not match its file is exactly the thing this repository's cache
contract refuses to consume silently; here it is explainable, and the shipped
arm C records the hash it actually used.

This paragraph originally predicted that the fix would make the widening "less
free than it looked" and that the shipped multiple would be **less** than 2.24x.
That prediction was wrong, and the measurement is the thing to keep: the fixed
ladder gives **2.26x** at the shipped point (126.97 -> 56.23 s over six steps),
against the pre-fix 2.24x. The reason is that the fix's extra work is a second
cross-entropy over the LM head on rows above width one, and at micro 4 that is
four rows rather than 64 micro-batches of one — it is real but it is not where
the step time lives, because the step is launch-bound either way. A prediction
that the architecture of the change made plausible lost to a six-step
measurement by 0.8 %, which is the margin §2 of the speedup report already
declares its noise floor. The corrected numbers replace the pre-fix ones in
§7.1 and in the speedup report.

**Why the frozen baseline is still the right reference.** Every run that
produced the frozen baseline — and every production V7/V8 run — trained at
micro-batch 1, where the defect is invisible and the fix is inert. The baseline
arm of the A/B therefore measures the same recipe it always did, and the fix
moves only the accelerated arm, onto the objective the baseline was already
optimising. That is the correct direction for a bug fix to move.

---

## 4. Verification inventory

Nothing below is a claim about intent; each line is a command that either passes
or does not.

| What | Where | Result |
| --- | --- | --- |
| Full test suite | `pytest tests/` | **710 passed**, 0 failed, 65 subtests passed, 74.77 s |
| S5 fast/slow path bit-identity | `tests/compose/test_v8_exact_accelerated.py` | forward *and* backward `torch.equal` |
| Query cache vs JSON cache | `compose.experiments.verify_query_tensor` | 12/12 splits MATCH, 0 differing elements |
| Flag gate | `tests/compose/test_v8_flags.py` | 27 tests: deny-list, unknown keys, recipe guards, accumulation derivation, and the actionable rejection message |
| Metrics path convention | `tests/compose/test_metrics_path_convention.py` | 4 tests: writer and reader agree at world 1 *and* >1 |
| A/B comparator | `tests/compose/test_compare_recipe.py` | 23 tests: reordered stream must MISMATCH; window multiset must survive a re-split; window mode reads per-sample legacy fields, reports the window size, and takes a directory of per-rank files |
| Sampler window invariance | `compose.experiments.sampler_window_invariance` | real 39 743-sample task-4 split: 4 positive splits identical window-for-window over the 620 comparable windows (0 diverging; the two world-1 sides take one extra 63-sample tail step) *and* the negative control 0/310 |
| End-to-end recipe A/B | `compose.experiments.compare_recipe --mode window` | `V8_RECIPE_EQUIVALENCE_REPORT.md` §3.6: 100 windows, 6 400 distinct samples, routing identical 0/100, `key_loss` max rel 6.17e-07 |
| Residual characterisation | `tests/compose/test_window_residual.py` | 12 tests, one per edge that decides a verdict: a synthetic residual returns its own mean and sd; a fixed floor must **not** grow the absolute column; a compounding one **must**; a half-written window is dropped |

Two of these deserve a note on *why they are shaped that way*:

- **The comparator is tested at the edges that matter.** A comparator that
  reports "equivalent" for everything would satisfy the brief's letter and
  prove nothing, so `test_reordered_stream_is_a_mismatch_even_with_identical_losses`
  pins the case a broken sampler would produce — every loss identical, the data
  stream reversed — and requires it to be called a MISMATCH. Its sibling
  requires a pure summation-order change to be called EQUIVALENT rather than
  MISMATCH, because a gate that fails everything is as useless as one that fails
  nothing.

- **The query verification is per sample id, not per row.** `features/train.json`
  and `queries.pt` need not agree on ordering, so the check aligns by
  `sample_id`, reports `id_order_identical` separately, and refuses to compare
  row `i` with row `i`. `max_abs_diff = 0.0` with `id_order_identical = true` is
  a much stronger statement than either alone.

---

## 5. What was refused, with the measurement that refused it

Each of these was on the brief's list as a candidate. Each is recorded with the
number that closed it, because "we chose not to" is not the same as "measured
and it does not pay".

| Candidate | Measured | Verdict |
| --- | --- | --- |
| **S2** token/label cache | `data_wait_time` **0.007 s** per optimizer step (world 2; 0.002 s at world 1) | Refused: 0.01 % of the step. Tokenisation already happens inside DataLoader workers, hidden by prefetch; a cache would add a fingerprint contract and a disk format for nothing. |
| **S3** historical-NLL cache | Training step evaluates **only** the current experts | Not applicable. V8-A is an offline one-shot teacher; the training loop never computes a historical NLL, so there is no repeated computation to delete. The redundancy is *inside* the teacher sweep and is a different problem. |
| **S4** multi-GPU frozen-oracle generation | Same as S3 | Not applicable to the training loop. Would matter for the teacher sweep's wall clock, which is out of this brief's scope and interacts with the frozen candidate lifecycle. |
| **S5 (brief's)** vectorised expert eval | Expert evaluation is not the bottleneck | Not applicable as written; the step's real cost at micro-batch 1 is launch-bound tiny kernels. Re-scoped to the per-layer selection decomposition, which *was* measurable. |
| **full visual-token disk cache** | Vision tower **0.69 s** of a 64.36 s step = **1.1 %** | Refused: tens of GB per task to remove one percent. The brief made this conditional on benchmarking, and this is the benchmark saying no. |
| **S8** length bucketing | `padding_ratio` **0.0** at micro-batch 1 | Already in the frozen baseline (`group_by_modality_length True`); at micro-batch 1 there is nothing to pad. Kept as an explicit flag so an un-bucketed control remains possible. |
| **`use_flash_attention`** | Not a `TrainingArguments` field in transformers 4.33.3 | Refused as named (verified by introspection). Replaced by `attn_implementation`, which is actually consumed in `_build_model`; changing the attention kernel genuinely changes numerics, so it stays off until measured. |

### 5.1 The S5 result that corrected the plan

The per-layer selection decomposition removes ~3 device synchronisations per
layer — **448 `ComposeLinear` calls per micro-step** (224 injected layers, twice
over, because gradient-checkpoint recompute replays the forward), which is
**14 336 per optimizer step** at GA 32 — and `compose_linear_cpu_time` was
**59.8 %** of the step body at the time. It was reasonable to expect a large
win. (An earlier revision said "448 per optimizer step", which is the per
micro-step figure; the trace's `compose_layers_forward` column reports the
14 336.)

**The A/B measured nothing.** Two runs at micro-batch 1 / GA 64 / world 1, six
optimizer steps each, skip 2:

| arm | s/step | sd | min | max |
| --- | --- | --- | --- | --- |
| S5 off | 125.913 | 0.838 | 124.316 | 127.019 |
| S5 on | 126.286 | 4.695 | 121.897 | 135.633 |

The difference is **+0.372 s/step, 1.003x**, against a standard error of
**1.947 s/step** — an effect of **0.19 SE**. There is no measurable speedup; an
earlier informal reading of "~3 %" was noise and has been withdrawn. (Source:
`docs/reports/data/s5_ab_profiles.json`.)

The reconciliation is the important part: at micro-batch 1 the step is
**launch-bound**, and the annotated CPU timer was absorbing GPU wait time at each
`.item()` synchronisation rather than measuring CPU work. Those syncs cost what
they cost — but the GPU was going to be busy for that wall-clock interval
anyway, so removing them frees the CPU and not the step. A CPU-shaped timer on a
GPU-shaped bottleneck is the trap this measurement walks into, and profiling
with `--profile_sync False` before trusting `compose_linear_cpu_time` is what
avoids it.

What S5 *is* worth is that it is **bit-identical** (384/384 micro-steps,
`docs/reports/data/s5_ab_recipe.json`) and structurally simpler, so it is kept
and shipped at zero cost. It is **not** claimed as a speedup, and the plan moved
to the levers that can actually move a launch-bound step — which the S6 sweep
then found, an order of magnitude larger.

A second correction belongs here: the S6 arms were run with S5 **on**, while the
S6 base arm was the S5-**off** run. With S5 now measured at 1.003x, that
conflation is smaller than the measurement error it would introduce, so the S6
ratios in §7 stand as measured, with no correction applied.

---

## 6. Measurement tooling built for this work

Every claim in these reports is produced by a script in the repository, not by
hand.

| Tool | Answers |
| --- | --- |
| `compose/train/profiler.py` | One JSONL row per optimizer step: the brief's phase list, micro-batch truth for padding, peak VRAM, step/sample/token rates. Off by default and no-op when off. |
| `compose/experiments/compare_profiles.py` | Per-step means and ratios across arms; normalises pre-fix traces (§3.1). |
| `compose/experiments/compare_recipe.py` | Field-by-field recipe A/B on the trained numbers, with `BIT_IDENTICAL` / `EQUIVALENT` / `MISMATCH`. |
| `compose/experiments/window_residual.py` | What a `MISMATCH` *means*: whether the residual is a stationary rounding floor — unbiased (t), plateauing in the absolute column, worst window under σ·√(2 ln n) — or a trajectory that is diverging. Projects the tolerance to the brief's 500- and 1000-step rungs. |
| `compose/experiments/baseline_summary.py` | Phase means, allreduce skew, throughput and VRAM for one or more profiler traces, at any world size; the source of every number in `V8_SPEED_BASELINE.md`. |
| `compose/experiments/autotune_microbatch.py` | Ranks micro-batch/accumulation splits by measured median step time; **refuses** any candidate whose product leaves the effective batch at 64. |
| `compose/experiments/verify_query_tensor.py` | Per-sample-id equivalence between the JSON and tensor query caches. |

The profiler's own design rule is worth stating: it is a measurement, so it must
not change what it measures. `timed` boundaries are `perf_counter()` pairs
around existing calls with a synchronisation at the outer boundary only; `cpu`
probes time inner loops with no synchronisation at all; and the sample/token
counts come from the collator on the CPU, because calling `.item()` in the
trainer to get them would add exactly the synchronisation the profiler exists to
attribute.

---

## 7. Ablation results, S0–S11

The brief prescribes an ordered ablation and asks that any step which speeds
things up but degrades recipe equivalence be rolled back. The table below is that
list, each row with the measurement that decided it. Rows marked *not applicable*
were not skipped for convenience: the brief also forbids the twelve accelerations
that would have made them pay, and in each case the reason is the same — the
repeated computation the step targets **does not exist in this runtime**
(`V8_CURRENT_RUNTIME_AUDIT.md` §6).

| # | Step | Measurement | Result | Decision |
| --- | --- | --- | --- | --- |
| **S0** | Frozen baseline | 200-step trace, task 4, world 2, micro 1 / GA 32 | **64.36 s** per optimizer step; **0.497 samples/s per GPU** | reference |
| **S1** | Query cache normalisation | Real 1.30 GB (1.21 GiB) `features/train.json` vs `queries.pt` | **25.31 s → 0.78 s** (32x), 3.54 GB RSS avoided; values bit-identical per sample id over 12/12 splits | **shipped** |
| **S2** | Token/label/length cache | `data_wait_time` **0.007 s** per optimizer step | 0.01 % of the step | refused |
| **S3** | Historical-NLL cache | Training loop evaluates current experts only | no repeated NLL exists to delete | not applicable |
| **S4** | Multi-GPU frozen-oracle generation | same | the teacher is a one-shot offline stage, outside the training loop | not applicable |
| **S5** | Expert-evaluation vectorisation | Re-scoped to the per-layer selection decomposition; A/B at micro 1 / GA 64, 6 steps | **1.003x**, effect **0.19 SE** — indistinguishable from zero | **shipped at zero cost**, *not* claimed as a speedup |
| **S6** | Micro-batch / accumulation split | Sweep of 4 splits at fixed effective batch 64, world 1 | micro 4: **0.443x** (2.26x). micro 8: 0.335x but +3.9 GiB | **shipped at micro 4** |
| **S7** | Gradient checkpointing off | micro 1 / GA 64, world 1, profiler traces, skip 2 | 125.91 s → **124.85 s (0.992x, +0.8 %)** — same-shape but *not* paired runs — and peak memory **identical to 3 decimals** (15.128 GiB alloc, 15.527 GiB reserved) | refused, on the memory equality |
| **S8** | Length bucketing | Already on in the frozen baseline; `padding_ratio` 0.000 at micro 1 | 0.000 → 0.072 at micro 4 (a *cost* of S6, not a win) | kept as an explicit flag |
| **S9** | DDP `no_sync` | `allreduce_time` 0.00006 s at world 1; at world 2 median **0.066 s** of a 64.36 s step on rank 0 (0.1 %) | nothing to win; one flattened all-reduce already, and the rest is straggler time (baseline report §4.1) | not applicable |
| **S10** | DataLoader / mmap / NVMe | `data_wait_time` **0.007 s** of a 64.36 s step | prefetch already hides it | refused |
| **S11** | Checkpoint / logging | `metrics_time` 0.014 s; `save_steps 1000000` → no mid-run checkpoint | 0.02 % | refused |

### 7.1 S6 in detail, and one caveat stated plainly

These are the **fixed-code** arms — the same runs the equivalence report gates, so
the timing and the equivalence evidence come from one generation of the code.
They supersede an identically-shaped pre-fix sweep; where the two overlap they
agree (micro 4 ratio 0.443 vs 0.446, micro 8 0.335 vs 0.334), and the micro-2 arm
moved from 0.699 to 0.645 because the earlier reading carried an outlier.

| arm | s/step | ratio | peak alloc | text padded tokens | padding ratio |
| --- | --- | --- | --- | --- | --- |
| micro 1 / GA 64 | 126.97 | 1.000x | 15.13 GiB | 4 707 | 0.0000 |
| micro 2 / GA 32 | 81.95 | 0.645x | 15.68 GiB | 4 871 | 0.0336 |
| **micro 4 / GA 16** | **56.23** | **0.443x** | **16.80 GiB** | 5 063 | 0.0702 |
| micro 8 / GA 8 | 42.53 | 0.335x | 19.00 GiB | 5 201 | 0.0951 |

**Every column here is a six-step statistic** — s/step and the token counts are
means over the six steps, peak alloc is the maximum. `V8_RECIPE_EQUIVALENCE_REPORT.md`
§3.1 quotes step 1's padding ratios instead (0.0312 / 0.0670 / 0.0887), because
they are paired there with a single step's token count; that is the same
measurement over a different subset, not a disagreement.

The mean valid token count is **4 707 in all four arms** (per-step 4 669–4 746,
identical in every arm, at every width) — the arms differ only in the padding
they add, so the padding column is the entire cost of the width change and nothing
else is hiding in it.

micro 8 is 1.32x faster again and was **not** shipped, on memory rather than on
equivalence: it peaks at **19.00 GiB allocated** (+3.87 over the baseline, +2.20
over the shipped point) and **21.82 GiB reserved** on a 24 GiB card, under 2.2 GiB
of headroom, with the padding ratio 35 % above micro 4's (0.0951 vs 0.0702). The
equivalence residual is *not* the reason — at width 8 it is smaller than at width
4 (§3.3 of the equivalence report). The shipped point is micro 4, which buys
2.26x for +1.67 GiB of peak allocation (16.80 vs 15.13).

Every arm runs **6 optimizer steps and all six are measured** — these arms exist
to time a step, not to train, and at six steps the spread is far below the
difference between any two arms. It was measured rather than assumed:

| arm | mean s/step | per-step sd | SE of mean | ratio (mean) | ratio (median) | SE of ratio |
| --- | --- | --- | --- | --- | --- | --- |
| micro 1 / GA 64 | 126.97 | 1.55 % | 0.80 s | 1.000 | 1.000 | — |
| micro 2 / GA 32 | 81.95 | 1.53 % | 0.51 s | 0.645 | 0.645 | ± 0.004 |
| **micro 4 / GA 16** | **56.23** | 2.47 % | 0.57 s | **0.443** | **0.444** | **± 0.004** |
| micro 8 / GA 8 | 42.53 | 3.05 % | 0.53 s | 0.335 | 0.331 | ± 0.004 |

Two things this settles. The **shipped point is tight**: 0.443 ± 0.004, and the
median estimator agrees with the mean to 0.2 %, so no single step is carrying it.
And **no arm is an outlier any more** — the sd range is 1.5-3.1 % across all four,
where the pre-fix micro-2 arm showed 6.06 % with one step at 95.8 s against
83.8-87.3 for its siblings. That is the useful reading of the change: a 6 %-sd
arm with one slow step is what GPU contention looks like, and the fixed ladder,
run under the same shared-GPU conditions, no longer produces one.

**Every arm processes exactly 64 samples per optimizer step** (`samples_mean 64.0`
in all four), so these are ratios of the same work done four ways, with no
normalisation and no extrapolation.

**Why the arms exited `rc=1`.** The three non-base arms failed *after*
`trainer.train()` returned, in the post-training metrics load:
`FileNotFoundError: train_steps.rank0.jsonl` — the ranked-metrics-path bug fixed
later the same evening (the fix is what makes the S7 arm exit `rc=0`). The
quantity these arms report is `window_wall_time`, written by the profiler
*during* training and unaffected by anything after it. The teardown failure cost
these arms their audit sidecars and nothing else. It is noted because `rc=1` on
an experiment that produced a number is exactly the kind of thing that should
never be discovered later by someone reading a log.

**The caveat: the sweep ran on a 512-sample smoke split, not the real 39 743-sample
one**, because four arms have to fit in a GPU-hour and each arm pays a 75 s model
build. Two things make the numbers transferable, and one measurement closes the
question:

- The smoke split is the head of the real split (`v7_t4_train_0`…), and its mean
  sequence length is **648.7-648.9** against the baseline's **648.6 / 648.3**
  (rank 0 / rank 1) on the real split — the per-step work is the same work.
- The smoke split's micro-1 arm measures **0.509 samples/s per GPU** over the
  same 3-step warm-up window this document uses elsewhere, or **0.504** over the
  6-step window the ladder table quotes; the real 200-step baseline measures
  **0.497**. The two agree to **1.4-2.4 %** depending on which window is taken.
  An arm that reproduces the real baseline's per-GPU throughput to a couple of
  percent is measuring the same thing — and the range rather than a single figure
  is the honest report, because these are 3-to-6-step windows on a 512-sample
  smoke split and the two conventions answer slightly different questions.
  (An earlier revision said "0.5096 against 0.5016, ratio 1.016x": the first was
  a third window again, and the second was a partial baseline trace, superseded
  by the finished run's 0.497.)

The question is closed directly by the A/B pair itself — `v8_baseline_task4_200step_20260911`
(micro 1 / GA 32) against `v8_armC_world2_task4_20260911` (micro 4 / GA 8) — both
on the real 39 743-sample split, both at world 2, so the *only* difference
between them is the split. That is a stronger closing than a world-1 pair would
have been, since it measures the ratio at the world size the baseline actually
used; see `V8_SPEEDUP_REPORT.md` §2.

**The S7 correction is worth recording.** An in-flight read of that arm's metrics
summed `training_step_sec` over a *half-written* optimizer step and produced
"118.9 s, +5.5 %". Re-measured over complete steps only, GC-off is **124.85 s
against 125.91 s** — 0.8 %, inside the same 0.8 s/step run-to-run spread S5
showed. Both figures are the profiler's `window_wall_time` mean with the tool's
default `--skip-steps 2`, which is what leaves 4 and 6 steps respectively; the
comparison is reproducible with
`compare_profiles --arm GC_on=... --arm GC_off=... --skip-steps 2` and is saved
at `data/s7_gcoff_vs_gcon.json`.

One thing about that pair is worth stating rather than leaving to be discovered:
**the GC-on side is not a purpose-built S7 arm.** It is `v8_s5_ab_off`, the S5
A/B's own baseline run — there is no `v8_s7_gcaon` directory, and the GC-off arm
was never run beside a GC-on twin. The two are the same shape (micro 1 / GA 64 /
world 1, same frozen recipe) and were run on the same GPU, but an hour apart
(their traces were written at 19:34 and 20:31 on the shared card), so they do not
sample a common set of contending neighbours. That is why the timing difference
of 0.8 % must not be read as a tight measurement — and also why it does not need
to be, because **the S7 decision does not rest on it**:

| | GC on (`v8_s5_ab_off`) | GC off (`v8_s7_gcoff_bs1`) |
| --- | --- | --- |
| window wall (skip 2) | 125.91 s | 124.85 s |
| peak **allocated** | 15.128 GiB | 15.128 GiB |
| peak **reserved** | 15.527 GiB | 15.527 GiB |

The memory columns are identical to three decimals. Gradient checkpointing off
frees **nothing** on this runtime at micro 1 — the dominant terms are the 7B
weights, their optimizer state and `ComposeLinear`'s materialised expert deltas,
none of which recompute touches. The premise that motivated the experiment
(GC-off buying headroom for a wider micro-batch) is refuted by construction, and
that refutation is a memory equality rather than a timing comparison, so it does
not inherit the pair's weakness. Reading a running experiment's partial output as
a result is the error; this is what it cost.

### 7.2 What the S6 win actually is

The step at micro 1 is **launch-bound**: 64 samples means 64 micro-batches means
thousands of tiny kernels per optimizer step, and the GPU spends much of the step
waiting on the CPU to enqueue them. Widening the micro-batch amortises the launch
cost over 4x more work per kernel, and every phase scales with it — the S6 sweep
shows `current_expert_forward_time` at 0.43x, `backward_time` at 0.49x,
`llm_forward_time` at 0.43x, all tracking the wall clock rather than one phase
dominating.

That is also why the *earlier* plan was wrong. `compose_linear_cpu_time` — the
largest single annotated phase, 60 % of the step body — turned out to be a CPU
timer absorbing GPU wait at each `.item()` sync (S5.1 above). The bottleneck was
never the phase that looked largest; it was the *number of kernels*, which no
phase column reports directly.
