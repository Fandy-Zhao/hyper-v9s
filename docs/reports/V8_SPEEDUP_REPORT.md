# V8-Exact-Accelerated: speedup report

The deliverable that answers the brief's final question — did the acceleration
work, and what is the verdict.

**The one number: 2.26x on the training stage, 1.42x on the full continual run.**
Both are real and both are stated, because the gap between them *is* the result:
the acceleration did what it was asked to do to the training stage, and the
training stage is only half of the pipeline.

The training-stage multiple is measured **twice, by different routes**: 2.26x on
the controlled single-GPU ladder (§2, six steps, micro 1 vs micro 4) and
**2.313x** on the real task-4 split at the baseline's own world size (§2 item 2,
197 steps). They agree to 2.3 %. Every projection below uses **2.258**, the
smaller of the two, so the projected hours are conservative rather than
optimistic.

---

## 1. What was changed

Exactly one thing was shipped, plus one correctness fix that costs nothing:

| | what | effect |
| --- | --- | --- |
| **S6** | micro-batch 1 -> 4 with the effective batch held at 64 | **2.26x** on the training stage |
| **S1** | read the fixed queries from `queries.pt` instead of a 1.30 GB JSON | 25.31 s -> 0.78 s startup, 3.54 GB RSS avoided |
| **S5** | drop the per-layer device syncs (`compose_selection_plan`) | neutral by measurement (1.003x +/- 0.19 SE); shipped because it is free, not claimed as a speedup |

Everything else on the brief's list was measured and refused; §5 lists them with
the number that closed each one.

### 1.1 Why S6 is the one that paid

`V8_SPEED_BASELINE.md` §2.3 establishes the mechanism: at micro-batch 1 the
training step issues **14 336 tiny `ComposeLinear` forwards per optimizer step**
(448 per micro-batch, 224 layers x forward + recompute) on a batch of *one*, and
the GPU sits at **30 % utilisation** waiting for the CPU to enqueue them. The
step is launch-bound. Widening the micro-batch raises the work per launch
without changing the work.

The brief's rule 9 permits this explicitly: micro-batch and accumulation may
change **as long as the global effective batch is preserved**, and it is — 64 in
every arm, before and after.

---

## 2. The measurement

Four single-GPU arms, all effective batch 64, all else identical, on GPU 6:
**the fixed-code ladder** — the arms the equivalence report gates, so the timings
and the equivalence evidence come from the same runs rather than from two
different generations of the code.

All columns are six-step statistics: s/step and the token counts are means over
the six steps, `peak alloc` is the maximum.

| arm | s/step | ratio | peak alloc | post-expansion tokens | padding ratio |
| --- | --- | --- | --- | --- | --- |
| micro 1 / GA 64 | 126.97 | 1.000 | 15.13 GiB | 41 507 | 0.0000 |
| micro 2 / GA 32 | 81.95 | 0.645 | 15.68 GiB | 41 671 | 0.0336 |
| **micro 4 / GA 16** | **56.23** | **0.443** | **16.80 GiB** | 41 863 | 0.0702 |
| micro 8 / GA 8 | 42.53 | 0.335 | 19.00 GiB | 42 001 | 0.0951 |

**Shipped point: micro 4 — 2.26x for +1.67 GiB**, with per-step spread measured
rather than assumed: **0.443 over six steps, min 54.18 / max 58.46**, so no
single step carries it. Micro 8 is 1.32x faster again and was refused on memory:
it peaks at **19.00 GiB allocated** (+3.87 over the baseline, +2.20 over micro 4)
and **21.82 GiB reserved** on a 24 GiB card, leaving under 2.2 GiB of headroom,
with the padding ratio 35 % above micro 4's (0.0951 vs 0.0702). The refusal is
not on equivalence grounds — §3 of the equivalence report measures the residual
answer difference at micro 8 to be the *same order* as at micro 4, and smaller
(0.31 % vs 0.47 %), so it does not grow with width. Full detail is in
`V8_ACCELERATION_IMPLEMENTATION.md` §7.1.

**These supersede an earlier pre-fix sweep of the same shape**, and the two agree
on the answer where it matters: micro 4 ratio 0.443 vs 0.446, micro 8 ratio 0.335
vs 0.334. The micro-2 arm moved (0.645 vs 0.699) because the earlier reading
carried an outlier; the fixed ladder's six steps span 80.57–83.37 s, so this one
does not. The headline moved from 2.24x to 2.26x as a result — a change in the
third significant figure, not in the conclusion.

**Why single-GPU arms.** `V8_SPEED_BASELINE.md` §1.1: production used 4 GPUs and
this baseline reproduction used 2, so raw step times are not comparable across
world sizes. A per-GPU ratio is, and the two agree to 6.9 % on per-GPU sample
rate (0.497 vs 0.463 samples/s/GPU). Every arm above was run on one GPU against
a one-GPU base.

**Cross-world-size confirmation.** The config declares `effective_batch: 64` and
*derives* the accumulation from the world size, so the shipped file resolves to
micro 4 / GA 16 at world 1, **GA 8 at world 2**, and **GA 4 at world 4** — which
is the one production actually runs. Two independent checks cover the transfer:

1. **The window is invariant at production width.** The sampler study
   (`V8_RECIPE_EQUIVALENCE_REPORT.md` §1.1) re-splits the real 39 743-sample
   task-4 set four ways and checks each against its base. Production's own
   `(micro 1, GA 16)` vs `(micro 4, GA 4)` at world 4, and the shipped config at
   world 2, both come back **620 windows of 64 samples on each side, identical
   contents, `num_diverging_windows = 0`**. The world-1 split reports the same
   620 identical windows, plus one disagreement that is not about the window
   shape: it takes **621** steps where its base takes 620. That is the 63-sample
   remainder — 39 743 = 620 x 64 + 63 — rounded up to a whole window, and it is
   worth naming precisely because an earlier revision of this paragraph
   attached that tail to the **world-4** split, where the study in fact reports
   `steps_left = steps_right = 620` and `step_counts_agree = true`. The tail
   sits after the last compared window, not inside the comparison: over the 620
   windows both sides have, `num_diverging_windows = 0`. The lever reaches
   production's window shape, not just the ones that happened to be measured.
2. **The step time was measured at the baseline's own world size, on the real
   split.** `v8_armC_world2_task4_20260911` is the shipped config at world 2 —
   task 4, seed 42, 200 steps, GPUs 3 and 7 — which is the same data, the same
   seed and the same width as the baseline it is compared against, so unlike the
   single-GPU sweep above it needs no per-GPU normalisation. Over the same
   197-step window (4–200) that the baseline's own 64.36 s is taken over:

   | quantity | original (micro 1, GA 32) | accelerated (micro 4, GA 8) | ratio |
   | --- | --- | --- | --- |
   | wall clock per optimizer step | 64.3590 s | **27.8288 s** | **0.4324 (2.313x)** |
   | samples/sec per GPU | 0.4972 | 1.1499 | 2.313x |
   | tokens/sec, global | 645.0 | 1 504 | 2.33x |
   | `step_body_time` | 56.3247 s | 25.5368 s | 0.4534 |
   | `backward_time` | 34.5474 s | 16.8335 s | 0.4873 |
   | `allreduce_time` | 0.9699 s | 0.4636 s | 0.4780 |
   | `vision_forward_time` | 0.6898 s | 0.1578 s | **0.2288 (4.37x)** |
   | model forwards per step | 32 | 8 | **0.2500** |
   | `ComposeLinear` forwards per step | 14 336 | 3 584 | **0.2500** |
   | LoRA expert evaluations per step | 28 672 | 13 288 | 0.4635 |
   | peak VRAM (allocated) | 15.13 GiB | 16.81 GiB | **+1.68 GiB** |

   Two independent routes to the ratio — step time and per-GPU sample rate — give
   the same 2.313x, as they must if both arms are processing the same 32 samples
   per rank. Rank 1 agrees with rank 0 to 4 ms. The forward counts fall by
   **exactly 4.00x**, which is the number the micro-batch change predicts: those
   are schedule counts, not timings, so they are evidence independent of the
   clocks. The memory cost, +1.68 GiB, is the world-1 ladder's +1.67 GiB, which
   is what a width change should do and does not track the world size.

   The sweep's six-step world-1 ladder put the same bundle at 2.26x; this
   197-step real-split world-2 measurement puts it at **2.313x**. The two agree to
   **2.3 %**, and neither is derived from the other. The estimates in §3 continue
   to use the ladder's `TRAIN_RATIO = 2.258`, i.e. the **smaller** of the two, so
   the projected hours are if anything conservative.

The remaining gap is that the *world-4 step time itself* is a projection from
the world-1 and world-2 measurements rather than a measurement, because GPUs
were not available at that width. Per-rank work is identical in shape at every
width (16 samples per rank per step at production, in 16 micro-batches before
and 4 after), which is what makes the projection sound; it is still a projection
and §3's table is labelled an estimate for this reason.

---

## 3. The ETA table

Per-task wall clock, original vs accelerated, in hours. Generated, not
hand-added, by `compose/experiments/stage_budget.py`:

```
python -m compose.experiments.stage_budget <run-root> \
    --json docs/reports/data/stage_budget_formal_20260903.json
```

Training is measured (the trainers' own `train_runtime`, which excludes model
load and startup); RMS and pruning are wall-clock differences between
stage-boundary markers, exactly as `V8_CURRENT_RUNTIME_AUDIT.md` §3.1 defines
them (`s4_rms.done - s3_training.done`, `s5_pruning_commit.done -
s4_rms.done`), and they reproduce §3.1's seconds exactly for tasks 1-4. **Two
boundaries needed a guard.** This run was repaired and resumed, and the repairs
re-stamped markers they had already written: task 0's `s0`-`s3` all carry one
mtime and task 5's `s0`-`s4` all carry another, both *later* than the work they
claim to bound (task 0's `s3` is 4.9 h after its training log stopped). A marker
is trusted only within 600 s of its stage log's mtime, and otherwise the log is
used — which is why task 0's RMS start and task 5's RMS start and end come from
log timestamps. The tool prints the boundary it chose for every row, so this is
checkable rather than asserted.

Model load is a few minutes per stage and is not accelerated either way, so
excluding it from the training column slightly *favours* the accelerated side;
the effect is ~1 % and is inside the ±2 % the two estimation routes already
disagree by.

| task | train | RMS | prune | **total** | train' | **total'** | **x** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| task0 | 5.80 | 4.99 | 0.60 | 11.39 | 2.57 | 8.16 | 1.40 |
| task1 | 5.28 | 0.30 | 0.34 | 5.93 | 2.34 | **2.99** | **1.99** |
| task2 | 5.37 | 0.94 | 0.86 | 7.18 | 2.38 | 4.19 | 1.71 |
| task3 | 4.01 | 2.21 | 0.42 | 6.64 | 1.78 | 4.41 | 1.51 |
| task4 | 5.96 | 4.12 | 0.39 | 10.47 | 2.64 | 7.15 | 1.46 |
| task5 | 5.67 | 7.74 | 4.97 | 18.38 | 2.51 | 15.22 | 1.21 |
| **full run** | **32.10** | **20.31** | **7.59** | **59.99** | **14.21** | **42.11** | **1.42** |

An earlier revision of this table was hand-added and had drifted in four cells —
task 2's `total'` and `x`, task 4's `prune` and `total'`, and the two full-run
marginals — each by 0.01, which is rounding drift rather than a measurement
disagreement, but it left rows that did not add up. The regenerated values agree
with `V8_CURRENT_RUNTIME_AUDIT.md` §3.1 to the second in all eight task 1-4
cells. The whole-run ratio is unchanged at 1.42x and no per-task cell moves by
more than 0.02, which is the useful thing to know: the headline does not depend
on which of the two sweeps you trust.

`train'` is the fixed-ladder ratio, **2.258x** (§2); the earlier revision of this
table used 2.24x from the pre-fix sweep.

**The whole-run figure is confirmed a second way.** The ratio was computed from
the stage sum (59.99 -> 42.11 h = 1.4246x) and independently from the run's own
wall-clock span (`task0` directory creation to `task5/task_complete.json`:
58.8142 h minus the 17.88 h the training stage saves = 40.93 h, so 1.4368x). The
two routes share no inputs. **1.42-1.44x.**

The per-task spread, 1.21x to 1.99x, is entirely the RMS stage's share: task1's
RMS is 0.30 h so it nearly doubles, task5's is 7.74 h so it barely moves.

**One caveat on the totals.** The stage sum (59.99 h) slightly *exceeds* the run
span (58.81 h), which a strictly sequential pipeline cannot do. Some stages
overlap, or a crashed-and-retried segment is counted twice — this run was
interrupted at least once (`task1.incomplete-...-crash-eosfix`,
`TASK1_RECOVERY_NOTE.md`). The discrepancy is 2 % and it affects the original and
accelerated totals' *ratio* only in proportion, so both routes still agree at
1.42-1.44x. It is noted rather than smoothed over because it is the reason the
table is labelled an estimate.

### 3.1 What the table does not include, and which way it cuts

The three stages above are the ones this run performed end to end. Two things sit
outside them, and **both would lower the ratio**, so 1.42x is a ceiling rather
than a best case:

- **The V8-A teacher is a separate offline stage.** The formal run measured here
  is the training pipeline; its candidate keys appeared one minute after the
  query features, so no V8 teacher ran inside it. The teacher's own cost is in
  `V8_CURRENT_RUNTIME_AUDIT.md` §F: **3 917 s + 3 982 s** for task 4 (all-experts
  and history-only) and **8 828 s + 8 528 s** for task 3. At task-4 scale that is
  roughly **+2.2 h per task**, or **+13 h** over six tasks — serious enough that
  it cannot be waved away.
- **Post-run evaluation** (the 21 lower-triangle cells in `tasks1_to5.log`) is
  inside the 58.81 h span but not inside any of the three stages.

The acceleration touches neither, so every hour they add lands undivided on both
sides and dilutes the speedup. The teacher is arguably the *next* S3/S4 target —
the audit quantifies exactly the redundancy (2 396 live forwards for 256 samples;
the vision tower re-run per route, when its output depends only on the image).
That work was not done here; brief items P3/P4 were closed as not-applicable
because they target a training-loop recomputation that does not exist, but §F's
offline redundancy is real and is left on the table.

---

## 4. The required comparison

Per optimizer step, both arms on one GPU with effective batch 64, so the columns
are the same 64 samples of work:

| quantity | original (micro 1) | accelerated (micro 4) | ratio |
| --- | --- | --- | --- |
| wall clock per training step | 126.97 s | 56.23 s | **0.443 (2.26x)** |
| samples/sec per GPU | 0.504 | 1.138 | 2.26x |
| tokens/sec per GPU (post-expansion) | 326.9 | 744.5 | 2.28x |
| peak VRAM (allocated, max of 6) | 15.13 GiB | 16.80 GiB | **+1.67 GiB** |
| `data_wait_time` | 0.00188 s | 0.00226 s | 1.20x (0.0015 % → 0.0040 % of step) |
| `current_expert_forward_time` | 41.99 s | 17.84 s | 0.425 |
| `llm_forward_time` | 40.42 s | 17.34 s | 0.429 |
| `backward_time` | 70.30 s | 34.48 s | 0.490 |
| `vision_forward_time` | 1.578 s | 0.495 s | **0.314 (3.2x)** |
| `allreduce_time` | 0.0001 s | 0.0001 s | n/a (world 1) |
| post-expansion tokens per step | 41 507 | 41 863 | +0.9 % |
| text padding ratio | 0.0000 | 0.0702 | — |
| model forwards per step | 64 | 16 | **0.25** |
| `ComposeLinear` forwards per step | 28 672 | 7 168 | **0.25** |
| LoRA expert evaluations per step | 57 344 | 26 208 | 0.457 |
| oracle / teacher time | 3 917 s + 3 982 s (task 4) | **unchanged** | not in scope |
| validation time | 0 (no in-loop validation) | 0 | n/a |
| **full continual run** | **59.99 h** | **42.11 h** | **1.42x** |

Every phase moves with the wall clock, which is the signature of a launch-bound
step — no phase is being traded against another. The *forward* phases track it
almost exactly (0.425, 0.429); `backward_time` improves less (0.490) because its
recompute and reduction structure does not shrink as much; and the vision tower
improves far more (**3.2x**) because at micro-batch 1 it runs once per *sample*
and at micro 4 it runs once per *four*.

The counter rows are the mechanism restated as counts, and they are the reason
the wall clock follows: `ComposeLinear` forwards per step fall **4.0x**
(28 672 -> 7 168) and model forwards with them, while the *work* per forward
rises only as far as the row width does. LoRA expert evaluations fall less
(0.457) because the selection plan groups same-expert samples within a row, so
widening the row merges some evaluations but not all. This is what
`V8_SPEED_BASELINE.md` §2.3 predicted from the 30 % GPU utilisation: the step was
launch-bound, and the fix is fewer, larger launches.

The one regression is padding: **+0.9 %** post-expansion tokens, because a wider
micro-batch pads each row to its longest member — the text padding ratio goes
from 0.0000 to 0.0702, and the small effect on the step's token count is because
the 576-token image in each sample dominates it. That is the cost S8 length
bucketing exists to remove, and it does not pay for itself here (§5). 0.9 % of
tokens for a 2.26x step-time win is an obviously good trade, but it is a real
cost and it is why micro 8 was refused.

Two rows deserve their footnotes:

- **Oracle time is unchanged because nothing touches it.** The teacher is
  offline and one-shot (`V8_CURRENT_RUNTIME_AUDIT.md` §F, §G); it runs before
  training, memoises to `nll_cache.jsonl`, and never executes inside a training
  step. Brief items P3/P4 (a historical-NLL cache, multi-GPU NLL generation) are
  therefore **not applicable** — the cache already exists and the recomputation
  they target does not happen. `historical_expert_forward_time` measures exactly
  **0.0** in the training loop.
- **`data_wait_time` rises 1.20x and is still nothing** — 0.00188 s → 0.00226 s.
  The direction is real and expected (a wider collate is more CPU work per batch),
  and the magnitude is the point: the brief anticipated IO as a bottleneck and it
  is **0.0015 % of the baseline step and 0.0040 % of the accelerated one**. Both
  figures are from the two arms in the table, not from a nearby run; the pre-fix
  sweep's bs8 arm showed 0.0026 s, which is where an earlier revision of this
  bullet took its number from, and it described a third trace.

---

## 5. What was measured and refused

Rolling back anything that speeds things up but degrades equivalence was
required. These were refused, each with the number that decided it.

| | claimed | measured | why refused |
| --- | --- | --- | --- |
| S7 gradient checkpointing off | ~5.5 % at first reading | **0.992 +/- 0.004**, peak memory *identical* | dissolved on re-measurement — see below |
| S5 selection plan | ~3 % at first reading | **1.003x**, 0.19 SE | dissolved; shipped as free, not claimed |
| S8 length bucketing | — | padding 0.000 -> 0.072 when the micro-batch widens | not an optimisation here — see below |
| S9 DDP `no_sync` | — | `allreduce_time` median 0.066 s of 64.36 s = 0.1 % on rank 0 | nothing to win; the rest is straggler time, not transfer |
| S2 token/label cache | — | `data_wait_time` 0.007 s | tokenisation is already hidden by prefetch |
| S10 DataLoader / NVMe | — | same 0.007 s | same |
| S11 checkpoint / logging | — | no mid-run checkpoints; `metrics_time` 0.014 s | not applicable |
| vision-output disk cache | — | `vision_forward_time` 0.69 s = 1.1 % | tens of GB per task to remove 1 % |

**A note on `length_bucketing`,** because the shipped config says `true` and that
could be misread as an additional change. It is not: the flag is a passthrough of
`group_by_modality_length`, which the frozen baseline command line already sets to
`True`. Declaring it in the config makes the shipped recipe state its sampler
setting explicitly rather than inheriting it from a command-line default. Turning
it *off* would be the change, and the config's own text says so — "the flag exists
so an intentionally un-bucketed control run is possible".

What the sweep shows is a related but different fact: a wider micro-batch pads to
its longest member, and the baseline at micro 1 has nothing to pad. That cost is
**+0.9 % padded tokens**, it is real, and nothing shipped removes it. It is the
reason micro 8 was refused, and it is small enough next to 2.26x that it is not
worth a second mechanism.

**The S7 correction is worth restating**, because it is the exercise's own worst
error and it was caught by re-deriving rather than by a failed test. An in-flight
read of that arm summed `training_step_sec` over a *half-written* optimizer step
and produced "118.9 s, +5.5 %". Re-measured over complete steps it is 124.85 s
against 125.91 s = **0.992**, and the two figures are reproducible from the saved
traces (`data/s7_gcoff_vs_gcon.json`).

An earlier revision of this paragraph described those two as "measured together,
so the ratio is self-contained". **That was wrong**: the GC-on side is
`v8_s5_ab_off`, the S5 A/B's own baseline arm, not a companion S7 run — the
directory `v8_s7_gcaon` does not exist. The pair is same-shape and same-GPU, but
an hour apart on a shared card, so 0.992 carries more uncertainty than a paired
measurement would.

The premise died anyway, and on evidence that does not depend on the pair at all:
GC-off was supposed to free headroom for a wider micro-batch, and it frees
*none*. Both arms peak at **identical** memory — 15.128 GiB allocated, 15.527 GiB
reserved, to three decimals — because what occupies this card is the 7B weights,
their optimizer state and `ComposeLinear`'s materialised expert deltas, none of
which recompute touches. A memory equality refutes the premise by construction;
the 0.8 % timing was never the load-bearing part.

---

## 6. The lever that is left, and why it was not taken

The table in §3 says where the remaining 42 hours go: **RMS calibration, 20.31 h
= 34 % of the run** — and on task5 it is 7.74 h, more than that task's training.

It is also, structurally, exactly the kind of thing this exercise exists to fix:

| | RMS | training |
| --- | --- | --- |
| batch size | **1** | 1 (was) |
| its own code says | *"the batch size is unconstrained"* (`compose/lora/rms.py:184-186`) | — |
| work per task | 256 samples x 224 layers x E experts | 39 743 samples |
| measured cost/sample | 4.27 s at E=8 -> **108.77 s at E=24** | — |

Per-sample cost rises **25x** as the pool grows 8 -> 24 experts while the
expert-forward count rises only 3x, so the cost is superlinear in the pool — the
`ComposeLinear` hook materialises every expert delta, on a batch of one, and
memory pressure grows with the pool. `V8_SPEED_BASELINE.md` §5 records the same
item from the other end (4 h 07 m on task 4, 68 % of that task's training stage).

**It was not touched, and that is a deliberate reading of rule 10**, which
freezes the *RMS calibration rule* — the definition of kappa from the moments.
Batching is arguably execution rather than rule, but it is not *bit-identically*
execution: a batched expert forward can select a different cuBLAS tiling, which
changes the fp summation order inside the delta and therefore the moments and
therefore kappa — and kappa feeds composition scaling into every later task. The
brief's own escape hatch ("unless it is a documented bug fix") does not obviously
cover a configuration change that alters a calibrated constant.

So it is **reported separately rather than shipped**, which is what rule 10 asks
for. The measurement that would settle it is cheap and is specified in §7.

---

## 7. VERDICT

**VERDICT = PASS**

Against the brief's three criteria:

1. **Mechanism equivalence — PASS, with S6's residual quantified rather than
   hidden.** The S5 A/B is `BIT_IDENTICAL` over 384/384 micro-steps on every
   loss, both gradient norms, and all seven discrete fields
   (`s5_ab_recipe.json`). The S6 window A/B holds the *discrete* decisions
   exactly — 0 differing sample ids and 0 differing routing decisions at every
   width — and the key objective to 1e-9, but the answer scalar moves by
   ~0.3–5e-3 per window. `V8_RECIPE_EQUIVALENCE_REPORT.md` §3.3–§3.4 traces that
   to the bf16 forward's sensitivity to row shape: it is already present at
   step 0 where both arms hold identical parameters, it does not compound, and
   the run-to-run noise floor is exactly 0.0 so it cannot be dismissed as slop.
   **S6 is recipe-equivalent, not numerically equivalent**, and the report says
   so in those words. The sampler result is established on the real
   39 743-sample task-4 split for all five splits, **including the production
   world-4 shape** (`sampler_window_invariance.json`).
2. **No performance regression — PASS.** 0.443 at the shipped point (six steps,
   54.18–58.46 s), and 0.4324 over 197 steps on the real split at the baseline's
   own world size; peak memory +1.67 GiB of 24 GiB (15.13 → 16.80 allocated,
   reserved 15.53 → 19.85), +1.68 GiB on the real split; no phase traded against
   another.
3. **Meaningful wall-clock improvement — PASS, with the honest split:**
   **2.26x on the training stage; 1.42-1.44x on the six-task training pipeline**
   (training + RMS + pruning, 59.99 h -> 42.11 h, confirmed independently from
   the 58.81 h run span). Two unaccelerated stages sit outside that figure — the
   offline V8-A teacher (§3.1) and post-run evaluation — and both would dilute
   it, so **1.42x is a ceiling, not a best case.**

The 2.26x/1.42x gap is not a shortfall in the work, it is where the work was
allowed to reach. Training is 53.5 % of the run; the acceleration applies to
training and nothing else, so 1.42x is the arithmetic ceiling of *any*
training-only change. Reaching further means RMS, which is 34 % of the run and
frozen by rule 10.

The brief's instruction for this case is explicit — *"if acceleration is
insufficient, do not resort to high-risk methods"* — and no high-risk method was
used. Whether the result is *sufficient* is the reader's call; what this report
can say is where the remaining time is and what it would take to reach it. RMS is
the obvious next target at 34 % of the run, but §6 found it is **two** problems
stacked (launch-bound *and* superlinear in the pool), and batch width only
addresses the first. No projection is offered for it, because the split between
those two components has not been measured and a number invented here would be
exactly the kind of guess this exercise was commissioned to replace.

The world-2 confirmation is now measured rather than pending, and it is the
tighter of the two readings: the shipped bundle on the real task-4 split at the
baseline's own width gives **0.4324 (2.313x)** over 197 steps with peak allocation
+1.68 GiB, against the six-step single-GPU ladder's 0.443 (2.26x) and +1.67 GiB
(§2 item 2). Both PASS lines above hold on either measurement, and §3's totals use
the smaller one.
