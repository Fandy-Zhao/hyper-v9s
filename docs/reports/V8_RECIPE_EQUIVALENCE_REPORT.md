# V8-Exact-Accelerated: recipe-equivalence report

The brief's condition for shipping any acceleration is that *the training recipe
did not move*. This report is the evidence for that claim. It is deliberately
organised as argument-then-measurement, because the interesting question is not
"do the numbers match" but "**is the comparison a fair test of the thing we
shipped**".

Every number below is produced by a script in this repository; the raw outputs
are in `docs/reports/data/`.

---

## 1. What "the same recipe" means here, precisely

An optimizer step is defined by four things:

1. **which samples** it sees,
2. **what decision** the router makes for each of those samples,
3. **what objective** is evaluated, and
4. **what update** is applied.

An execution change is legitimate exactly when it leaves all four fixed. The
subtlety is (1): "which samples" is a statement about the *global accumulation
window* — the 64 samples that produce one gradient — and **not** about which rank
received which sample or in what order.

That distinction decides which accelerations are even testable, so it is
established first, by measurement rather than by reading the sampler.

### 1.1 The window is the effective batch; how it is split does not matter

LLaVA builds its sampler as

```python
LengthGroupedSampler(self.args.train_batch_size,               # per_device x n_gpu
                     world_size=self.args.world_size * self.args.gradient_accumulation_steps)
```

so the *megabatch* — the unit that is shuffled, length-sorted, then dealt out —
has size

```
megabatch = (per_device x n_gpu) x (world x GA)
```

**`n_gpu` is 1, so this is just the effective batch.**
`TrainingArguments.n_gpu` counts the devices *this process* can see, and
`_setup_devices` sets it to 1 whenever the run is distributed — which every
launch in this repository is, under `torch.distributed.run`. So
`train_batch_size = per_device`, the DataLoader iterates batches of `per_device`
samples, and

```
megabatch = per_device x world x GA = effective_batch
```

**One optimizer step consumes exactly one megabatch.** Three facts compose, and
none of them is an assumption about the trainer's intent:

1. `Trainer.get_train_dataloader` builds `DataLoader(batch_size=train_batch_size)`
   over the sampler and hands it to `accelerator.prepare`;
2. torch's `BatchSampler` therefore chops the flat sampler output into
   `train_batch_size`-sized batches, in stream order;
3. accelerate's `BatchSamplerShard` runs with `split_batches=False` (the
   default), so `_iter_with_no_split` yields, on process `r`, the batches whose
   index is `r mod world_size`.

Rank `r` therefore runs batches `r, r+world, r+2*world, ...`, and its `GA`-th
consecutive batch ends a step. Because `get_length_grouped_indices` lays each
megabatch out as `world x GA` consecutive chunks of `per_device` each, a rank's
`GA` batches per megabatch land exactly on a megabatch boundary: a step is a
megabatch, and the union over the ranks is the whole of it.

#### The arithmetic is measured, not read off the source

An earlier revision of this section got this wrong in a way worth recording,
because the fix is the evidence. It claimed a `n_gpu x effective_batch`
megabatch, with `n_gpu` the *globally visible* device count, and concluded that
world size moved the window. Both halves are false. Three real runs of this
trainer settle it:

| run | `per_device x world x GA` | samples per optimizer step, observed | source |
| --- | --- | --- | --- |
| baseline, world 2, GA 32 | 1 x 2 x 32 = **64** | **64** (32 per rank) | this run's own `rank0/rank1` metric rows |
| bs1 arm, world 1, GA 64 | 1 x 1 x 64 = **64** | **64** | `v8_s6fix_bs1_20260911` |
| bs4 arm, world 1, GA 16 | 4 x 1 x 16 = **64** | **64** (16 micro-steps of 4) | `v8_s6fix_bs4_20260911` |

and 39 743 / 64 = **621**, which is the step count the production task-4 run
actually took.

The world-2 row is the one that decides the question. A rank logs **32** samples
per step, not 64: that is what `train_batch_size = per_device` predicts, and what
`train_batch_size = per_device x world` forbids. The world-1 rows cannot
distinguish the two readings, because there the product and `per_device` coincide
— which is exactly how the error survived an earlier pass.

One more check is free and comes from a different direction entirely: the
one-GPU bs1 arm takes **126.97 s** per step and the two-GPU baseline **64.36 s**
for the same 64 samples. A ratio of 2.00 is what "the same work on half the
devices" looks like, and 126.97 / 64.36 = **1.97** — the arm is a 6-step window
and the baseline a 197-step mean, so the two windows differ; the agreement is to
1.5 %.

`compose.experiments.sampler_window_invariance` reconstructs this mechanism
directly on the real task-4 training split (39,743 samples, `train_full.json`),
rebuilding `modality_lengths` exactly as `compose/train/data.py` does. It does
not model the window — it runs the sampler, then applies the batch/shard
arithmetic above:

| split compared | effective batch | megabatch | **step windows identical** | steps |
| --- | --- | --- | --- | --- |
| (micro 1, GA 64, world 1) vs (micro 4, GA 16, world 1) | 64 | 64 vs 64 | **yes, 620/620** | 620 vs 621 |
| (micro 1, GA 32, world 2) vs (micro 4, GA 8, world 2) | 64 | 64 vs 64 | **yes, 620/620** | 620 vs 620 |
| **the production shape**: (1, GA 16, world 4) vs (4, GA 4, world 4) | 64 | 64 vs 64 | **yes, 620/620** | 620 vs 620 |
| **world size alone**: (1, GA 32, world 2) vs (4, GA 16, world 1) | 64 | 64 vs 64 | **yes, 620/620** | 620 vs 621 |
| **CONTROL, effective batch moved** — (1, GA 32, world 2) vs (4, GA 16, world 2) | 64 vs **128** | 64 vs **128** | **no, 0/310** | 620 vs 310 |

The tool asserts both directions: every split that holds the effective batch
fixed must agree, *and* the control — which moves it — must disagree, *and* the
reconstructed mechanism must agree with the model-free megabatch arithmetic on
all five rows. A tool that reported "everything moved" would be as useless as one
that reported "everything agrees".

Note what row 4 does that the earlier version of this table could not: it varies
**world size alone** with the effective batch held at 64, and the windows still
agree. Under the wrong `n_gpu` reading this row was the control, and it "moved"
only because the arithmetic had inflated its megabatch to 128. The real control
has to move the *effective batch*, and that is now row 5.

The 620 vs 621 asymmetry on rows 1 and 4 is a tail effect, not a disagreement:
the final megabatch is short (63 samples), so whether it completes one more
accumulation window depends on the micro-batch width. `pad_to_multiple` — which
`V7ComposeTrainer` sets under `v7_require_full_coverage` — is what removes that
ambiguity in a real run. Only the 620 comparable windows are compared.

Four consequences, and each one is load-bearing:

- **S6 is a scheduling change, not a recipe change.** The same 64 samples reach
  the optimizer, so nothing about the objective or the update rule moves. This
  holds for *any* split of the effective batch, not just the one the arm uses.
- **Bit-identity is unavailable, and asking for it would be a category error.**
  The order inside the step — and hence the floating-point summation order of the
  gradient — genuinely differs. So does which rank holds which sample, which
  changes the DDP reduction tree.
- **The comparison is valid across world sizes too, which is a stronger
  statement than the earlier draft could make — and it cost nothing to get.**
  Nothing in the design *needed* the world size held fixed; the claim had been
  weakened to fit a formula that was wrong. Arm C is still run at world 2, but
  that is now a free choice (it matches the baseline's parallelism, so the DDP
  reduction shape is identical) rather than a constraint — see §2.
- **The production shape is checked rather than extrapolated.** Production runs
  at world 4 and the shipped config resolves to micro 4 / GA 4; row 3
  establishes the split-invariance there, on that world's own windows.

The synthetic arm of the same tool (two modalities, 3000 samples) reproduces the
result with the modality-splitting branch of the sampler actually exercised;
task 4 is single-modality, where that branch collapses to the plain length
grouping. Source: `docs/reports/data/sampler_window_invariance.json`.

### 1.2 What this costs the comparison

Because the window contents are fixed but the order is not, the honest verdict
for the shipped flag set is **EQUIVALENT, not BIT_IDENTICAL**, and the
comparison has to be made at the window level rather than the micro-step level:

- `compose.experiments.compare_recipe --mode window` groups the metric rows by
  the trainer's own `step` field, merges the ranks, and compares, per optimizer
  step, the **multiset of sample ids**, the **per-sample routing decision**, and
  the **window loss**.
- Routing is compared as `sample_id -> (route_type, selected_current_ids)`, so
  it stays an exact, per-sample decision even though the samples sit on
  different ranks on the two sides. A histogram would have hidden exactly the
  kind of change the brief is worried about.

The comparator's own tests pin both failure modes it must not have: it calls a
reordered *stream* a `MISMATCH` even when every loss is identical
(`test_reordered_stream_is_a_mismatch_even_with_identical_losses`), and it calls
a pure summation-order change `EQUIVALENT` rather than `MISMATCH`
(`test_summation_order_noise_is_equivalent_not_mismatch`). A gate that passes
everything and a gate that fails everything are equally useless.

---

## 2. The arms

| arm | what it is | batch shape | effective batch = megabatch | world | GPUs | steps |
| --- | --- | --- | --- | --- | --- | --- |
| **S0 / baseline** | the frozen pre-V8 command line, no V8 flags at all | micro 1, GA 32 | 64 | 2 | 3, 7 | 200 |
| **C / shipped** | `--compose_v8_config configs/v8_exact_accelerated.yaml` | micro 4, GA 8 | 64 | 2 | 3, 7 | 200 |

Both arms: task 4, `train_full.json`, seed 42, effective batch 64,
`--profile_training True`, identical expert pool, key state and runtime contract.
The comparison uses the **first 100 optimizer windows of each**, which is the
first rung of the brief's 100 / 500 / 1000 ladder; the remaining 100 steps are
simply not compared.

Arm C's command line is the baseline's own `/tmp/run_v8_baseline.sh`, and a
token-level diff of the two confirms exactly two differences:

1. `--compose_v8_config` (plus the query-tensor directory the per-arm config copy
   fills in), and
2. `--per_device_train_batch_size 1 --gradient_accumulation_steps 32` is absent,
   because the *config* owns the batch shape. Stating
   `--gradient_accumulation_steps 32` next to `effective_batch: 64` is the
   contradiction the startup guard exists to refuse.

**One correction to the arms, recorded because the first draft of this design
was wrong in a way that would have made the A/B meaningless — and one choice
that an earlier draft wrongly defended as a constraint:**

- **Arm C runs 200 steps, not 100.** `warmup_ratio 0.03` and the cosine decay
  are defined over `max_steps`, so the earlier arm C — which used
  `--max_steps 100` — held a *different learning rate at every step* than the
  baseline it was compared against. That is a recipe change under the brief's
  rule 10, and it is not one the acceleration is allowed to make. Arm C now
  trains the baseline's own 200 steps, and the first 100 windows are compared.
- **Arm C runs at world 2 — by choice, not necessity.** An earlier draft
  required world 2 because it believed a world-1 arm would read 64-sample
  windows against the baseline's 128. That belief came from the `n_gpu` error in
  §1.1 and is false: at micro 4 / GA 16 a world-1 arm has the same effective
  batch 64 and therefore the same windows (that is row 4 of the §1.1 table).
  World 2 is kept because it matches the baseline's parallelism, so the DDP
  reduction shape is identical between the arms and the comparison isolates the
  batch shape. It is a free variable now, and §1.1 says why.

The correction does not change what is being tested. The question §3 answers is
whether S1 + S5 + S6 together leave the per-window sample set, routing decision
and objective where they were.

### 2.1 What is being tested, and what is not

Arm C bundles **three** flags: `cache_queries` (S1), `compose_selection_plan`
(S5) and the micro-batch split (S6). Two of them were already established
separately, so the bundle is not a leap of faith:

- **S1** — `compose.experiments.verify_query_tensor` compares the JSON feature
  cache against the tensor cache **per sample id** over all six tasks x both
  splits (12/12 splits `MATCH`, 0 differing elements, `max_abs_diff = 0.0`).
- **S5** — `s5_ab_recipe.json`: S5 off vs on, 384/384 micro-steps, is
  **BIT_IDENTICAL** on every loss, both gradient norms, and all seven discrete
  fields. This is the strongest verdict the comparator can return, and it is the
  right one for a change that removes redundant *CPU* work without touching the
  arithmetic.

That leaves **S6** as the genuinely new claim, and the one this report has to
carry.

---

## 3. Results

Two questions, answered in order: does S6 *alone* leave the window, the routing
decision and the objective where they were (§3.1–§3.5), and does the *bundle*
that ships (§2.1) do the same on the baseline's own data (§3.6, arm C).

### 3.1 The S6 band

The S6 isolation study holds the effective batch at 64 and moves only the split
— world 1, GPU 6, six optimizer steps each, all on the corrected code:

| arm | per_device | GA | effective batch / megabatch | wall s/step | padding ratio |
| --- | --- | --- | --- | --- | --- |
| `bs1` | 1 | 64 | 64 | **126.97** | 0.0000 |
| `bs2` | 2 | 32 | 64 | **81.95** | 0.0312 |
| `bs4` | 4 | 16 | 64 | **56.23** | 0.0670 |
| `bs8` | 8 | 8 | 64 | **42.53** | 0.0887 |

The timing column is the whole point of S6 and is discussed in
`V8_SPEEDUP_REPORT.md`; it is repeated here only because the width that buys the
speed is also the width that has to be shown harmless. bs1 → bs8 is **2.99×**
single-GPU, and the padding column is the mechanism.

**The padding column here is step 1's value, because the decomposition below is
step 1's.** The six-step means are slightly higher — 0.0336 / 0.0702 / 0.0951 —
and they are what `V8_ACCELERATION_IMPLEMENTATION.md` §7.1 and `V8_SPEEDUP_REPORT.md`
§2 quote; the step-1 figure is used here because it pairs with a single step's
token count, which is what makes the arithmetic below exact. Neither table is a
different measurement of the same thing, and the two must not be read as
disagreeing.

Each of the step's 64 samples carries one 576-token image, so the 41 495
post-expansion tokens at width 1 are 36 864 image tokens plus 4 631 of text; no
row is padded there (ratio 0.0000), while at width 8 the ratio is 0.0887 — that
is **457** padded text tokens (0.0887 × 5 152), which is exactly the measured
rise in the same step's post-expansion count, 41 952 − 41 495. The extra tokens
are padding, and the 64 samples behind them are the same ones.

### 3.2 The key term: a real defect, now fixed

The first gating of this ladder failed, and the failure was real rather than
numerical. The key objective scaled *with the row width* — a ratio equal to the
width, to four significant figures — because `ComposeLlavaForCausalLM` inherits
`LlamaForCausalLM` rather than `LlavaLlamaForCausalLM`, so the per-sample
answer-loss branch never ran for a compose model and the trainer's request for one
was swallowed by `**kwargs`; the key term was then accumulated as a per-micro-batch
*sum* against an answer term that was a *mean*. That is a unit bug, not an
acceleration, and it is reported separately under the brief's rule 10 carve-out in
`V8_ACCELERATION_IMPLEMENTATION.md` §3.3.

With the fix in place the same ladder gives:

| comparison | row width | key max abs | key max rel | pre-fix key ratio | pre-fix routing |
| --- | --- | --- | --- | --- | --- |
| bs1 vs bs2 | 2 | 1.397e-09 | 1.572e-08 | **2.0014** | differs |
| bs1 vs bs4 | 4 | 8.382e-09 | 9.890e-08 | **4.0028** | differs |
| bs1 vs bs8 | 8 | 3.725e-09 | 4.193e-08 | **8.0056** | differs |

The key objective now moves at floating-point noise, `scales_with_row_width` no
longer fires, and — the part that matters more — the **routing is exactly
identical at every width**, where pre-fix it was not at any of them. Both columns
are measured on the *same* pairs as their post-fix counterparts, so the table
reads down a column rather than across two experiments
(`data/prefix_keywidth_bs1_vs_bs{2,4,8}.json`).

The small excess over the exact width (2.0014 for 2, 8.0056 for 8) is not noise
around a clean ratio — it is the unit bug *and* its consequence. The wrong
gradient was applied, so the router genuinely drifted, which is why the pre-fix
routing column is "differs" on all three rows: on `bs1 vs bs2` a single sample
(`v7_t4_train_451`, route `NewNew`) has its second current expert flip from 16 to
17 by window 5. A bigger wrong objective drifts sooner and further, which is the
same statement as the ratio column.

### 3.3 What still differs, and why it is not the objective

The gate still returns `MISMATCH`, and it is worth being precise about what is
left, because the honest reading is narrower than the verdict word:

| comparison | sample sets | per-sample routing | answer max abs | answer max rel | verdict |
| --- | --- | --- | --- | --- | --- |
| bs1 vs bs2 | identical | identical | 2.976e-03 | 4.593e-03 | MISMATCH |
| bs1 vs bs4 | identical | identical | 3.067e-03 | 4.735e-03 | MISMATCH |
| bs1 vs bs8 | identical | identical | 2.243e-03 | 3.101e-03 | MISMATCH |

The window membership is exactly invariant (0 differing sample ids), and every
per-sample routing decision is exactly invariant (0 differences), at every width.
Only the answer scalar moves.

One fact in that table is worth drawing out, because it rules out a reading that
would change the verdict: the residual **does not grow with the width**. Width 8
is the widest arm and has the *smallest* worst-case relative difference
(3.101e-03, against 4.735e-03 at width 4). A distortion introduced by batching
would scale with how much batching there is; this does not, because it is
rounding, not bias. The three columns are worst cases over the same six steps, so
they are directly comparable.

**The decisive experiment is free, and it is at step 0.** Both arms start from
identical parameters, so any difference in the loss at step 0 is forward-pass
arithmetic and nothing else — no optimizer step, no accumulated divergence, no
routing drift can be involved. bs1 records each sample's *individual* loss, since
one sample is one row; the batched arms record row means over the same 64
samples. Comparing them:

| width | window answer loss | mean of the same 64 individual losses | window rel | **per-row \|batched − mean(individual)\|** | max per-row |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.721153 | 0.721153 | — | 0.0000 | 0.0000 |
| 2 | 0.719121 | 0.721153 | −2.82e-03 | 0.0080 | 0.0338 |
| 4 | 0.720033 | 0.721153 | −1.55e-03 | 0.0061 | 0.0169 |
| 8 | 0.723396 | 0.721153 | +3.11e-03 | 0.0023 | 0.0049 |

A single sample's answer loss moves by up to **3.4 %** when it is evaluated in a
padded batch instead of alone, with the same weights and the same loss
definition. Those per-row errors are large and of mixed sign, so they partially
cancel and the window mean moves only ~0.3 % — which is precisely the effect
size the gate is reporting above. The window column here shows the sign directly:
widths 2 and 4 shift the step-0 window *down*, width 8 shifts it *up*, and none
of the three magnitudes tracks the width. This table is a single window at step
0, not a worst case over six steps, so its numbers are not the ones in the table
above; both are reported, and they are different statistics of the same effect. The mechanism is that padding and row width
change the reduction shape of every matmul in the forward (and, under S5's
selection plan, which samples share an expert's grouped matmul), and bf16
arithmetic answers a different reduction shape with a different rounding.

Two further checks rule out the alternative explanations:

- **It does not compound.** Across the six steps the window difference ranges
  from 1.2e-04 to 4.7e-03 with no growth and no consistent sign (bs2: −2.8e-03,
  +1.3e-03, −2.1e-03, −4.6e-03, −3.3e-03, +3.2e-03). A moved objective or a
  diverging trajectory would be systematically signed and would grow; noise with
  a zero mean does exactly this.
- **It is not run-to-run variation.** The same configuration run twice on the
  same data is **bit-identical** — 43 overlapping metric rows, `max|Δ answer|`
  = `max|Δ key|` = `max|Δ total|` = **0.000e+00**. So the ~0.3 % is a real,
  deterministic consequence of the width change, not measurement slop. That is
  also why it cannot be waved away as "within noise": the noise floor here is
  exactly zero.

### 3.4 The tolerance basis, stated rather than chosen

The comparator's default `--rel-tol 1e-3` was calibrated for a different
situation: a pure summation-order change, which in practice lands at ~1e-7 (S5
measures 0.0). A micro-batch-width change is a strictly larger perturbation, and
§3.3 measures how large: **~5e-3 at the window level, worst case over six
steps**, with a per-row spread an order of magnitude above that.

So the strict gate at 1e-3 returns `MISMATCH`, and the same comparison at a
tolerance derived from the measurement returns `EQUIVALENT`. Both are recorded in
`docs/reports/data/`; neither is presented as the answer. The number that matters
is not the tolerance but the measurement behind it — a threshold loosened until
the test passed would have been worthless, and the reason this one is not that is
that the step-0 experiment fixes it independently of any run-to-run agreement:
with identical parameters, no implementation of a wider micro-batch can do better
than ~0.3 % here, because that is what the model's own bf16 forward does when the
row shape changes.

The principled way to remove the residual is not a tolerance at all. It is
unpadded — packed or varlen — attention, which processes each row member at its
own length and so reproduces the width-1 arithmetic at any width. That is
exactly the `use_flash_attention` item on the brief's list, it is a much larger
change than this exercise's budget allows, and it is recorded here as
**measured, unclaimed work** rather than quietly folded into S6's cost.

The six-step worst case above understates the effect's tail, and §3.6 measures it
properly over 100 windows on the shipped bundle. The correction is worth stating
where the number is used rather than only where it is superseded: 5e-3 is the
worst of *six* draws, and a stationary noise of the measured σ produces a worst
of ~1.6e-02 by n = 6 and ~2.65e-02 by n = 100, so the six-step figure was
under-powered rather than wrong. §3.6 derives the tolerance from the distribution
instead, which is also what makes it extrapolate to the 500- and 1000-step rungs
this report does not reach.

### 3.5 What §3.1–§3.4 establishes

S6 changes the *allocation* of the effective batch across rows, not the batch,
not the samples, not the routing, and not the objective. Its measured cost is a
~0.3 %-per-window perturbation of the answer scalar at step 0 — the arithmetic
signature of bf16's sensitivity to reduction shape, which the brief's rule 9
sanctions when it permits the micro-batch to change at a fixed effective batch.
The verdict for S6 is therefore **recipe-equivalent, not numerically
equivalent**, and the report says so in those words rather than in a single word.

"Does not compound" was established here over six steps; §3.6 re-establishes it
over 100 windows on the shipped bundle and finds the same answer, with the
absolute residual plateauing at ~3.3e-03 while the loss decays — which is what
makes the claim a measurement rather than an impression.

### 3.6 The shipped bundle — arm C

The bundle is `configs/v8_exact_accelerated.yaml` — the three effective flags of
§2.1: S1's query cache, S5's selection plan and S6's micro-batch split — and arm C
is that bundle run at **the baseline's own world size** (two ranks, GPUs 3 and 7,
task 4, seed 42), so the comparison does not have to explain a width change and a
flag change at once. Both arms walk the same 100 optimizer windows of 64 samples,
which is **6 400 distinct sample ids**, out of each arm's 200. (The config also
declares `length_bucketing: true`, which is the sampler the baseline command line
already runs; it is a declaration, not a change, which is why §2.1 counts three.)

**The recipe claim is exact, and it is the claim that would have caught a
forbidden acceleration:**

| | measured over the 100 windows |
| --- | --- |
| window membership (`sample_ids` per window) | identical — **0 of 100** windows differ |
| per-sample routing (`route_type`, `selected_expert_ids`) | identical — **0 of 100** windows differ |
| routing divergences | **none**; the list is empty |
| `key_loss`, the router's own objective | max abs **4.98e-08**, max rel **6.17e-07** |

The first two rows are the test. Most of what the brief forbids is a change to
*which samples the current expert gets to compete on* — a teacher frozen to its
task-start value, a truncated candidate list, a skipped current-expert
competition, a subsampled reuse set — and every one of those lands in the sampler
or the router before it lands in a loss. Over 6 400 samples, none of either
moved. The loss columns catch the remainder: an inflated Key batch or a moved
loss weight shifts `key_loss` without disturbing a single discrete field, so the
6e-07 on that row is a check the routing comparison cannot make on its own.

**The objective does move, and both verdicts are recorded rather than one:**

| artifact | tolerance | verdict |
| --- | --- | --- |
| `data/recipe_ab_window.json` | default (`abs 1e-06` / rel `1e-3`) | **MISMATCH** — `answer_loss` max rel 2.44e-02 |
| `data/recipe_ab_window_tol.json` | derived below (rel `3.5e-02`) | **EQUIVALENT** |

The two files agree on every number — `answer_loss` max abs 1.1856e-02, max rel
2.4392e-02; `total_loss` max abs 1.1856e-02, max rel 2.4008e-02 — and differ only
in the word at the top, which is the point §3.4 was making. What follows is the
measurement that decides which word is the honest one.

The residual was characterised over the same 100 windows by
`compose.experiments.window_residual`, which exists because "the worst window
moved by 2.4 %" and "the trajectory is diverging" are the same number and
opposite claims:

- **It is unbiased.** Mean *signed* relative residual **−1.2565e-03**, sd
  8.7361e-03, so t = **−1.44** against a standard error of 8.74e-04. A moved
  objective leaves a signed offset; this is a mean that is zero to within its own
  noise.
- **It is stationary, and the worst window is exactly the worst window a
  stationary process predicts.** The largest of n draws from a zero-mean process
  of sd σ sits near σ·√(2 ln n). With σ measured at 8.7e-03, that predicts
  2.65e-02 at n = 100 and the observed max is **2.50e-02 — a ratio of 0.943**.
  The extreme value is not an outlier; it is what 100 draws of this noise look
  like.
- **It does not compound, and the block table is what shows it.** The relative
  residual grows early and then plateaus, which on its own would be alarming;
  the absolute column explains it:

  | windows | mean abs rel | mean abs **abs** | mean loss |
  | --- | --- | --- | --- |
  | 0–19 | 3.266e-03 | 1.914e-03 | 0.59972 |
  | 20–39 | 6.006e-03 | 2.673e-03 | 0.44916 |
  | 40–59 | 8.883e-03 | 3.874e-03 | 0.42930 |
  | 60–79 | 8.110e-03 | 3.350e-03 | 0.40573 |
  | 80–99 | 9.001e-03 | 3.282e-03 | 0.36858 |

  The absolute residual **plateaus at ~3.3e-03 from window 40 onward** while the
  loss decays from 0.60 to 0.37. The relative column therefore rises and then
  flattens for the only reason it can — its denominator is shrinking under a
  fixed arithmetic floor. A divergence would grow the absolute column, and the
  absolute column stops growing. (An early read of the still-running arm did
  produce one much larger figure, −7.4e-02, in whatever window was being appended
  at that moment. It is not a residual: that window held 3 of its 8 micro-steps,
  so its row-mean is an average over different rows. Re-read once the window
  filled, the same index gives +8.4e-03. `window_residual` drops such windows
  against each arm's own modal row count, and no window in the compared 0–99
  range is one.)

**This corrects §3.4, which had too few windows to see the tail.** The figure
quoted there — "~5e-3 at the window level, worst case over six steps" — is the
worst of six draws, and six draws cannot see the 2.4e-02 that 100 draws produce
(σ·√(2 ln 6) = 1.6e-02 would already have been the expectation at n = 6). The
tolerance in the table above is therefore derived from the *distribution* rather
than from a worst case: σ·√(2 ln n) gives 3.08e-02 at n = 500 and 3.25e-02 at
n = 1000, so **rel 3.5e-02** covers the brief's whole 100 / 500 / 1000 ladder with
margin.

That number is worth reading against the one a pass-seeking threshold would have
used. The observed 100-window max is 2.50e-02, so a tolerance chosen to pass
*this* run would sit just above it — 2.6e-02, or 3e-02 at the outside. 3.5e-02 is
deliberately not that: it is set by the **1000-step** projection, i.e. by a window
count this exercise never runs and therefore never got to fit. The residual
circularity is stated plainly in point 1 below rather than hidden here.

Two things this section does **not** claim:

1. **σ is estimated from the same 100 windows it is used to predict.** The
   extreme-value check is a consistency check, not an independent validation.
   The independent evidence is §3.3's step-0 experiment, on different arms and a
   different width, which measured the same mechanism (per-row perturbation up to
   3.4 %, window mean ~0.3 %) with no optimizer step involved at all.
2. **The residual is not perfectly zero-mean, and the halves say where.** The
   first 50 windows average **−3.075e-03** and the last 50 **+0.562e-03**, a
   half-to-half drift of **2.94σ**. The parsimonious reading is a width offset
   that is largest early, not a divergence — and the absolute column is again what
   makes it readable: the first half's signed *absolute* residual is
   **−1.55e-03**, which is the sign and order of the width-4 offset §3.3 measures
   independently at step 0 (−1.12e-03), while the second half is **+0.22e-03**,
   i.e. *smaller*. A divergence grows; this shrinks. An offset of ~1e-3 absolute,
   over 100 steps, with routing exactly invariant and the key objective at 6e-07,
   is the arithmetic §3.3 describes and not a different recipe.

The verdict for the shipped bundle is therefore the same in kind as S6's, and for
the same measured reason: **recipe-equivalent, not numerically equivalent** —
every discrete decision identical, the router's objective at rounding, and the
answer scalar perturbed by bf16 reduction-shape noise that this section shows to
be unbiased, stationary and non-compounding.

---

## 4. Reproducing

```bash
R=/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs
FORMAL=$R/v7_gpu01_cached_query_formal_20260903
BASE=$R/v8_baseline_task4_200step_20260911        # S0, world 2, micro 1 / GA 32
ACCEL=$R/v8_armC_world2_task4_20260911            # arm C, world 2, micro 4 / GA 8

# the structural premise (§1.1)
python -m compose.experiments.sampler_window_invariance \
    --data-path $FORMAL/task4/data/train_full.json \
    --report docs/reports/data/sampler_window_invariance.json

# the window A/B (§3) -- the trainer's own step field reconstructs the global
# optimizer window, so the two arms need not share a world size or a rank split
python -m compose.experiments.compare_recipe --mode window \
    --arm S0=$BASE/metrics --arm C=$ACCEL/metrics \
    --steps 100 --report docs/reports/data/recipe_ab_window.json \
    --markdown docs/reports/data/recipe_ab_window.md

# the same comparison at the tolerance §3.4/§3.6 derive -- identical numbers,
# different verdict -- plus the characterisation of the residual itself
python -m compose.experiments.compare_recipe --mode window \
    --arm S0=$BASE/metrics --arm C=$ACCEL/metrics \
    --steps 100 --rel-tol 3.5e-2 \
    --report docs/reports/data/recipe_ab_window_tol.json \
    --markdown docs/reports/data/recipe_ab_window_tol.md
python -m compose.experiments.window_residual \
    --arm S0=$BASE/metrics --arm C=$ACCEL/metrics \
    --steps 100 --report docs/reports/data/recipe_ab_window_residual.json

# the per-micro-step A/B, for arms that DO share a world size and accumulation
# (these two are world 1, so their metrics are unsuffixed and live under
#  training/, not metrics/)
python -m compose.experiments.compare_recipe \
    --arm S5_off=$R/v8_s5_ab_off_20260911/training \
    --arm S5_on=$R/v8_s5_ab_on_20260911/training \
    --accumulation 64 --report docs/reports/data/s5_ab_recipe.json
```

The launch scripts for both arms are frozen at `/tmp/run_v8_baseline.sh` and
`/tmp/run_v8_armC_world2.sh`; arm C additionally writes a copy of the config it
actually used, with the query-tensor path filled in, plus a `config.sha256`, into
its own `logs/` directory. Both arms write per-rank metrics
(`train_steps.rankN.jsonl`) because both run at world 2, which is what lets
`--mode window` merge the ranks into the global window.

All three §3.6 artifacts were regenerated against the **completed** 200-step arm C
— not against the partial run they were first read from — and every number in the
section reproduced to the last digit. That matters for the window count rather
than the totals: the compared range is the first 100 windows either way, so the
edit is a check that a live file was not being read, which is the failure the
completeness test in `window_residual` exists to catch.
