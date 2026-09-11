# Hyper-LLaVA V8 — Implementation and Acceptance Report

**Method**: V8 Answer-Supervised Multi-Key Expert Pool
**Branch**: `exp/v8-answer-supervised-multikey`
**Audit base**: `main` @ `9ff2b285b2cb4ac131519d9a499ff37fd25a55f3`, working tree clean
**Written**: 2026-09-11

This report answers PART 49's causal chain Q1–Q6 with artefacts produced by real
runs. Where an experiment was not executed, the section says **NOT RUN** or
**NOT TESTED** and gives the cost reason; nothing is inferred into a result.

---

## 1. Executive Summary

**V8 in one paragraph.** V7 grows an expert pool of `1 Expert : 1 Key` and routes
every sample to the two experts whose single key scores highest. V8 asks two
different questions. **V8-A**: which samples does a *historical* expert already
answer correctly, decided by an Answer-Supervised Teacher that uses training-time
ground truth, with task correctness (`M`) deciding `solved` and teacher-forced
answer NLL (`L`) used only to rank among experts that already solved — never as a
threshold. **V8-B**: give one historical expert a *second* key on a later task's
distribution (`1 Expert : N Keys`), so the router can find it there, and train a
candidate expert only on what no historical expert can reach — while the
historical LoRA, its committed origin keys, and the base model stay frozen.

**What was built.** `compose/v8/*` — the multi-key pool, the teacher, the
metric adapters, lazy alias-key creation, the gradient gating, the freeze ledger,
the trainer, the inference router — plus eight experiment drivers, all new files
beside V7 rather than edits to it. `git diff --stat 9ff2b28..HEAD` is **34 files,
11,874 insertions, 0 deletions**, and **zero** of those files are under
`compose/v7/`, `compose/adapters/`, `compose/eval/` or `llava/`. The suite is
**608 tests passing** (75.76 s), 548 of them pre-dating this work.

**What was measured.** Four full teacher runs — Task 3 (IconQA) and Task 4
(CLEVR-Math), each in an all-experts and a history-only scope — over the frozen
23-expert pool, 256 validation samples each, 7.02 GPU-hours total, with the
official evaluator scoring every route. 64,768 seeded pair NLLs replayed
byte-exactly in all four (`max_abs_diff 0.0`).

**The headline result, and the correction it needs.** Against the NLL-oracle
ceiling, V8's policy closes **95.7 %** of the gap on Task 4 (94.14 % vs V7's
67.97 %) and **106.5 %** on Task 3 (87.11 % vs 67.97 %). Those numbers are real
but they are **not reuse**, and most of what they measure is self-reuse: the
all-experts scope lets the teacher route to the task's own experts, which do not
exist while that task is being learned. With the task's own experts removed, the
same teacher reaches 44.92 % and 33.59 % — **below V7's 67.97 %** on both tasks,
and above the frozen base alone (16.41 % / 17.97 %) by **+28.51** and **+15.62**
points. Genuine cross-task reuse is **73/256 = 28.52 %** (Task 4) and
**40/256 = 15.63 %** (Task 3); more than half of each "historical reuse" figure
is the frozen base answering correctly on its own.

**The strongest evidence for V8's design.** The capability is present but the
router cannot find it: with historical experts only, solver recall@1 is 27.4 %
and 27.5 %, against 71.9 % and 79.7 % when the task's own experts are present.
§17 measures the causal claim directly — add one untrained alias key per solving
expert and historical recall@1 rises to **60.3 %** and **80.0 %**, with **100 %**
of the key-selection wins going to the alias rather than the origin key. That is
the `1 Expert : N Keys` design working exactly as specified, on real data, before
any training.

**What was not run.** V8-B — the candidate expert, the alias-key training, the
gradient gating on real data — is **NOT RUN**. §19 sizes it from §21's measured
costs and recommends a one-task pilot over a 500-sample teacher (~3 h) rather
than a full six-task loop. The 311 samples that both tasks' history-only runs
could not solve are the worklist such a pilot would be judged against, and on
Task 3 that set is *certified*: all 12 visible historical experts were scored
against all 170 of them.

**Two verdicts, reported separately (PART 37).**

* **Code acceptance: PASS.** The specification's mechanisms are implemented,
  tested and enforced — the freeze by an optimizer whitelist plus a checksum
  ledger, the teacher's `M`/`L` separation by construction and by a rejection
  test, inference purity by an AST scan with negative controls. V7 is untouched
  and its tests pass by construction. Two real defects were found *during* the
  campaign and fixed with regression tests (the key-target rule was not total;
  `create_alias_keys` built a key the pool rejects), and a third — a path bug in
  the Task-0 parity harness — is fixed but not yet re-run. None of the three was
  hidden, and none is a violation of the specified method.
* **Method acceptance: PARTIALLY VERIFIED — the retrieval half holds, the reuse
  half is smaller than the headline and the training half is unmeasured.** V8-A's
  teacher does identify samples historical experts solve, and §17 shows a second
  key substantially fixes the retrieval failure. But a history-only V8 policy
  does **not** beat V7 on either task, genuine cross-task reuse is 28.5 % / 15.6 %,
  and V8-B — the part that would make the pool grow more slowly — was never run.
  This is **not** an implementation failure, and PART 46 item 20 is honoured: the
  failures above are reported rather than dropped.

**Recommendation: NO-GO for the full six-task loop; GO for one Task-1 V8-B
pilot**, with the §19.5 scope and the decision rule in §25.5.

---

## 2. Initial Repository State

| Item | Value |
| --- | --- |
| Branch at start | `main` |
| Commit at start | `9ff2b285b2cb4ac131519d9a499ff37fd25a55f3` — *docs(v7): expert-learning summary report for the six-task formal run* |
| Working tree | clean (no modified or untracked files) |
| Development branch | `exp/v8-answer-supervised-multikey`, branched from `9ff2b28` |
| Python env | `/home/zhaozhuofan/miniconda3/envs/hyper` |

**Existing V7 architecture found in the tree** (read from code, not assumed):

* `compose/adapters/lora.py` — `ComposeLinear` / `ComposeSelection`, the
  multi-expert LoRA composition kernel (`MAX_ACTIVE_EXPERTS = 4`,
  `MAX_INFERENCE_EXPERTS = 2`, `PAD_EXPERT_ID = -1`,
  `DEFAULT_PAIR_SCALE = 1/sqrt(2)`).
* `compose/pool*`, `compose/v7/pool.py` — V7's `1 Expert : 1 Key` pool: a single
  `nn.ParameterDict` of 1536-D routing keys, each key id `==` expert id.
* `compose/eval/` — `eval_task.py` (fixed expert or precomputed two-expert route),
  `nll_eval.py` (teacher-forced answer NLL), `load_compose.py` (kappa
  calibration from `rms_calibration` in the committed manifest).
* `compose/train/`, `compose/v7/hf_trainer.py` — training loop with a tensor-identity
  optimizer whitelist at `compose/v7/hf_trainer.py:265`.
* `compose/v8/` — **does not exist** at the audit base; it is created entirely by
  this work.

**Available checkpoints** (all frozen, never written to by any V8 code path):

| Artefact | Path | sha256 |
| --- | --- | --- |
| Committed key state | `…/v7_gpu01_cached_query_formal_20260903/task5/committed/v7_keys.pt` | `337891cf…c0bd9b` |
| Expert manifest | `…/task5/committed/compose_experts.json` | `0128c3c4…c54f9d` |
| Expert weights | `…/task5/committed/compose_experts.bin` | `b9f68186…3012b5` |
| Pair diagnostic (NLL seed + V7 answers) | `…/v7_final_pool_pair_upper_val256_20260907/` | — |

The three sha256 values above are byte-identical to the fingerprints V7's own
pair-diagnostic recorded on 2026-09-07 in
`…/v7_final_pool_pair_upper_val256_20260907/task0/selection_plan.json →
source_fingerprints_before`. They were re-hashed after the V8-A campaign and are
still identical (§10).

---

## 3. V7 Audit

Read from the checked-out code at `9ff2b28`. Each row states what V7 actually
does, which is what V8 is measured against.

### 3.1 Fixed multimodal query

`compose/v8/query.py` reuses V7's query **verbatim** — `q = L2Norm(concat(LayerNorm(z_visual),
LayerNorm(z_text)))`, 1536-D, zero parameters, detached. The cached queries used
by every experiment are the V7 ones at
`…/v7_gpu01_cached_query_formal_20260903/task{N}/features/val.json`, whose
contract hash is `6c51879f401179b50452e223795ee97cde4f228898d9921a6d7bcf4cbb59232e`.
V8 did not recompute, rescale or re-normalise a single query.

### 3.2 Routing

V7 routes over `num_experts` keys where key id `==` expert id, so a `topk` over
keys is a `topk` over experts *by construction*. Its budget is Top-2
(`compose/eval/eval_task.py:304-305` refuses any validation selection that is not
two distinct experts, which means V7's own harness can express exactly
"one fixed expert" or "one two-expert route per sample" and nothing else).

### 3.3 Expert pool

| Property | V7 value |
| --- | --- |
| Experts | 23 (task0: 0–3, task1: 4–7, task2: 8–11, task3: 12–15, task4: 16–19, task5: 20/21/23) |
| Keys | 24 `{candidate: 0, historical: 23, pruned: 1}` |
| Pruned | expert 22 |
| LoRA | rank 8, alpha 16, 32 layers × 7 projections, 19,988,480 params/expert |
| Keys per expert | exactly 1 |

### 3.4 Training, pruning, checkpoint

V7 trains with an optimizer whitelist keyed on tensor identity
(`compose/v7/hf_trainer.py:265`), prunes whole experts, and commits
`v7_keys.pt` + `compose_experts.json` + `compose_experts.bin` per task.

### 3.5 Teacher infrastructure

**V7 has no teacher** — and this is stated by V7's own runner, not inferred:
`compose/experiments/v7_task_run.py`'s docstring says *"Unlike V6.2 this runner
has no teacher, residual, clustering, calibration or warm-up stages. The declared
full train split is used for query center and training from optimization step
one."* The live formal run's stage markers confirm it (`s0_full_data`,
`s1_fixed_queries`, `s2_candidates`, `s3_training`, `s4_rms`, `s5_pruning_commit`).

**But the repo does contain the teacher V8 grew out of.** The legacy V6.2
pipeline `compose/experiments/task_run.py` (S0–S12, *Query-Clustered Residual
Expert Discovery*) has:

* **S2 "teacher search"** — `empty/single/pair over the per-sample Top-M`, i.e.
  the same search shape as V8's STEP A/B/C plus pair search; and
* **S3 "residual split"** — `old_teacher_loss > tau_res`, with `tau_res: 2.0`
  hard-coded at `task_run.py:168` and applied at `task_run.py:1270-1299`.

S3 is precisely the rule the V8 specification forbids — *"绝对禁止：Answer NLL <
某个统一阈值 => solved"* — a single global magic number deciding which samples
need new capability. V8's substantive contribution is therefore well-posed
against a real predecessor rather than a straw man: it keeps the S2 search shape
and replaces the S3 signal with the task metric. That is also why the V8 work
does not need to build a teacher harness from nothing (see §19's cost model).

### 3.7 What the legacy teacher proves about V8's design

Two things worth separating, because they point opposite ways.

**V8's search shape is inherited, not invented.** The legacy teacher's
hyperparameters and V8's are the same numbers:

| Legacy V6.2 (`task_run.py:158-172`) | V8 (`compose/v8/config.py:63-65`) |
| --- | --- |
| `router.top_m: 8` | `historical_top_m: 8` |
| `teacher.top_k_for_pair: 4` | `pair_top_k_single: 4` |
| `teacher.max_pairs: 6` | `max_pairs: 6` |
| `residual.tau_res: 2.0` | **removed** — no NLL threshold exists |

So V8's M=8, K_s=4 and C(4,2)=6 are the repo's established budget, and the
method's actual delta is narrower and sharper than "a new teacher": **the same
search shape, decided by task correctness instead of NLL**, plus the multi-key
pool and the per-sample gradient gating that make reuse trainable without
touching history.

**The legacy teacher is the concrete thing V8 fixes.** Legacy S2 ranks singles
by `loss + lambda_expert` and pairs by raw NLL delta
(`task_run.py:1088-1096`, `:1110-1135`) and then lets S3 split residual samples
on a global `tau_res = 2.0`. A sample whose correct answer the pool already
produces, but with high NLL, is sent to a new expert; a sample with low NLL and a
wrong answer is treated as solved. V8's causal-chain Q1 is exactly the question
this cannot answer, which is why V8-A measures it directly.

V7's own adjacent infrastructure that V8 reuses rather than reinvents: `compose/eval/nll_eval.py` (teacher-forced answer
NLL, whose forward body V8's `AnswerNLLScorer._forward` reproduces exactly — see
§13's seed verification, which proves it numerically) and the pair diagnostic
`v7_final_pool_pair_upper_val256_20260907`, which contains 253 pairs × 256
samples = 64,768 precomputed pair NLLs plus V7's per-sample generated answers.
V8 uses those as a *seed* and spot-checks them live before trusting them.

### 3.6 The one V7 restriction V8 had to work around

`compose/eval/eval_task.py:304-305` raises
`"V7 validation selection must contain two distinct experts"`. V8's policy needs
per-sample cardinality 0 (BaseOnly) / 1 (Reuse1) / 2 (Reuse2). The restriction is
a property of the V7 *harness*, not of the model — `ComposeLinear` already
handles every cardinality through `ComposeSelection`. Rather than fork model
code, V8 added `compose/v8/generate.py`, a generation harness that builds a
per-sample `ComposeSelection` itself. This is the BLOCKER recorded in PART 48
form: **BLOCKER** = the V7 evaluator cannot express V8's cardinality; **WHY** =
its cardinality-2 assertion; **IMPACT** = no official per-sample metric could be
computed for V8 policy routes; **PROPOSED MINIMAL FIX** = a harness that emits
the selection directly, leaving `compose/eval/` untouched (implemented, and the
V7 file is byte-unchanged).

---

## 4. V7 → V8 Mapping

The full table is in `docs/reports/V7_TO_V8_IMPLEMENTATION_MAPPING.md` (Phase 1
deliverable, written before any V8 code). Summary of the substantive deltas:

| Component | V7 behaviour | V8 behaviour | Implementation |
| --- | --- | --- | --- |
| Keys per expert | 1 (key id == expert id) | N (1 origin + M task-alias) | `compose/v8/pool.py` |
| Router | `topk` over keys == topk over experts | score every key, aggregate per expert by **max**, then Top-2 | `compose/v8/routing.py` |
| Reuse decision | none (routing is unconditional) | Answer-Supervised Expert Teacher decides `solved` by **task correctness** | `compose/v8/teacher.py` |
| Capability signal | answer NLL | task metric (`M`) decides; NLL only ranks/ties/diagnoses | `compose/v8/metric_adapter.py`, `config.py:66-85` |
| Sample states | n/a | BaseOnly / Reuse1 / Reuse2 / Residual | `compose/v8/selection.py` |
| Gradient scope | one task's experts | candidate LoRA + this task's alias keys only, per-sample gated | `compose/v8/gating.py` |
| Inference | fixed/2-expert route | router → composition, GT never read | `compose/v8/inference.py` |

---

## 5. V8 Architecture As Implemented

```
full task data
  → fixed query (V7's, verbatim, zero params)
  → STEP A  base evaluation                → BaseOnly  (selected_set = [], stop)
  → STEP B  historical Top-M recall (M=8)
  → STEP C  single-expert evaluation
        any single solved → best_single = lexicographic_min(-metric, nll, expert_id)
                          → Reuse1, STOP (no pair search)
  → only if no single solved: pair search over shortlist K_s=4 → C(4,2)=6 pairs
        a legal pair that also clears the marginal-metric rule → Reuse2
  → else Residual: keep the best historical *context* by
        (highest metric, lowest NLL, smaller cardinality) even though nothing solved
  → alias-key learning (lazy) + residual candidate learning (gated)
  → pruning → commit → multi-key inference
```

The states are persisted per sample (`teacher_result.json`) with their
`decision_reason`, so every routing decision in this report can be re-derived
from disk.

---

## 6. Correctness + NLL: The Two Separated Signals

The specification's hard rule — *"绝对禁止：Answer NLL < 某个统一阈值 => solved"* —
is enforced structurally, not by convention:

* `compose/v8/config.py:66` `solved_signal: str = "task_metric"`, and
  `config.py:79-85` raises if `nll_use_as_solved_threshold` is set or
  `solved_signal` is anything else. The ban is re-checked on the composed config
  at `config.py:332`.
* `compose/v8/metric_adapter.py` is the only producer of `solved`; the teacher
  calls it and never compares an NLL to a threshold.
* NLL's permitted roles are exactly: ranking among *already solved* experts,
  a soft confidence, pair marginal contribution, residual context, tie-break,
  diagnostics.

Tests: `test_09_low_nll_with_wrong_prediction_must_not_become_solved` (a low-NLL
wrong answer stays unsolved), `test_10_correct_prediction_with_higher_nll_remains_solved`
(a high-NLL correct answer stays solved), and
`test_teacher_rejects_nll_solved_threshold_configuration` (the configuration
cannot even be constructed).

---

## 7. Task Metric Adapters

| Task | Dataset | Official metric | Per-sample capability signal | `solved` definition | Proxy? | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ImageNet-R | Accuracy | exact match (case-insensitive) | value ≥ 1.0 | no | — |
| 1 | ArxivQA | Accuracy | exact match | value ≥ 1.0 | no | — |
| 3 | IconQA | Accuracy | exact match | value ≥ 1.0 | no | — |
| 4 | CLEVR-Math | Accuracy | exact match | value ≥ 1.0 | no | — |
| 2 | VizWiz | Average | **not decomposable** | — | `v8_caption_capability_proxy_v1` | corpus-level COCO scoring only |
| 5 | Flickr30k | Average | **not decomposable** | — | `v8_caption_capability_proxy_v1` | corpus-level COCO scoring only |

`require_decomposable(task)` **raises** for tasks 2 and 5 rather than inventing a
per-sample substitute. Consequence, stated plainly: **V8-A can only be run on the
four exact-match tasks.** That is a property of the official metric, not a
choice made to make results look better.

---

## 8. Multi-Key Pool

`compose/v8/pool.py` implements `1 Expert : N Keys`.

* Key ids are `e{expert}_t{task}_{type}`, so the mapping is reversible and
  auditable from the id alone.
* Every expert keeps exactly one `origin` key — the V7 key, migrated by
  `MultiKeyExpertPool.load_v7_pool` — plus zero or more `task_alias` keys.
* Lifecycle: `candidate` → `historical` → `pruned`. The migration asserts
  `num_alias_keys == 0` for a V7 pool, so a migrated pool cannot silently claim
  alias structure it does not have.
* Validation (`pool.validate()`) rejects an alias key sitting on the origin task
  and rejects a second origin key, so "which task created this expert" stays
  single-valued.
* Routing aggregation is **max over an expert's active keys**
  (`compose/v8/routing.py:score_matrix`), and `active_key_ids(excluded_experts)`
  can exclude a subset of experts without deleting anything — which is how the
  `--history-only` scope is implemented.

---

## 9. Training and Gradient Gating

`compose/v8/gating.py` — V8 does **not** filter the dataset (PART 46 item 4 is
explicitly avoided). Every current-task sample keeps its reuse decision, and the
decision is expressed as a per-sample weight:

```
L_answer_residual = sum_i r_i * L_answer_i / max(sum_i r_i, 1)
```

| State | Candidate LoRA gradient | This task's alias-key gradient |
| --- | --- | --- |
| BaseOnly | zero | zero |
| Reuse1 | zero | zero |
| Reuse2 | zero | zero |
| Residual | **non-zero** | non-zero where support exists |

Tests: `test_16/17/18` (zero candidate gradient for BaseOnly/Reuse1/Reuse2),
`test_19` (non-zero for Residual), and — the one that matters most —
`test_30_gradient_leakage_on_mixed_baseonly_reuse_residual_batch`, which runs a
mixed batch and asserts the candidate gradient equals the residual-only batch's
gradient. `gradient_leakage_probe` (`gating.py:335`) reports `max_abs_delta`.

The trainable-parameter whitelist is `audit_trainable_parameters`
(`gating.py:112`), which decides membership by **tensor identity** (mirroring
`compose/v7/hf_trainer.py:265`) so a renamed parameter cannot slip through, and
`enforce_freeze_policy` (`gating.py:162`) applies it to real tensors and returns
`audit.ok`, which must be true before training starts.

### 9.1 The alias-key objective, and a defect the integration test found

Alias keys are trained with `L_key = lambda_pos * L_pos + lambda_rank * L_rank`
(`key_learning.py:202`, `lambda_pos = 1.0`, `lambda_rank = 0.1`,
`ranking_margin = 0.2`). `L_pos` pulls the key toward its positive queries'
centroid; `L_rank` is a margin hinge that pushes it away from the
highest-scoring *other* key on each sample where it is NEGATIVE.

Writing the chained integration test exposed a real defect here, which is
recorded rather than smoothed over:

* **Symptom.** `L_key` had a gradient of ~0 (measured norm `5.16e-08`, float
  noise) on a freshly created alias key, for both a 3-support and a 1-support
  key. Training could not have moved the keys at all.
* **Cause.** Two facts compounded. (i) `L_rank` was built from `float(...)`
  values, so it was a *constant* with no gradient w.r.t. any key. (ii) The only
  differentiable term left, `L_pos = mean(1 - cos(q_i, k))`, has its exact
  stationary point at `Normalize(mean(q_i))` — precisely the centroid
  `create_alias_keys` (`key_learning.py:128`) initialises a key with. The
  objective was therefore inert at initialisation by construction.
* **Fix (commit `050ee42`).** The hinge now uses `(query @ key)` for the current
  key, so the gradient stays attached, and the tensor similarity of the hardest
  competitor — which remains **detached**, because a historical key must receive
  no update, not even an incidental one from being pushed away. Numeric loss
  values are unchanged; only the gradient path changed.
* **Evidence after the fix.** The same probe now measures gradient norms
  `3.3e-02` and `4.7e-02`, an SGD step moves exactly the two alias keys and no
  historical key, and every historical key still holds `grad is None`.
  `test_alias_key_ranking_loss_is_a_gradient_carrying_hinge` pins the hinge
  arithmetic on exact basis-vector geometry (violated margin `= 1.2` exactly,
  satisfied margin `= 0.0` exactly, gradient reaches the current key only).

Had this not been found, V8-B would have run with a key-learning stage that
silently did nothing and the resulting numbers would have been attributed to
"key geometry is insufficient" rather than to a bug.

### 9.2 The three-valued rule, and a second defect found the same way

The teacher scores the **pool-wide candidate list** on every unsolved sample:
`teacher.py` STEP C loops over `candidates`, the union of every pending
sample's recall, not over the sample's own Top-M. A sample's `recall` is thus
its own 8 experts while `tested_singles` is the union actually scored — in the
formal Task 4 run, 10 experts. `recall ⊆ tested_singles`, and only
`tested_singles` carries evidence.

The Reuse1 branch built its labels over `recall` alone and then ran
`targets.setdefault(expert, NEGATIVE)` over `tested_singles`. Both consequences
are spec violations:

* an expert **outside** the recall that solved the sample is an alternative
  solver and must be `IGNORE` — PART 9's "禁止：E5 = negative", because that
  label pushes an alias key away from a query its expert demonstrably solves;
* if the *best* single came from outside the recall, the expert the teacher
  actually selected would be labelled `NEGATIVE` instead of `POSITIVE`.

* **Measured, on real data.** In the smoke run (`0911_v8a_smoke/task4`), record
  `v7_t4_val_10` recalled `[19,17,16,18,14,15,13,0]` and was scored on
  `[0,3,13,14,15,16,17,18,19]`; expert 3 solved the sample, was not recalled,
  and was recorded `negative`. One mismatch in 3 Reuse1 samples
  (`samples_with_out_of_recall_solver: 1`) — the forbidden label occurred, it
  was not merely latent.
* **Measured on the formal Task 4 run** (`0911_v8a_all/task4`, 256 samples,
  `v8a_label_audit.py`): **33 mismatched labels**; **28 samples** where an
  expert outside that sample's Top-8 solved it; and **3 samples where the
  *selected* expert was outside the sample's own recall** — the case that was
  hypothetical when this section was first written. Those 3 lost the `POSITIVE`
  label of the expert the teacher had actually selected, so any label-based
  consumer (the runner's own recall curve, the alias-key positive support) sees
  two of them as having no solved expert at all. The routing decision itself is
  untouched, and the solver set is now read from `single_values` in
  `v8a_recall_audit.py` precisely so the reported recall numbers do not inherit
  the mislabelling.
* **Fix.** The rule now runs over `sorted(tested_singles ∪ recall)`, which is
  total over everything that was scored. `test_01`–`test_30` are unchanged
  (in those fixtures `tested_singles == recall`); the new guard
  `test_reuse1_targets_are_total_over_every_scored_expert_not_just_the_recall`
  pins both edge cases — the out-of-recall solver becoming `IGNORE`, and the
  out-of-recall *selected* expert keeping its `POSITIVE`.
* **Blast radius.** `key_targets` is written *after* `_decide` has chosen the
  state and the route, so it does not enter the state, the selected set, the
  achieved metric or any V8-A routing number: the campaign's V8-A metrics are
  unaffected in either direction. What it does touch is (i) the key objective's
  IGNORE/NEGATIVE sets, and (ii) POSITIVE-derived summaries — the runner's
  `teacher_expert_recall_at_k` and alias-key support counts. Both are now
  computed in the report from primary evidence instead of labels:
  `v8a_recall_audit.py` reads `single_values` + `selected_experts`, and
  `v8a_label_audit.py` reports the mismatches per run, so every affected number
  is re-derived by a documented rule from stored evidence rather than patched.

Note on artifact provenance: run 1 of the formal campaign was already in flight
when this fix landed, so its `teacher_result.json` was written by the pre-fix
code; runs 2–4 use the fixed rule. The audit above is what quantifies the
difference per run — no artifact is rewritten in place.

### 9.3 A third defect, left unfixed on purpose until the campaign ends

`compose/v8/audit.py:251-255` and `:258-261` phrase two of the four outcome
cases in terms of **alias keys** — CASE_B is "reusable experts exist and are
recalled, yet the alias keys do not convert that into a better metric", and
CASE_D says "check train-vs-validation recall for alias-key overfitting". This
is wrong for V8-A, which by construction trains nothing and creates no alias
key: any V8-A outcome can only be about the *origin* keys and the fixed-query
geometry.

This matters because `diagnose()` is a **runtime** call, not an offline
analyser: `compose/experiments/v8_task_run.py:818` writes its verdict straight
into `analysis.json:diagnosis`. Run 1's artefact therefore contains

```json
"case": "CASE_D_OR_IMPROVED",
"interpretation": "recall is high and the metric improved; check train-vs-validation recall for alias-key overfitting before concluding"
```

The `case` is defensible (recall is indeed high and the metric did improve); the
*reason* names a mechanism that does not exist in this experiment.

* **BLOCKER.** Not a blocker — the verdict is correct, the mechanism named in the
  prose is not.
* **WHY it is not fixed yet.** Runs 3 and 4 are in flight. `diagnose()` is
  executed by the runner, so editing it now would mean runs 3–4 and run 1
  produced their artefacts under different code, which is precisely the
  provenance break this report avoids everywhere else. The fix is deferred to
  after the campaign and is applied to the module only; the existing artefacts
  are not rewritten.
* **IMPACT while deferred.** Readers of `analysis.json` see wording that
  over-attributes to a mechanism V8-A does not exercise. Every number in the
  file is unaffected — this is prose inside a summary string. The report's own
  §14–§16 and §23 give the corrected reading (origin-key geometry and selection
  ordering, not alias-key overfitting).
* **PROPOSED MINIMAL FIX** (post-campaign). Reword CASE_B to "the origin keys
  already rank a solver first, so the remaining loss is in route selection and
  composition rather than in recall", and CASE_D to "check that the win is not
  an artefact of selection ordering before concluding"; then assert in
  `tests/compose` that no `interpretation` string mentions alias keys for a run
  whose `run_config` reports no training. The second half is the part that
  prevents the defect from recurring.

### 9.4 Two defects found in the campaign's own analysis tooling

Both were found by cross-checking against `analysis.json` rather than by
inspection, and both are recorded because they changed a number the report
would otherwise have published.

**(a) `v8a_scope_gap.py` counted `BaseOnly` samples as unresolved.** The first
version asked only `selected_experts != []`, but a sample whose answer the frozen
base already gets right is `BaseOnly` with an *empty* route by design (PART 8:
base solves → stop). Those 42 samples were therefore binned as "unsolved in
either scope", producing `unresolved = 57` where the truth is 14. The tell was
that `historical_reuse` disagreed with the history run's own solve count in
`analysis.json` (115). The fix makes `_solved()` return true for `BaseOnly`, and
the script now reports `bucket_sum`, which must equal `samples_compared`, plus
`solved_in_all_scope`, which must equal the runner's count — the two checks are
in the artefact itself rather than in this prose. A second, smaller error
surfaced in the same fix: the `else` branch still swallowed a history-only solve,
hiding the very anomaly (§14) it was meant to surface; it now has its own branch.

**(b) The orchestrator produced two byte-identical case-study files under
misleading names.** `v8a_cases.py` always joins *both* run roots — it is not
scope-parameterised — so running it once per scope wrote `cases_all_task4.json`
and `cases_history_task4.json` with identical content. Anyone comparing them
would conclude "the case mix is the same in both scopes", which is true but only
because the file ignores the scope argument. It is now run once per task as
`cases_task{N}.json`.

Neither defect touches the V8 method, the frozen pool, or any run artefact: both
are in post-hoc analysis scripts that read finished runs and write new files.

---

## 10. Freeze Audit

### 10.1 Code-level enforcement

`enforce_freeze_policy` freezes the base backbone, every historical LoRA expert
and every historical committed key, and sets trainable **only** this task's
candidate experts and this task's alias keys. A frozen historical key that still
requests gradient is a hard failure, not a warning.

### 10.2 Exact-tensor comparison

`capture_frozen_ledger` / `verify_frozen_ledger` (`gating.py:261`, `:285`)
checksum every frozen key and every frozen LoRA tensor, run a real optimizer, and
require `status == "UNCHANGED"`. `test_04` additionally proves the check has
teeth: after the ledger passes, mutating `pool.keys["e0_t0_origin"]` by 0.5 makes
`verify_frozen_ledger` raise `GradientGatingError`.

### 10.3 Frozen artefacts of this campaign

The V8-A runs perform **no training at all** (no optimizer is constructed, no
`.backward()` is called), so there is no post-training tensor to compare — that
comparison is part of V8-B (§19). What can be and was verified is that the frozen
artefacts are bit-identical across the whole campaign:

| Parameter | Expected | Observed (after V8-A) | Checksum before | Checksum after | PASS? |
| --- | --- | --- | --- | --- | --- |
| Base backbone | unchanged | not written by any V8 path | (V7 weights, untouched) | same | PASS |
| Historical LoRA (`compose_experts.bin`) | unchanged | `b9f68186…3012b5` | `b9f68186…3012b5` | `b9f68186…3012b5` | PASS |
| Origin keys (`v7_keys.pt`) | unchanged | `337891cf…c0bd9b` | `337891cf…c0bd9b` | `337891cf…c0bd9b` | PASS |
| Historical alias keys | none exist in a migrated V7 pool | `num_alias_keys == 0` asserted at load | n/a | n/a | PASS |
| Current candidates | n/a in V8-A (no training) | no candidate LoRA created | n/a | n/a | NOT APPLICABLE |

The "checksum before" column is V7's own recorded fingerprint from
2026-09-07; the "after" column was computed on 2026-09-11 after the V8-A runs.

---

## 11. Test Results

```
$ pytest tests/ -q
608 passed, 3 warnings, 8 subtests passed in 75.76s (0:01:15)
```

Split:

| Suite | Collected | Result |
| --- | --- | --- |
| `tests/compose/test_v8_answer_supervised_multikey.py` (TEST 01–30) | 44 | all pass |
| `tests/compose/test_v8_generation_harness.py` | 16 | all pass |
| V7 regression (rest of `tests/compose/`) | 548 | all pass |

44 = 30 names from PART 15, plus the integration tests of §9.1, §10.2 and §12,
plus `test_22b_current_task_expert_gets_no_alias_key` — the regression test added
when §17 exposed the `create_alias_keys` defect. Nothing in that file is
parameterised, so collected and defined counts agree. The three rows sum to 608,
which is the suite total. An earlier draft said 42 for the first row and 607 for
the total; that draft did not add up (42 + 548 + 16 = 606), and the corrected
numbers here are the ones `pytest --collect-only` reports.

The 30 named tests from PART 15 all exist and all run. Several are parameterised
over states or configs, which is why the file collects 42 tests for 30 names; the
remainder are the integration tests described in §9.1, §12 and §10.2.

Run with no deselection: nothing is skipped, xfailed or marked slow, including
the tests that load the real committed V7 pool (`V7_FORMAL_KEYS` is present on
this machine, so those run against the real artefact rather than being skipped).

### 11.1 TEST 01–30, and what each one actually pins

| Test | Pins |
| --- | --- |
| 01 | fixed query has zero trainable parameters (contract hash asserted) |
| 02 | historical LoRA has `requires_grad == False` |
| 03 | historical LoRA checksum unchanged after a real optimizer run |
| 04 | historical committed key unchanged, and a 0.5 mutation is *detected* |
| 05 | one expert can own several keys (the `1 Expert : N Keys` premise) |
| 06 | router aggregates key scores per expert by max, not by sum |
| 07 | one expert cannot occupy two Top-2 slots |
| 08 | `solved` is decided by the task metric, not by any NLL |
| 09 | a **low-NLL wrong** answer is not solved — the PART 3 prohibition, in code |
| 10 | a **correct** answer with higher NLL stays solved |
| 11 | multiple solved experts → best is POSITIVE, the rest IGNORE, unsolved NEGATIVE |
| 12 | pair search is not executed once a single solved (`decision_reason`) |
| 13 | a pair cannot be chosen on NLL improvement alone |
| 14 | a pair must clear the task-specific marginal criterion |
| 15 | base-solved → `selected_set == []` and stop |
| 16–18 | candidate LoRA gradient is exactly zero for BaseOnly / Reuse1 / Reuse2 |
| 19 | candidate LoRA gradient is non-zero for Residual |
| 20 | a Reuse sample can still train a historical *alias* key |
| 21 | the alternative solved expert receives no negative gradient |
| 22 | a zero-support historical expert gets no alias key (lazy creation) |
| 23 | alias-key centroid initialisation is the normalised positive mean |
| 24 | the inference path never reads a ground-truth answer |
| 25 | teacher-cache reload is deterministic |
| 26 | resume preserves the expert/key mapping |
| 27 | V7 checkpoint migration creates origin keys (no alias keys) correctly |
| 28 | the task-0 no-history path runs |
| 29 | the original V7 tests still pass |
| 30 | gradient leakage on a mixed BaseOnly/Reuse/Residual batch is zero |

The suite is not a smoke test: 09/10 are the two directions of the
correctness-vs-NLL separation, 13/14 are the two directions of the pair-legality
rule, and 16–19 plus 30 are the four gating states and their interaction.

---

## 12. V7 Regression and Task 0 Parity

**V7 regression**: 548 tests under `tests/compose/` that predate this work still
pass. `git diff --stat 9ff2b28..HEAD` reports **34 files changed, 11,347
insertions(+), 0 deletions(-)** — every change is a new file (`compose/v8/*`,
`compose/experiments/v8*.py`, the two new test files) plus additive edits to
`CHANGELOG.md` and `docs/module_status.md`. Nothing under `compose/v7/`,
`compose/adapters/`, `compose/eval/` or `llava/` was modified, which is why
`test_29_v7_original_tests_still_pass` passes by construction. Zero deleted
lines is the mechanical statement of "V8 does not break V7": no V7 code path was
edited, only new modules were added beside it.

**Task 0 parity** (`compose/experiments/v8_task0_parity.py`). Task 0 is the one
place where V7 and V8 are *supposed* to agree: there is no history to reuse, the
candidate set is the same four experts, and V8's multi-key aggregation degenerates
to one key per expert — i.e. V7's plain inner product. The check re-runs V8's
router over the task-0 validation queries with only the task-0 experts visible
and compares against the routing V7 recorded in its own
`task0/selection_plan.json`:

| Route | V7 count | V8 count |
| --- | --- | --- |
| `{0,1}` | 2 | 2 |
| `{0,3}` | 135 | 135 |
| `{1,2}` | 109 | 109 |
| `{2,3}` | 10 | 10 |

**Route multiset identical — MATCH.**

### 12.1 Answer parity: two crashes, one real defect, retry queued

The route check needs no generation, so it stands on its own. The *answer*
check — regenerate each sample through V8's engine under the identical route and
diff the text against the answers V7 actually produced — has now been attempted
twice and **still has no result**, and the report says so rather than leaving a
blank. The second attempt found a genuine code defect, which is the more useful
outcome and is recorded here in full.

| Attempt | When | Outcome |
| --- | --- | --- |
| 1, route only | 07:56 | `task0_parity.json` written with `route_parity` and no `answer_parity` — this is the MATCH above, and it is unaffected by what follows |
| 2, `--generate` | 07:56 | `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR` inside `prepare_inputs_labels_for_multimodal` (the vision conv): the shared device was too full for cuDNN to get a workspace |
| 3, `--generate` | 14:51 | Got an **exclusively-owned** device (the exclusivity gate below worked — the model loaded and all 256 answers were generated), then died at scoring: `KeyError: 'v7_t0_val_0'` |

#### The defect the third attempt exposed

`v8_task0_parity.py:87` sets `diagnostic = Path(args.diagnostic_root) / "task0"`,
which is right for the file it reads there — `task0/selection_plan.json`, the V7
pair plan. Line 158 then composed the answers path as
`diagnostic / "generation_accuracy" / "task0" / "actual" / "answers.jsonl"`,
producing

```
<root>/task0/generation_accuracy/task0/actual/answers.jsonl
```

while the file V7 actually wrote is at

```
<root>/generation_accuracy/task0/actual/answers.jsonl
```

— one `task0` too many. The convention is the run root, not the per-task
directory: the same layout is read correctly at
`compose/experiments/v8_task_run.py:800` and `compose/experiments/v8a_cases.py:97`.

**Why it failed silently rather than loudly, which is the part worth keeping.**
`_jsonl()` returns `[]` when the path is not a file, so `v7_answers` became the
empty dict; the very next line computes
`compared = [s for s in sample_ids if s in v7_answers]`, which would have been
`[]` — the code *had* the evidence, one line before it needed it, and did nothing
with it. The failure surfaced 30 lines later as a bare `KeyError` on the first
sample id, which names neither the missing file nor the wrong path. An empty diff
and a perfect diff are indistinguishable downstream of that line.

#### Fix applied

`compose/experiments/v8_task0_parity.py`, three changes, and the third is the one
that would have saved both hours:

1. The path is built from `Path(args.diagnostic_root)` — the run root — with a
   comment recording why the two must not be composed.
2. A missing answers file now raises `FileNotFoundError` **naming the path**, so
   the next occurrence says what is wrong in the message rather than in a
   traceback.
3. A zero-overlap answers file now raises `ValueError` instead of scoring
   nothing: "scoring it would silently compare nothing". The `compared` list that
   already existed is now load-bearing.

Verified offline without a GPU: the corrected path resolves to
`…/v7_final_pool_pair_upper_val256_20260907/generation_accuracy/task0/actual/answers.jsonl`,
which exists and holds 256 rows, and `task0/selection_plan.json` still resolves
where the pre-existing code expects it. The module parses clean.

#### Retry 3, queued

`experiments/runs/0911_v8a_formal/run_parity_retry.sh` (pid 567830) waits for the
incumbent deferred runner to exit, then for a device with **no compute process
belonging to another user** and >= 20,000 MiB free, then re-runs the same
command with `--generate`. It is a separate script rather than an edit to
`run_deferred_gpu_work.sh` precisely because that script is currently executing
and bash re-reads a running script from disk.

The gate is not incidental. Attempt 2's crash was on a device shared with another
user's job, where cuDNN could not get a workspace; and seizing a device that
someone else is merely idling on would risk OOM-ing their work as well as ours.
The measured headroom makes the condition workable rather than a demand for an
empty machine: §21.1 puts peak allocator memory at 14.83 GiB and resident VRAM at
16,910 MiB.

**Status.** §12.1's conclusion remains "route parity verified, answer parity
pending" until `task0_parity.json` acquires an `answer_parity` key. This is a
**BLOCKER-class gap in evidence**, and since attempt 3 it is also a
**fixed-and-unverified code change**: the parity harness's path bug is repaired
and the repair is verified structurally (the path exists, the guards raise), but
no run has yet executed the repaired code end to end. No claim in this report
depends on answer parity — it is a cross-check on §5's claim that V8's engine
reproduces V7's answers under V7's route — and the report will say so either way
once the run lands.

---

## 13. V8-A Setup

V8-A is the low-cost validation the specification asks for first (PART 5: "优先
利用已有 V7 Expert Pool 做低成本 V8-A 验证"): it applies the *full* V8 teacher and
policy to a task's validation split using the existing frozen pool, so the method's
core assumption can be tested without training anything.

| Item | Value |
| --- | --- |
| Input checkpoint | `…/v7_gpu01_cached_query_formal_20260903/task5/committed` (final pool, 23 experts) |
| Tasks | 4 (CLEVR-Math), 3 (IconQA) |
| Samples per task | 256 (the full `val_full.json` split — no subsampling) |
| Recall budget M | 8 |
| Pair shortlist K_s | 4 (→ 6 pairs) |
| Scopes | `all` (pool as committed) and `history-only` (the task's own and later experts excluded from routing) |
| Teacher cache | seeded from 64,768 frozen V7 pair NLLs, live-verified before use |
| Hardware | one GPU (RTX 4090, 24 GB), `CUDA_VISIBLE_DEVICES=3` |
| Runner | `compose/experiments/v8_task_run.py` |

**Why both scopes.** In the real continual setting, the pool at task `t` holds the
experts of tasks `< t`; the task's own experts do not exist yet. An `all`-scope run
against a pool that happens to contain task 4's own experts would credit V8 with
reuse it never had the chance to perform. `--history-only` excludes those experts
(their keys are excluded from routing, not deleted). The pair of runs therefore
brackets the truth: `all` is an upper bound, `history-only` is the honest continual
setting, and the gap between them is itself a measurement (§14).

**Metric scope.** Only exact-match tasks are eligible (§7): 0, 1, 3, 4. Tasks 2
and 5 use corpus-level COCO scoring with no per-sample decomposition, so
`require_decomposable` refuses them and no proxy was substituted for the
per-sample `solved` decision.

**No training.** V8-A trains nothing: no optimizer is constructed and no backward
pass is run. It measures *whether the historical pool already contains the
capability*, which is Q1 of the causal chain and the precondition for everything
downstream.

---

## 14. Teacher Capability Analysis: can the old experts be reused?

The question the whole method rests on (PART 49 Q1). Every number below is
`states` / `state_rates` from `task{N}/analysis.json`, i.e. the teacher's verdict
distribution over the full 256-sample validation split — not a subsample, and
not a proxy.

| Task | Scope | BaseOnly | solved by a single expert | solved only by a pair | unresolved (Residual) |
| --- | --- | --- | --- | --- | --- |
| 4 (CLEVR-Math) | all-experts | 42 (16.41 %) | 178 (69.53 %) | 21 (8.20 %) | 15 (5.86 %) |
| 4 (CLEVR-Math) | history-only | 42 (16.41 %) | **71 (27.73 %)** | **2 (0.78 %)** | **141 (55.08 %)** |
| 3 (IconQA) | all-experts | 46 (17.97 %) | 160 (62.50 %) | 17 (6.64 %) | 33 (12.89 %) |
| 3 (IconQA) | history-only | 46 (17.97 %) | **37 (14.45 %)** | **3 (1.17 %)** | **170 (66.41 %)** |

Read the `all-experts` rows with the scope caveat attached: experts 16–19 are
Task 4's *own* experts and 12–15 are Task 3's, and none of them existed when
their task was learned, so "reused" there includes self-reuse. The history-only
rows are the honest continual-learning numbers, and §20 splits the two.

**This is the most important comparison in the V8-A campaign, and it goes
against the all-experts headline — on both tasks.** With the task's own experts
removed, the number of samples any historical expert can solve drops from 178 to
71 on Task 4 (69.53 % → 27.73 %) and from 160 to 37 on Task 3 (62.50 % →
14.45 %); pair-only solves collapse from 21 to 2 and from 17 to 3; the Residual
set grows from 15 to 141 and from 33 to 170. In other words **four of every five
samples V8-A appeared to "reuse" on Task 3 were being solved by the task's own
experts** — which is not reuse at all, because those experts do not exist at
learning time. The `v8a_scope_gap.py` join gives the exact split for both:

| Quantity | Task 4 | Task 3 | Meaning |
| --- | --- | --- | --- |
| Historical reuse (solved with the historical pool) | **115 (44.92 %)** | **86 (33.59 %)** | reachable from the pool as it stood at learning time |
| — of which base-only (needs no expert at all) | 42 (16.41 %) | 46 (17.97 %) | the base already answers correctly |
| — of which a historical expert route | **73 (28.52 %)** | **40 (15.63 %)** | genuine cross-task reuse |
| Self-reuse | 127 (49.61 %) | 138 (53.91 %) | solved only with the task's own (or a later task's) experts |
| Unresolved in both scopes | 14 (5.47 %) | 32 (12.50 %) | no expert in the whole pool solves it |

The three buckets sum to 256 exactly on both tasks (`bucket_sum ==
samples_compared`), and the scope totals reproduce the runners' own counts —
`solved_in_all_scope` is 241 and 223, `historical_reuse` is 115 and 86, exactly
the `analysis.json` solve counts for the four runs. Those cross-checks are why
this table is the one the report quotes: the script's first version counted
`BaseOnly` samples as unresolved and reported 57 unresolved samples instead of
14 on Task 4, and the disagreement with `analysis.json` is what caught it.

Two things follow, and the second is the one that matters for the verdict.

1. **The reuse effect is real but small, and it shrinks with task index.** The
   genuine cross-task expert route covers 28.52 % of Task 4 and 15.63 % of
   Task 3. Self-reuse *exceeds* historical reuse on Task 3 (53.91 % vs 33.59 %)
   whereas on Task 4 they are comparable (49.61 % vs 44.92 %).
2. **On both tasks the majority of the "historical reuse" number is the frozen
   base answering correctly on its own** — 42 of 115 on Task 4, 46 of 86 on
   Task 3. Strip that out and the historical expert pool contributes a genuine
   cross-task route to 28.5 % and 15.6 % of samples respectively. The correct
   baseline for any V8-A claim is therefore the base-only solve rate (16.41 % /
   17.97 %), not zero and not V7 (§15).

The residual sets still keep a best historical context (§17): 15 samples on
Task 4 (11 single, 4 pair) and 33 on Task 3 (13 single, 20 pair).

One cross-scope anomaly is reported rather than smoothed, and it occurs **exactly
once on each task**: `v7_t4_val_211` and `v7_t3_val_189` are each solved in the
history-only scope but *not* in the all-experts scope, which at first looks
impossible — the history pool is a strict subset. It is not a counter bug and it
is verified from both runs' artefacts, by two different mechanisms:

* `v7_t4_val_211`: in the history scope the pair `{13, 0}` solved the sample,
  while in the all-experts scope Task 4's own experts occupy ranks 1–4 of the
  Top-8 window (`[19, 17, 16, 18, 15, 14, 13, 0]`), so `{13, 0}` never entered
  the K_s = 4 pair shortlist and was never composed.
* `v7_t3_val_189`: in the all-experts scope the Top-8 window is
  `[15, 14, 13, 12, 0, 3, 19, 17]` — five experts that do not exist at learning
  time crowd out every one of the seven that do, and the sample ends in
  `Residual` with no route at all. In the history scope the window is
  `[0, 3, 2, 1, 10, 8, 4, 5]` and expert **7**, which appears in neither window,
  is selected and solves it — because STEP C scores every recalled candidate on
  every pending sample, not only the ones inside that sample's window (§23).

A **wider pool can hide a solver** — from the pair shortlist in the first case,
from the window itself in the second. That is the same crowding effect §23
identifies as a bottleneck, and the reason this report does not treat
`historical ⊆ all` as an identity. `solved_in_history_only_but_not_all` is 1 on
each task and 0 in every other direction; `base_state_disagreements` is empty on
both, i.e. the base's own verdict never changed between scopes.

---

## 15. V8-A Routing Results

**Table 1 — official metric.** `V7 Metric` is the V7 actual route on the same
split (`v7_final_pool_pair_upper_val256_20260907`), `Teacher UB` is that
campaign's answer-NLL oracle pair (the ceiling V8-A can reach without training),
`V8 Metric` is the V8 policy's route replayed through the frozen model and
scored by the official evaluator, and `GapClosed = (V8 − V7) / (UB − V7)`.

`BaseOnly` is included because it is the only baseline that is comparable across
scopes: it is what the frozen base scores with *no* expert route at all, so it is
the number a "reuse-only" policy has to beat.

| Task | Scope | BaseOnly | V7 Metric | Teacher UB | V8 Metric | V8 − V7 | V8 − BaseOnly | GapClosed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 (CLEVR-Math) | all-experts | 16.41 | 67.97 | 95.31 | **94.14** | **+26.17** | +77.73 | **0.957** |
| 4 (CLEVR-Math) | history-only | 16.41 | 67.97 | 95.31 | 44.92 | −23.05 | **+28.51** | −0.843 |
| 3 (IconQA) | all-experts | 17.97 | 67.97 | 85.94 | **87.11** | **+19.14** | +69.14 | 1.065 |
| 3 (IconQA) | history-only | 17.97 | 67.97 | 85.94 | 33.59 | −34.38 | **+15.62** | −1.913 |

**The history-only rows are not V8-vs-V7 results, and must never be quoted as
one.** The comparison is not like-for-like: V7's 67.97 comes from a router that
is *allowed* to use the task's own experts (16–19 / 12–15, which exist in the
pool precisely because that task has already been trained), while the
history-only V8 policy is forbidden from using them by construction. Those rows
measure **how far the frozen historical pool gets with no task-specific experts
available** — 44.92 on Task 4 and 33.59 on Task 3, i.e. 23 and 34 points below
what V7 achieves *with* them. Read that way they are among the campaign's most
useful numbers: they are the ceiling for "reuse only", and the gap between them
and 67.97 is the headroom a V8-B Candidate Expert would have to close. What they
are not is evidence that V8's routing is worse than V7's, because the two
policies were not given the same pool.

The `V8 − BaseOnly` column is the like-for-like comparison the method can
actually claim: against a policy that routes to nothing, V8-A's historical
routing adds **+28.51 points on Task 4 and +15.62 on Task 3**, and its
all-experts routing adds +77.73 and +69.14. Both are real gains — but the
all-experts column is inflated by self-reuse (§14), so the honest summary is
**+28.51 / +15.62 from reuse alone**, against +26.17 / +19.14 over V7 when the
task's own experts are allowed.

A note on GapClosed: on Task 3 / all-experts it exceeds 1 (1.065) because the V8
policy's 87.11 is *above* the 85.94 "teacher upper bound". That is not a
contradiction — the ceiling there is the **answer-NLL oracle pair** from the V7
diagnostic, a route chosen by ranking candidate pairs by ground-truth answer NLL.
V8's teacher ranks *within* the solved set by metric first and NLL only as a
tie-break, so it can find a route the NLL-only ranking does not. It is a small
piece of positive evidence for keeping `M` and `L` separate (§6) rather than
collapsing them into one score.

The teacher's own verdict count and the official scorer's count agree exactly on
all four runs (`reconciliation.difference == 0`: 241/223/115/86 samples solved by
the teacher, the same four numbers scored correct by the evaluator). That is the
check that the metric adapter used for `M` and the official metric used for the
headline number are the same function — and it is also what makes the
`BaseOnly` column above directly comparable with the metric columns.

**One quantified upper-bound caveat.** STEP C scores the pool-wide candidate
union (10 experts on Task 4, 18 on Task 3) rather than each sample's own Top-8,
so a sample can be solved using an expert its own recall window never contained.
That happened on 3 samples (1.17 %) in Task 4 / all-experts, 3 (1.17 %) in
Task 3 / all-experts and 11 (4.30 %) in Task 3 / history-only
(`selected_route_outside_own_recall_window` in `v8a_recall_audit.py`), so a
router restricted to each sample's own Top-8 would score at most
`(42 + 196) / 256 = 92.97 %` on Task 4 rather than 94.14 %, and
`(46 + 174) / 256 = 85.94 %` on Task 3 rather than 87.11 %. The union is the
teacher's *search* budget, not a leak of answers — but the numbers are upper
bounds and the size of each bound is a small, counted number, not "unknown".

**Table 2 — router recall, counted over solving-expert instances.** `Recall@k` is
the recall of the current single-key (`1 Expert : 1 Key`) router: the fraction of
**solving-expert instances** that fall inside the router's Top-k window, over the
solver set defined in §16. A sample solved by three experts contributes three
instances, so missing any one of them counts as lost capability. This is
`solver_recall_at_k` from `v8a_recall_audit.py`, the §16 tool.

An earlier revision of this table published those per-instance numbers under a
header that described them as "the fraction of *samples* with a solving expert
inside the Top-k". Those are different denominators. The alias-key counterfactual
is measured per sample, so rather than mix the two, this table is unambiguously
per-instance and the per-sample alias comparison is **Table 3 in §17**. The two
are not interchangeable and this report does not quote them interchangeably.

| Task | Scope | V7 R@1 | V7 R@2 | V7 R@4 | V7 R@8 | SetExact |
| --- | --- | --- | --- | --- | --- | --- |
| 4 (CLEVR-Math) | all-experts | 0.719 | 0.849 | 0.950 | 0.990 | *§17 prose* |
| 4 (CLEVR-Math) | history-only | **0.274** | **0.452** | **0.740** | **1.000**† | *§17 prose* |
| 3 (IconQA) | all-experts | 0.797 | 0.910 | 0.983 | 0.994 | *§17 prose* |
| 3 (IconQA) | history-only | **0.275** | **0.300** | **0.500** | **0.825** | *§17 prose* |

† **Degenerate by construction on Task 4, and flagged rather than quoted.** In
that run the pool holds 16 visible experts and the recall window is Top-8, while
STEP C's candidate union for the 214 unsolved samples happens to be exactly those
same 8 experts (`tested_minus_recall_size_distribution: {"0": 256}` — the
tested-minus-recall size is 0 for all 256 samples). Every scored expert is
therefore inside every sample's own window, so `R@8 = 1.000` carries no
information about the router. The informative numbers are `R@1 = 0.274` and
`R@2 = 0.452`. **Task 3's history-only run does not share this degeneracy** — its
window is 8 of 12 visible experts, so the tested-minus-recall size is 4 on 210
samples and `R@8 = 0.825` is a real measurement.

**Those 0.274 and 0.275 are the strongest single piece of evidence for the
alias-key idea in this campaign.** With the task's own experts present, the
current keys rank *some* solver first 71.9 % and 79.7 % of the time — but that is
largely the self-experts ranking themselves well on their own distribution.
Restricted to historical experts, the same keys put a solver first only 27.4 %
and 27.5 % of the time: **the fixed-query geometry knows which expert can do the
job far less reliably than the all-experts numbers suggest.** §16 shows this is
not because the capability is absent, and §17 measures, per sample, what a
current-task alias key buys against exactly this gap.

**BLOCKER / WHY / IMPACT / PROPOSED MINIMAL FIX — `SetExact`.** The
specification names this column but does not define it. The closest verifiable
reading is "the router's Top-k set equals the set of experts the teacher found
solving the sample", which for k = |solving set| is exactly `V7 R@k` computed
over the solver set already tabulated — reporting it twice under two names would
be padding. The nearest *additional* measurable quantity is the fraction of
samples where the router's Top-1 is the expert the teacher selected
(`selected_recall_at_k["1"]`, §16, 0.332 on Task 4 / all-experts). This report
publishes that instead of inventing a definition, and flags the gap rather than
silently substituting.

---

## 16. Full-Pool Recall Audit: capability failure or retrieval failure?

Every number here comes from `compose/experiments/v8a_recall_audit.py`, which
recomputes the curves from primary evidence — `single_values` + `solved_threshold`
+ `selected_experts` — rather than from the three-valued labels, so the result
does not depend on which labelling rule wrote the record (§9.2). "Solver" means
*a scored expert whose metric met the threshold*, unioned with the selected
route; a sample with no solver is a genuine Residual.

| Task | Scope | samples | base solved | solved by ≥1 expert | Residual | selected R@1 | R@2 | R@4 | R@8 | solver R@1 | R@2 | R@4 | R@8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | all-experts | 256 | 42 | 199 | 15 | 0.332 | 0.608 | 0.889 | 0.985 | **0.719** | 0.849 | 0.950 | 0.990 |
| 4 | history-only | 256 | 42 | 73 | 141 | 0.164 | 0.425 | 0.699 | 1.000† | 0.274 | 0.452 | 0.740 | 1.000† |
| 3 | all-experts | 256 | 46 | 177 | 33 | 0.424 | 0.819 | 0.977 | 0.983 | **0.797** | 0.910 | 0.983 | 0.994 |
| 3 | history-only | 256 | 46 | 40 | 170 | 0.100 | 0.125 | 0.400 | 0.725 | 0.275 | 0.300 | 0.500 | 0.825 |

† Task 4's history-only `R@8 = 1.000` is degenerate for the reason given in §15
Table 2's footnote: there the candidate union *is* each sample's Top-8 window, so
`capability_present_but_outside_recall_window` is 0 and
`selected_route_outside_own_recall_window` is 0 — there is no wider union for a
route to hide in, and §15's upper-bound caveat does not apply to that row.
**Task 3's history-only row is not degenerate**, and the difference is worth
stating because it makes the row more informative rather than less: its window is
8 of 12 visible experts, `capability_present_but_outside_recall_window` is 7, and
`selected_route_outside_own_recall_window` is 11, so `R@8 = 0.825` and §15's
upper-bound caveat are both real there.

That last row deserves one more sentence, because it carries a stronger claim
than the other three. The Task 3 history-only run recalled 12 distinct experts
over its 210 unsolved samples — *every* expert the scope makes visible — and
STEP C scores every candidate on every pending sample, so `tested_singles` is
`[0..11]` even for a sample whose own window is 8 wide
(`tested_minus_recall_size_distribution: {"0": 46, "4": 210}`). Its 170 Residual
samples were therefore scored against the complete visible historical pool.
**For that run the numbers in this table are exact, not lower bounds**, and the
audit's own caveat — "an expert that was never recalled was never tried" — does
not apply. Task 4 is the opposite case: 8 of 16 visible experts were recalled, so
its 141 Residual is a bound. §23 develops the distinction.

Three readings, in order of how much they matter. The first is the one that
changed when the history-only runs finished: **the all-experts rows flatter the
router, because the experts that dominate their top ranks are the task's own.**

1. **Retrieval is the bottleneck for historical experts, and the all-experts
   number hid it.** Over the whole pool, 71.9 % (Task 4) and 79.7 % (Task 3) of
   solvable samples have a solver ranked **first**. Restricted to historical
   experts, the same keys put a solver first only **27.4 %** and **27.5 %** of
   the time (`solver R@1`, history-only rows), and only 45.2 % and 30.0 % inside
   the Top-2. The current fixed-query geometry is good at ranking an expert on the
   distribution it was trained on and weak at ranking a *historical* expert on a
   new distribution — which is precisely the deficit a current-task alias key
   exists to close, and the reason §17's measurement matters more than the
   headline metric. §17 finds the same deficit from the other side: with an alias
   key added, historical `R@1` rises to 60.3 % and 80.0 %.
2. **Selection ordering is weaker than retrieval, in both scopes.** The expert
   the teacher *selects* sits first only 33.2 % and 42.4 % of the time over the
   whole pool and 16.4 % and 10.0 % restricted to history. The gap is not a
   failure — it is the NLL tie-break and the lexicographic rule choosing among
   several solvers (§6) — but a key that encodes "this expert solves *this*
   distribution" should rank its expert above equally-capable alternatives, and
   today it does not. Task 3 shows this most sharply: its history-only `R@4` is
   0.400 against a `R@1` of 0.100, so three quarters of the recoverable capability
   sits between rank 2 and rank 4.
3. **Past the router, what remains is capability, not retrieval.** In the
   all-experts scope only 2 samples on Task 4 (0.8 %) and 1 on Task 3 (0.4 %)
   have a solver in the visible order but outside the Top-8 window; in the
   history-only scopes the counts are 0 (Task 4, by construction) and 7
   (Task 3, 2.7 %). The Residual set is what the router cannot fix: 15 and 33
   samples with the whole pool available, **141 (55.1 %) and 170 (66.4 %) with
   only the historical pool** — samples no recalled expert solved at all. Those
   are lower bounds: an expert that was never recalled was never tried, which is
   exactly what a wider M or a better key could change — the audit states this in
   its own output rather than leaving it to be assumed.

The 141 and 170 history-only Residual samples are the load-bearing numbers for
V8's second half: together 311 of 512 samples are simultaneously the evidence
that historical reuse alone is insufficient and the precise worklist a V8-B
Candidate Expert would be trained on. V8-A cannot say whether such an expert
would learn them; it can say how many there are, that Task 3's set is larger, and
that they are not a retrieval artefact.
---

## 17. Alias Key Analysis

Answers causal-chain Q3: **if V8 gave a historical expert a second key on the
current task, would the router find that expert more often?** V8-A trains
nothing, so this is a counterfactual, computed by
`compose/experiments/v8a_alias_keys.py` over the exported teacher caches. For
each expert the teacher found solving, it creates exactly one alias key by the
specified initialisation `K_init(k, t) = Normalize(mean(q_i for i in P(k, t)))`
over that expert's positive samples — no optimisation, no gradient — and then
re-measures recall of a solving expert before and after. Lazy creation applies
(`|P(k, t)| > 0`), which is why the key counts are as small as they are.

**Table 3 — per-sample alias counterfactual.** `R@k` here is the fraction of
*samples* that have at least one solving expert inside the router's Top-k — a
per-sample denominator, deliberately different from §15 Table 2's per-instance
one. `before` is the frozen V7 key set (`1 Expert : 1 Key`); `after` adds the
alias keys; `wins` counts, per (sample, solving expert) pair, whether that
expert's best-scoring key is its alias or its origin key.

| Task | Scope | aliases created | R@1 before → after | R@2 before → after | R@4 before → after | R@8 before → after | key wins (alias / origin) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 (CLEVR-Math) | history-only | 7 | **0.164 → 0.603** | **0.425 → 0.904** | **0.699 → 0.973** | 1.000 → 1.000† | **75 / 0** |
| 3 (IconQA) | history-only | 11 | **0.100 → 0.800** | **0.125 → 0.925** | **0.400 → 1.000** | 0.725 → 1.000 | **43 / 0** |
| 4 (CLEVR-Math) | all-experts | 4 | 0.337 → 0.316 | 0.617 → 0.607 | 0.903 → 0.811 | 1.000 → 1.000 | 19 / 198 |
| 3 (IconQA) | all-experts | 4 | 0.424 → 0.441 | 0.819 → 0.842 | 0.977 → 0.972 | 0.983 → 1.000 | 4 / 190 |

† Degenerate in this run for the reason given in §15 (`tested_minus_recall_size`
is 0 on all 256 samples); the informative columns are `R@1` and `R@2`.

**The result splits cleanly by scope, and the split is the finding.**

1. **Where aliases are the only key on the expert — the history-only scope —
   they dominate.** Per-sample `R@1` goes 16.4 % → 60.3 % on Task 4 and
   10.0 % → 80.0 % on Task 3, and **every single winning key is an alias**
   (75/0 and 43/0). The origin key, which encodes "this expert was formed on
   Task 0/3/4", is simply the wrong question to ask of a Task 3 or Task 4
   sample; a key that encodes "this is what this expert looks like *on the
   current task's* queries" is the right one. This is the strongest evidence in
   the campaign that V8's `1 Expert : N Keys` design targets a real problem, and
   it is consistent with §15's low `R@1` on historical experts.
2. **Where origin keys already work — the all-experts scope — aliases are
   mostly redundant, and at small k they can actively hurt.** Only 4 historical
   experts had any positive support at all (the rest of the solved samples were
   solved by the task's own experts, whose aliases are correctly refused —
   §9.5), the alias wins 19 of 217 and 4 of 194 decisions, and on Task 4
   `R@1` *falls* 0.337 → 0.316 and `R@4` falls 0.903 → 0.811. The mechanism is
   the one §18/§23 already identify: alias keys are near-duplicates of the
   expert's behaviour on this distribution, so they add several similarly-scoring
   entries to a fixed-size Top-k window and can push a solver out of it. **An
   alias key is not free.**

Two honest limits on the table, both stated in the tool's own docstring:

- **It is an upper bound on a *trained* alias key at this support size**, because
  the centroid is built from the same split it is evaluated on. The right
  question to read it with is "is there retrieval headroom at all?", and the
  answer is unambiguously yes.
- **It is a lower bound on what the router can be made to see**, because it uses
  only the frozen zero-parameter query and no gradient.

The alias keys are not copies of the origin keys: `cos(alias, origin)` ranges
0.567–0.793 across all 22 created keys (median ≈ 0.71). A key that was a copy
would tell the router nothing; these are the same expert seen from a different
task's geometry.

**BLOCKER / WHY / IMPACT / PROPOSED MINIMAL FIX — the two all-experts rows could
not be produced as specified, and the blocker was a real defect.** The first
attempt at this analysis failed on both all-experts runs with
`MultiKeyPoolError: alias key e16_t4_task_alias sits on the origin task; use the
origin key`. That is the pool's own `validate()` refusing a key the analysis had
just built, and it was right to: `create_alias_keys` created an alias for *every*
expert with positive support, including the task's own experts, whose origin task
*is* the current task. V8 has no such key — `1 Expert : N Keys` means the origin
key plus aliases on *other* tasks. The defect was unreachable before because
every caller had passed a strictly historical pool; the all-experts scope is what
exposed it. **Fix applied** (`compose/v8/key_learning.py`, with regression test
`test_22b_current_task_expert_gets_no_alias_key`): an expert whose `origin_task`
equals the current task is skipped with that reason recorded, so the function now
enforces the invariant instead of building a pool that cannot validate. The two
all-experts rows above are the re-run after the fix; the four history-scope rows
were unaffected (those pools are historical by construction).

---
## 18. Sample Case Studies

Source: `experiments/runs/0911_v8a_formal/cases_task4.json` and `cases_task3.json`,
produced by `compose/experiments/v8a_cases.py`, which joins the two scope runs,
the frozen validation questions and the answers V7 actually generated in its pair
diagnostic. Totals: **Task 4 — V7 correct 174/256, V8 (history-only) correct
115/256**; **Task 3 — V7 correct 174/256, V8 (history-only) correct 86/256** —
the same 67.97 % / 44.92 % and 67.97 % / 33.59 % as §15, recomputed per sample by
the metric adapter rather than read from a summary, so the case lists and the
headline numbers cannot drift apart.

That both tasks score exactly 174/256 under V7 is not a copy-paste error; it is
verified from the diagnostic's per-task entries *and* by re-running the official
scorer over Task 3's answer file at `…/task3/actual/answers.jsonl` (§15). The
identical `result_text_sha256` in the two entries is explained by the scorer
emitting only `Samples: 256 / Accuracy: 67.97%`.

The four classes below overlap by construction (a sample can be both "V7 miss,
V8 fix" and "residual with context"), so the counts are not a partition and are
not summed anywhere in this report.

| Class | Task 4 | Task 3 |
| --- | --- | --- |
| A — V7 missed it, V8 fixed it | 37 | 19 |
| B — reusable, but not by a historical expert | 127 | 138 |
| C — solved by several experts (IGNORE) | 2 pair-solves | 3 pair-solves |
| D — residual with old context | 141 | 170 |

### A. V7 missed it, V8 fixed it — 37 (Task 4) / 19 (Task 3) samples

`v7_t4_val_106` — *"Subtract all small gray spheres. How many spheres are
left?"*, ground truth `3`:

| | route | state | correct |
| --- | --- | --- | --- |
| V7 actual | — | — | **no** |
| V8 all-experts | `[12]` | Reuse1 | yes |
| V8 history-only | `[12]` | Reuse1 | yes |

Expert 12's `origin_task` is **3 (IconQA)** — a CLEVR-Math counting question
answered by an expert trained on diagram question answering, recalled by the
teacher as a *single* with reason `single_solved_stop_before_pairs`, and it
survives the history-only scope because expert 12 is genuinely historical. This
is the cleanest example in the campaign of the thing V8 claims: an old expert
solving a new task's sample that the current system got wrong.

Task 3 has its own instance, in the mirror direction. `v7_t3_val_10` —
*"How many dots are there?  0. 60 / 1. 54 / 2. 47"*, ground truth `0`:

| | route | state | correct |
| --- | --- | --- | --- |
| V7 actual | — | — | **no** |
| V8 all-experts | `[12]` | Reuse1 | yes |
| V8 history-only | `[9]` | Reuse1 | yes |

Here the two scopes pick **different** experts — 12 (Task 3's own, IconQA) and 9
(`origin_task` 2, VizWiz) — and both are right. The historical scope finds the
answer with an expert from a completely different task, which is the strongest
possible form of this class.

`v7_t4_val_107` and `v7_t3_val_124` are the other flavour of class A: V7 is
wrong, and V8's **`BaseOnly`** state — no expert at all, `selected_experts: []` —
is right. The minimal-capacity rule (PART 8: base solves → stop) is not just an
efficiency device; here it converts a V7 miss into a hit by declining to route.
Task 3's instance asks *"What has been done to this letter?  0. turn / 1. slide /
2. flip"* and the frozen base gets it right where V7's routed experts did not.

### B. Reusable, but not by a historical expert — 127 (Task 4) / 138 (Task 3) samples

`v7_t4_val_0` — *"Subtract all small purple balls. Subtract all small gray shiny
cylinders. How many objects are left?"*, ground truth `5`:

| | route | state | correct |
| --- | --- | --- | --- |
| V7 actual | — | — | yes |
| V8 all-experts | `[18]` | Reuse1 | yes |
| V8 history-only | `[]` | Residual | **no** |

Expert 18 is one of Task 4's **own** experts (`origin_task = 4`). So this sample
is "reused" in the all-experts scope and unsolvable in the history-only scope —
it is one of the 127 samples (§14) that make the all-experts number look better
than historical reuse is. Its history-scope fallback kept `residual_context:
[1]`.

Task 3's version is the same shape and is the majority class there, not the
minority one. `v7_t3_val_1` — *"How many trees are there?  0. 3 / 1. 4 / 2. 2 /
3. 1 / 4. 5"*, ground truth `0`:

| | route | state | correct |
| --- | --- | --- | --- |
| V7 actual | — | — | **no** |
| V8 all-experts | `[12]` | Reuse1 | yes |
| V8 history-only | `[]` | Residual | **no**, `residual_context: [11]` |

Expert 12 again — Task 3's own. With 138 of 256 samples in this class on Task 3,
it is the **quantitative core of the scope critique**: more than half of Task 3's
all-experts wins come from experts that do not exist at learning time, and every
one of them is a sample a *new* candidate expert would have to cover rather than
reuse.

### C. Several experts solve it — the IGNORE rule in action

The case file's `alternative_solved` counter reads `positives > 1`, which only a
selected *pair* produces, so it actually counts pair-solves (21 on Task 4 and 17
on Task 3 in the all-experts runs; 2 and 3 in history-only) and not alternative
singletons. The right measurement for "multiple solved experts" is the
three-valued label itself, taken over the history-scope records:

| Quantity | Task 4 | Task 3 |
| --- | --- | --- |
| Samples carrying ≥1 `ignore` label | **78 / 256** | **73 / 256** |
| Total `ignore` labels | 401 | 461 |
| Total `positive` labels | 75 | 43 |
| Total `negative` labels | 1,572 | 2,384 |
| `ignore`-per-sample histogram | `{0: 178, 1: 19, 2: 9, 3: 5, 4: 2, 5: 1, 8: 42}` | `{0: 183, 1: 7, 2: 5, 3: 6, 4: 1, 5: 2, 6: 1, 7: 3, 8: 47, 9: 1}` |

`v7_t4_val_10` is the illustration: state `Reuse1`, `positive: [13]`,
`ignore: [0, 3, 12]`. Three experts that also solve the sample are labelled
IGNORE rather than NEGATIVE, which is PART 9's prohibition ("禁止：E5 =
negative") doing real work — labelling them negative would push their keys away
from a query they demonstrably answer. The 42 samples with eight IGNOREs are the
`BaseOnly` samples, where nothing should be pushed either way; Task 3 has 47 of
those and one sample with nine IGNOREs (12 experts tested, one positive).

Task 3's version, `v7_t3_val_184` — *"How many marbles are there? Estimate.  0.
about 70 / 1. about 40"*, ground truth `0`, `solved_experts: [0, 2]`: the teacher
composes the pair `[0, 2]` in the history scope and `[15]` alone in the
all-experts scope, and all three routes answer correctly. Both solvers are marked
POSITIVE and neither is pushed away, which is what makes the pair search able to
*use* them rather than fight them.

The `negative` count is the number that shows how selective the rule is: 1,572
and 2,384 expert-sample pairs were scored and found to solve nothing, against 75
and 43 positives. The teacher is not generous with the POSITIVE label.

### D. Residual with old context, and what a new expert would face — 141 (Task 4) / 170 (Task 3) samples

`v7_t4_val_1` — *"Subtract all balls. How many objects are left?"*, ground truth
`6`:

| | route | state | correct |
| --- | --- | --- | --- |
| V7 actual | — | — | yes (using expert 18) |
| V8 all-experts | `[18]` | Reuse1 | yes |
| V8 history-only | `[]` | Residual | **no**, `residual_context: [1]` |

No historical expert solves it, so the teacher falls to the residual rule
(highest metric, lowest NLL, smaller cardinality) and keeps expert 1 as context.
The sample is a concrete instance of the V8-B contract: a frozen historical
expert as context plus a new candidate that has to learn the delta. Whether the
candidate *can* is exactly what V8-A cannot answer — and this sample is also a
reminder that V7 is correct here, so a V8-B candidate must not make it worse.

`v7_t3_val_101` — *"Use dice to measure the line. The line is about (_) dice
long."*, ground truth `6`, is Task 3's version and shows the residual rule
choosing differently from the pair search: the all-experts scope composes
`[15, 12]` and answers correctly, while the history scope finds no single or pair
that works (`history_reason:
no_single_or_pair_solved_residual_context_kept`) and keeps expert 11 as context.
The pair that *would* have worked needed an expert that does not exist at
learning time.

`v7_t3_val_102` — *"How many fish are there?"*, ground truth `16` — is the other
end of this class and the honest limit of the whole method: V7 is wrong, the
all-experts scope returns `Residual` too (`all_selected: []`), and the history
scope keeps context `[10]`. Nothing in the committed pool of 23 experts solves
it. 32 of Task 3's 256 samples are in this position in both scopes (§14), against
14 on Task 4. These are the samples that only a newly trained expert can reach,
and they are the irreducible argument for the second half of V8.

### What the four classes say together

Class A shows the mechanism works and is worth training for: 37 and 19 samples
are solved by a *historical* expert that V7 got wrong. Class B and D together are
the honest limit: of Task 3's 256 samples, 86 are reachable from the frozen
historical pool and 138 are not; on Task 4 the split is 115 and 127. On both
tasks the samples that are *not* reachable include ones V7 already gets right.

That is the case for V8-B being a *residual* learner on top of frozen context
rather than a replacement router — and it is also why §19.5 recommends a pilot
rather than a full loop. The class that grew most between the two tasks is D
(141 → 170): the later the task, the larger the set no historical expert can
touch, which is the opposite of the monotone-improvement story a reuse-only
method would need.

---
## 19. V8-B (minimal continual loop, Task 0 → Task 2)

**Verdict: NOT RUN.** This section states what V8-B is, what it now costs, and
why it was not launched — the specification forbids starting a multi-day
training without an explicit request ("除非用户明确要求：不要自动启动预计持续多天的完整训练。
先给 evidence-based recommendation"), and the user has not given one.

### 19.1 What V8-B is

Task 0 → Task 2 as a continual loop, one task at a time, each task:

1. **Teacher** over the current task's *training* split: STEP A base, STEP B
   Top-M historical recall (M = 8), STEP C historical singles, then pairs over
   the K_s = 4 shortlist only if no single solved. Verdicts are written to the
   canonical cache (`write_teacher_result`, `stores_ground_truth: false`).
2. **Alias keys** created lazily from teacher positives
   (`create_alias_keys`, support ≥ `alias_support_threshold`).
3. **Gated training** (`compose/v8/trainer.py`): mixed batches, per-sample
   weight `r_i`, candidate LoRA + alias keys only, frozen ledger verified after
   the last step.
4. **Prune → commit → checkpoint**, then the *next* task, with the committed
   pool as its history.

Task 0 has no history, so its teacher is degenerate (BaseOnly/Residual only,
`selected_set = []`) and steps 1–2 reduce to the V7 recipe; the loop's novel
content is tasks 1 and 2.

### 19.2 Cost, derived from measurements rather than guessed

| Component | Measurement | Source |
| --- | --- | --- |
| V7 training, one task | 621 optimizer steps × 34.55 s = **5 h 58 min** (≈0.54 s/sample at accumulation 64) | `…/task4/logs/training.log`, `metrics/train_steps.rank*.jsonl` |
| V8-A teacher, one run | 3,840 route generations (256 samples × 15 routes); wall-clock from run 1 | §21 (measured) |
| Teacher budget for a *training* split | 2,000 samples (the repo's own declared budget, `compose/experiments/task_run.py:157`) | legacy config |
| ⇒ V8-B teacher, one task | 2,000 × ≤15 routes ≈ **6–8 h single-GPU**, ~1.5–2 h sharded over 4 | derived |
| ⇒ V8-B training, one task | ≈ V7's own ≈ **6 h** on the same 8-GPU recipe | derived |

**One task ≈ 12–14 GPU-hours; the Task 0 → Task 2 loop ≈ 35–40 GPU-hours**, plus
evaluation. Against a cluster where GPUs 2–7 are held by other users and only
one device was free for this work, that is a multi-day wall-clock commitment —
which is exactly the case the specification says not to enter unasked.

### 19.3 What is ready, and what the first run would still need

Ready and tested: the teacher and its cache, lazy alias keys, the gating and the
whitelist, the frozen ledger, pruning, commit, checkpoint/resume, the inference
router, and — added in this campaign — the training loop itself
(`compose/v8/trainer.py`, §19.4). Not written, because it belongs to the media
side that V8 reuses from V7 unchanged: the current-task dataloader that turns
`val_full`-style records into teacher-forced batches. In a real run
`forward_fn` is a thin adapter over the frozen backbone and
`compose.v7.training.teacher_forcing_token_nll`.

### 19.4 The training loop that was missing, and is no longer

Until this campaign V8 had no training loop at all: the primitives were tested
in isolation and nothing assembled them, so V8-B had no entry point and the
"B. current-task Candidate Expert LoRA / A. alias keys" trainable set of PART 4
was enforced only in the abstract. `compose/v8/trainer.py` now provides it:
`V8TaskTrainer` enforces the freeze and captures the ledger *before* building the
optimizer, runs mixed (never filtered) batches with `residual_answer_loss`,
trains candidate LoRA and alias keys in two parameter groups, checks the
gradient footprint after every step, offers `leakage_probe` for the audit trail,
and `finalize()` prunes, re-verifies the ledger, commits and checkpoints.

It is exercised by two tests on the real `ComposeLinear` path
(`test_v8_trainer_trains_the_candidate_and_the_alias_keys_only`,
`test_v8_trainer_leakage_probe_shows_mixed_batches_change_nothing`): after a full
epoch, only the candidate expert's gradient is non-zero, both supported alias
keys move, the historical LoRA tensors are bit-identical, the support-less key is
pruned, the ledger reports `UNCHANGED`, the task commits, and the checkpoint
resumes with `IDENTICAL` identity.

### 19.5 Recommendation

**Do not launch the full loop yet; do run a one-task V8-B pilot first.** The
V8-A evidence (§14–§17) determines whether the pilot is worth its ~12 GPU-hours,
and the pilot answers the two things V8-A structurally cannot: whether alias keys
*learned from the current distribution* raise recall (§16/§17 measure the
untrained centroid, which is an upper bound on retrieval but not a measurement of
learning), and whether the candidate expert can absorb residual capability on top
of a frozen historical context. Recommended sequence: Task 1 only (not Task 0,
which has no history and so tests nothing new), with the teacher over a
500-sample subsample rather than 2,000 — a ~3 h pilot that exercises every stage
end to end.

---

## 20. Expert Pool Efficiency

V8 changes the *key* side of the pool, not the expert side: the expert set V7
committed stays exactly as it is, and alias keys let one historical expert be
recalled in later task distributions. So this section has to be read in three
parts — the pool V7 actually grew (fully measured), what V8-A measured on top of
it without training anything (measured for the run that has finished), and the
*reduction* in new-expert growth that V8 is designed to produce (a V8-B
property, and therefore **not measured**; claiming it here would be exactly the
kind of unearned PASS PART 37 forbids).

### 20.1 The V7 pool, per committed checkpoint

Source: `task{N}/committed/compose_experts.json` under
`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_gpu01_cached_query_formal_20260903/`,
read directly from the frozen manifests. `Experts` lists the actual expert ids,
not a count derived from configuration.

| Task | New expert ids | Experts created | Cumulative live experts | Cumulative keys (origin / alias) | Adapter params | Per-expert params |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (ImageNet-R) | 0–3 | 4 | 4 | 4 / 0 | 79,953,920 | 19,988,480 |
| 1 (ArxivQA) | 4–7 | 4 | 8 | 8 / 0 | 159,907,840 | 19,988,480 |
| 2 (VizWiz) | 8–11 | 4 | 12 | 12 / 0 | 239,861,760 | 19,988,480 |
| 3 (IconQA) | 12–15 | 4 | 16 | 16 / 0 | 319,815,680 | 19,988,480 |
| 4 (CLEVR-Math) | 16–19 | 4 | 20 | 20 / 0 | 399,769,600 | 19,988,480 |
| 5 (Flickr30k) | 20, 21, 22, 23 (22 pruned) | 4 | 23 | 24 / 0 | 459,735,040 | 19,988,480 |

Two facts in that table are load-bearing for V8's premise:

* **Every task grows the pool by exactly four experts, and every expert gets
  exactly one key.** `task5/committed/v7_keys.pt` holds 24 tensors of shape
  `[1536]` (`schema_version 1`, `query_dim 1536`, `pool_version 23`) with
  `lifecycle` metadata of `historical: 23` and `pruned: 1` — the four-alias
  structure of §8 has no V7 counterpart, and `experts_with_multiple_keys` is
  empty.
* **Pruning removes the expert, not its key.** Expert 22 is absent from the
  task-5 manifest (23 live experts, 459,735,040 params = 23 x 19,988,480) while
  its key survives with `lifecycle: pruned`. Any V8 alias-key layer therefore has
  to decide what to do about a historical expert whose key exists but whose LoRA
  does not — the multi-key pool keeps keys addressable by lifecycle (§8).

### 20.2 What V8-A added, without training

V8-A runs the Answer-Supervised Teacher over the frozen pool and writes a route
per sample. It trains nothing, so the honest column is "0 new experts, 0 new
committed keys" — with the caveat that the *acquisition* of alias keys is
measured separately and non-committally in §17, and the *reduction* in expert
growth is a V8-B question.

All four runs are now complete, so this table covers both tasks and both scopes.
Every count is the run's own `states` from `analysis.json`; nothing is derived
across runs.

| Quantity | Task 4 all-experts | Task 4 history-only | Task 3 all-experts | Task 3 history-only |
| --- | --- | --- | --- | --- |
| New experts trained for the task | 0 | 0 | 0 | 0 |
| New committed keys for the task | 0 | 0 | 0 | 0 |
| Samples the base model already solved | 42 (16.41 %) | 42 (16.41 %) | 46 (17.97 %) | 46 (17.97 %) |
| Samples served by **one** expert | 178 (69.53 %) | **71 (27.73 %)** | 160 (62.50 %) | **37 (14.45 %)** |
| Samples served by **a pair** of experts | 21 (8.20 %) | **2 (0.78 %)** | 17 (6.64 %) | **3 (1.17 %)** |
| Residual (no route found) | 15 (5.86 %) | **141 (55.08 %)** | 33 (12.89 %) | **170 (66.41 %)** |
| Reuse rate (any non-empty route) | 199 (77.73 %) | **73 (28.52 %)** | 177 (69.14 %) | **40 (15.63 %)** |
| Residual ratio | 5.86 % | 55.08 % | 12.89 % | 66.41 % |
| Wall time | 65.3 min | 66.4 min | 147.1 min | 142.1 min |
| Routes generated | 2,612 | 2,826 | 4,336 | 3,814 |
| NLL evaluations (live / reused) | 2,396 / 216 | 1,968 / 858 | 4,036 / 300 | 2,776 / 1,038 |

**The scope qualifier in the header is not cosmetic, and this table is where it
bites.** The all-experts runs let the teacher route to the task's *own* committed
experts (16–19 on Task 4, 12–15 on Task 3) as well as to the historical ones, and
a task's own experts are exactly what is unavailable while that task is being
learned. The gap is not a rounding correction — it is a factor of 2.5 on Task 4
(178 → 71 single-expert samples) and **4.3 on Task 3** (160 → 37). The teacher's
own progress line says the same thing in situ: 178 samples on Task 4 and 160 on
Task 3 stopped at a solved single in the all-experts scope, against 71 and 37 in
the history-only scope.

The last column is an independent cross-check on §14, and it lands exactly. The
history-only runs' non-empty routes are 71 + 2 = **73** on Task 4 and 37 + 3 =
**40** on Task 3 — the same 28.52 % and 15.63 % that `v8a_scope_gap.py` produced
by joining the two scopes per sample, computed from a different pair of artefacts
by a different program. The join is therefore reproducing the runs' own state
counts rather than measuring something of its own, which is what makes §14's
scope-split table safe to quote.

The V7 comparison in §20.3 is deliberately drawn from §15 Table 2 rather than
from a new count: V7 does route over the whole committed pool, so "fraction of
samples V7's own router sends to an expert" is already measured there, and
re-deriving it here from a different artifact would risk two numbers for one
quantity.

### 20.3 The efficiency claim V8 is *designed* to make, and its status

V8's thesis is that `new_experts_per_task` should fall because samples a
historical expert already solves should not motivate a new LoRA. That claim has
three parts, and only the first two are measurable from this campaign:

1. **The opportunity exists, and it is smaller than the all-experts headline.**
   Of Task 4's 256 samples, 42 need no expert and 73 are served by a historical
   expert; on Task 3 the same split is 46 and 40. This is measured, and it is the
   history-only column of §20.2. The all-experts reuse rates (77.73 % and
   69.14 %) may not be quoted as the opportunity without the scope split, and
   §14 gives the per-sample join that produces the honest version.
2. **V7 misses part of that opportunity.** Measured in §15 Table 2 and analysed
   in §16: the teacher's own selected route lies inside its Top-8 recall window
   for 196 of 199 expert-routed samples on Task 4 and 174 of 177 on Task 3 —
   only 3 samples in each all-experts run were routed outside the window. What
   V7's *own* router does with the same pool is the other half: it puts a solver
   at rank 1 for 0.719 and 0.797 of solver samples over the whole pool, but only
   0.274 and 0.275 once restricted to historical experts. That collapse is the
   room V8's alias keys are meant to close, and §17 measures how much of it an
   untrained alias key already closes (`R@1` 0.164 → 0.603 and 0.100 → 0.800).
3. **V8 would therefore train fewer experts.** **Not measured.** Turning the
   reuse rate into a reduction requires choosing which samples still need a
   Candidate Expert, training it on the residual, and observing that the pool
   grows more slowly than V7's four-per-task. That is V8-B, and §19.5
   recommends a one-task pilot rather than asserting the result.

The correct reading of this section is therefore: V8-A establishes parts 1 and 2
with numbers on two tasks, and leaves part 3 open. The pool-efficiency benefit is
a *projection* until a V8-B run produces a task whose committed expert count is
below V7's four.

One thing this campaign does establish about the *cost side* of the trade, even
without training anything: **a scope that reuses less is a scope that searches
more**, and the search is a real cost line rather than a rounding error. The
history-only runs push far more samples past the singles step into pair search —
143 on Task 4 and 173 on Task 3, against 36 and 50 in the all-experts scope —
and score correspondingly more distinct pairs (13 vs 6 on Task 4, 16 vs 9 on
Task 3). On Task 4 that cost is visible directly: the history-only run takes
**1.7 % more wall time** while scoring *fewer* singles (8 vs 10). On Task 3 the
opposite holds, because the all-experts scope's 18 singles outweigh its smaller
pair search; the two effects are of comparable size and point in opposite
directions there. The marginal cost of the pair step is quantified in §21.1.

## 21. Efficiency

All figures below are measured, and each cites the artifact it came from.

### 21.1 Teacher cost, all four runs

All four runs are on 256 samples with M = 8, K_s = 4, one GPU. Timestamps are the
elapsed seconds the run prints in `run.log`; the deltas between them are the
phase costs. "Singles scored" is `|candidates| × |unsolved samples|` — the
work actually done, not a nominal count.

| Run | Load | STEP A+B end | STEP C end | Pair shortlist | NLL pass end | Total | Routes | Singles scored | s / single | s / NLL eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 4, all-experts | 102.8 | 341.9 | 2,611.6 | 2,894.1 | 3,912.2 | **3,917.3 s** (65.3 min) | 2,612 | 10 × 214 = 2,140 | 1.061 | 0.425 |
| Task 4, history-only | 97.5 | 320.8 | 2,071.3 | 3,170.2 | 3,977.5 | **3,982.4 s** (66.4 min) | 2,826 | 8 × 214 = 1,712 | 1.022 | 0.565 |
| Task 3, all-experts | 104.4 | 471.3 | 6,702.6 | 7,155.4 | 8,823.5 | **8,828.3 s** (147.1 min) | 4,336 | 18 × 210 = 3,780 | 1.649 | 0.413 |
| Task 3, history-only | 96.8 | 448.6 | 4,830.8 | 7,411.3 | 8,523.1 | **8,527.9 s** (142.1 min) | 3,814 | 12 × 210 = 2,520 | 1.739 | 0.401 |

Total campaign GPU time: **25,255.9 s = 7.02 h** for four teacher runs, two
tasks, 512 sample-route evaluations, ~13,600 generations and 64,768 seeded pair
NLLs replayed per run with `max_abs_diff 0.0` in all four.

**Why the two tasks cost different amounts, and it is not contention.** The four
runs were strictly sequential (08:54, 10:01, 12:28, 14:50 completion), so they
never overlapped each other, and the 1.6× gap between Task 4's ~1.0 s/route and
Task 3's ~1.7 s/route tracks something real: **Task 3 decodes longer answers.**
Measured over each run's `generation_cache.jsonl`, Task 4's routes generate
1.00 words per route at every route type (CLEVR-Math answers are single numbers),
while Task 3's generate 2.45 words for the base, 2.28 for singles and 1.00–3.12
for pairs. A two-parameter fit on the four single-scoring segments,
`seconds ≈ 0.57 + 0.49 × words`, reproduces all four measured rates to within
8 %, which is as much as four points can support and no more — but it makes the
direction unambiguous and, more usefully, it says what a V8-B budget should key
on: **cost is per generated *word*, plus a fixed per-route overhead**, not per
sample.

**The NLL pass is a real and separately-sized cost line.** After the pair
shortlist is scored, `compose/v8/teacher.py:328-346` re-scores *every* route the
teacher evaluated — base, every single, every pair — through the NLL scorer, so
that every NLL recorded in a decision belongs to a route that was actually
scored. That pass occupies the segment between the "scored N distinct pairs" line
and the "teacher states" line: 1,018 / 1,112 / 1,668 / 1,112 s. It is cheaper per
route than generation (≈0.40–0.57 s against ≈1.02–1.74 s) because it is a single
teacher-forced forward over the supervised tokens with no decode loop; the four
seed checks in every run confirm the supervision is 3 tokens wide
(`supervised_token_count: 3`).

**The GPU is not the bottleneck in any of the four runs.** At 14–34 % utilization
(mean ≈25 %, 85–95 W, sampled 5× at 3 s intervals during the Task 4 all-experts
run) the teacher is bound by per-route decode and Python overhead, not by tensor
compute. Two consequences worth recording, because they decide how V8-B should be
run rather than being trivia: cost scales with the **number of routes**, so M and
K_s are the cost levers (deeper recall is linearly more expensive), and the same
GPU can serve other work alongside a run of this shape. Peak allocator memory was
14.83 GiB for the two Task 3 runs and 14.80 GiB for the two Task 4 runs (resident
VRAM sampled at 16,910 MiB). That headroom is what makes the §12.1 exclusivity
gate (≥ 20,000 MiB free, no other user's compute process) a workable condition
rather than a demand for an empty machine.

### 21.2 Training cost (the comparison V8-B would have to beat)

V7's own Task-4 training is 621 optimizer steps at 34.55 s/step = **5 h 58 min**
for 39,743 samples, i.e. 0.54 s/sample at gradient-accumulation 64 (from
`…/task4/logs/training.log` and `metrics/train_steps.rank*.jsonl`). V8-B adds
the teacher on top of that, which is why the pilot in §19.5 is scoped at 500
teacher samples rather than 2,000.

**Cross-check.** §19.2 estimated a 2,000-sample teacher at 6–8 h single-GPU
*before* this campaign ran, from the V7 logs and the legacy `task_run.py`
budget. §21.1's four measured runs now bracket that: 1.02–1.06 s/route on
single-word answers and 1.65–1.74 s/route on the two-to-three-word answers, so a
2,000-sample teacher at ≈11 routes/sample is **6.2–10.6 h** depending on which
task's answer-length distribution it resembles. The pre-campaign estimate sits
inside the measured band, and the band's width is itself the useful part: the
pilot's cost depends on the target task's answer length, not on the sample count
alone.

### 21.3 Inference cost

V8-A's policy replay adds no generation: the routes it selects were already
generated during the teacher's search, so the replay is scoring only. The
1.07 s/route figure is what inference costs when it is paid separately — one
forward-generation per selected route, with the router itself being a matrix
product over the frozen query cache.

---

## 22. Leakage Audit

Four separate channels, each with a mechanism rather than a promise.

### 22.1 No test ground truth in the training teacher cache

The runner refuses to run the teacher over a test split at all:

* `v8_task_run.py setup()`: `if "test" in question_file.name: raise ValueError("refusing to run the teacher over a test split")`.
* V8-A reads `task{N}/data/val_full.json` — the *validation* split, which is the
  split the V7 baselines in this report were also measured on.
* `compose/v8/cache.py:_forbid_answer_fields` rejects any teacher-cache payload
  carrying an answer field, so the cache cannot accumulate GT answers even by
  accident.

### 22.2 No teacher at inference

`compose/v8/inference.py` is a separate module from `teacher.py` and imports
nothing from it. `test_24_inference_path_never_reads_ground_truth_answer` parses
the inference module's AST and asserts `assert_inference_purity(...)["pure"] is True`
with empty forbidden-identifier and forbidden-string sets. The test also proves
the scanner has teeth: a synthetic `def f(ground_truth): return oracle(ground_truth)`
is caught, while prose documenting the ban is not, and a smuggled literal bound
to another name is still caught.

### 22.3 No NLL against GT at inference

The inference path contains no NLL computation at all — it is
query → router → composition → generation. NLL lives in `teacher.py` and
`nll_eval`-equivalent code, which the inference policy never calls
(`test_24`'s identifier scan includes the NLL entry points).

### 22.4 Inference input set

`inference.py`'s forward takes `queries` only. Task labels, correctness, NLL and
the teacher cache are not parameters of any inference function; `validate_policy`
checks a policy against the pool without reference to any per-sample outcome.

### 22.5 Scope statement

The teacher *is* allowed to see validation ground truth — that is the method, not
a leak: it is what "Answer-Supervised" means, and the same information is what the
official metric scores against. What the audit above establishes is that this
information cannot reach the inference path, and that a *test* split can never
reach the teacher.
---

## 23. Failure Analysis

"效果不好" is not an explanation. This section attributes the result to one of
the six candidate bottlenecks the specification names (A–F), and says for each
whether it is **measured**, **measured as small**, or **not exercised by V8-A at
all**. Both tasks are now measured, so the table carries two columns and the
places where the two disagree are treated as findings rather than averaged away.

| # | Bottleneck | Task 4 (CLEVR-Math) | Task 3 (IconQA) | Evidence |
| --- | --- | --- | --- | --- |
| A | Historical capability (the pool simply cannot do it) | **Primary constraint** | **Primary constraint, and exact rather than a lower bound** | 141/256 (55.1 %) and 170/256 (66.4 %) Residual in the history-only scope; 73 and 40 samples are reached by a historical expert route |
| B | Candidate retrieval (the solver exists but is outside the recall window) | **Small in all-experts, zero in history-only** | **Present in history-only, and demonstrably costless** | all-experts `capability_present_but_outside_recall_window` = 2 (Task 4, ranks 12/9) and 1 (Task 3, rank 14); history-only = 0 and 7 |
| C | Key representation (the keys rank the right expert badly) | **Measured; the largest actionable gap** | **Measured; same magnitude** | solver R@1 = 0.274 / 0.275 and R@2 = 0.452 / 0.300 in history-only, against 0.719 / 0.797 and 0.849 / 0.910 with the task's own experts present |
| D | Key optimisation (alias keys not learning) | **Not exercised by V8-A** | **Not exercised by V8-A** | V8-A trains nothing; §9.1's hinge-gradient defect was found and fixed before the campaign, so the optimiser path is repaired but unmeasured |
| E | Pair composition (the right pair is never composed) | **Measured as small, with one verified case** | **Measured as small, with one verified case** | 2 and 3 pair-only solves in history-only; `v7_t4_val_211` and `v7_t3_val_189` are each solved in the history scope but never composed in the all-experts scope |
| F | New-expert learning (the candidate fails to absorb the residual) | **Not exercised by V8-A** | **Not exercised by V8-A** | Requires V8-B; §19.5 scopes the pilot |

**Why A is the primary constraint and not B, on Task 4.** The two are easy to
confuse, because both show up as "the sample is not solved". They separate
cleanly here: B would mean the solver is in the pool but the router never
surfaced it, whereas in the Task 4 history-only run every scored expert was
inside every sample's own window (`tested_minus_recall_size_distribution:
{"0": 256}`), so the router was not filtering anything out — the 141 Residual
samples were scored and simply had no solver among the 8 candidates. That count
is still a **lower bound**, because only recalled candidates are ever scored and
8 of Task 4's 16 visible historical experts were never tried on any sample.

**On Task 3 the same argument becomes exact, and that is the strongest single
result in this section.** The Task 3 history-only run recalled 12 distinct
experts over its 210 unsolved samples — which is *every* expert the history-only
scope makes visible. Because STEP C scores every candidate on every pending
sample, a sample whose own Top-8 window excluded a candidate still had that
candidate scored:
`v7_t3_val_189`'s window is `[0, 3, 2, 1, 10, 8, 4, 5]` while its `tested_singles`
is `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`, and expert 7 — absent from the
window — is the one that solves it. `tested_minus_recall_size_distribution` is
`{"0": 46, "4": 210}`: on every unsolved sample, four experts were scored beyond
the Top-8 window. So **Task 3's 170 Residual samples are not a retrieval artefact
and are not a lower bound — all 12 visible historical experts were scored against
every one of them, and none solved them.** The recall-audit caveat ("an expert
that was never recalled was never tried") does not apply to that run, and this
report says so rather than carrying the caveat forward by default.

B is nonetheless real on Task 3, just cheap: 7 Task 3 history-only samples have a
solver sitting at ranks 9–12, outside the *notional* Top-8 window. Because the
pool was small enough that those experts were tested anyway, B cost that run
**zero** samples. It would have cost real samples if the history pool had been
larger, which is the one place where a bigger pool is a liability.

**Why C is the most actionable finding.** A is a statement about the frozen pool
— nothing in V8 can change it without training, and that training is V8-B. C is
different: the capability is present (73 and 40 samples are solved once the right
expert is selected), but the current single-key geometry ranks a solver first
only 27.4 % and 27.5 % of the time on a new task's distribution. That is the gap
alias keys are defined to close, it is measurable without any training, and §17
measures it: with one untrained alias key per solving expert, historical `R@1`
rises to 60.3 % and 80.0 %. C is therefore not a hypothesis in this report; it is
a quantified deficit with a quantified partial remedy.

**What would falsify this attribution.** If a V8-B run produced alias keys and
recall-at-1 did not move, C would be wrong and the cause would be D (the
optimisation) or the fixed-query geometry itself — the query is frozen and
detached (§3.1), so no key can encode information the query does not carry. If
alias keys did move recall but the metric did not, the bottleneck would be E or
the selection rule. The report keeps those branches distinct rather than
treating a poor final number as evidence about any one of them.

**The 311-sample worklist.** Task 4's 141 and Task 3's 170 history-only Residual
sets are disjoint in the sense that matters: they are different samples of
different tasks, and together 311 of 512 sample evaluations describe capability
that the frozen pool does not have. On Task 3 that set is *certified* complete
with respect to the visible pool; on Task 4 it is bounded above by what 8
untried experts could add. If a V8-B pilot is run, this is the set it has to
move, and §19.5 scopes it accordingly.

**Failures that occurred, and are reported as such (PART 46 item 20).** Three
things in this campaign went wrong rather than merely underperforming, and all
three are recorded above instead of omitted: the three-valued key-target rule was
not total, and it had already written 33 mislabelled key targets into the pre-fix
run's artefact (§9.2, quantified by `v8a_label_audit.py`); `create_alias_keys`
created an alias key on an expert's own origin task, which the pool's
`validate()` then rejected — a real defect, exposed only by the all-experts scope
and fixed with a regression test (§17); and the Task 0 answer-parity check
crashed on a contended GPU and, at the time of writing, has produced no result
(§12.1). No run was re-run to make those disappear, and no artefact was
rewritten.

---
## 24. Acceptance Checklist

Every row cites the artefact that decides it — a test name, a file:line, or a
section of this report. "Test" means it is decided by a test that runs in the
608-test suite (§11); "report" means it is decided by a measured run.

| # | Requirement | Decided by | Status |
| --- | --- | --- | --- |
| 1 | V7 still runnable | 548 pre-existing tests pass; `compose/v7/`, `compose/eval/`, `llava/` byte-unchanged (`git diff --stat`) | PASS |
| 2 | Fixed query unchanged | `test_01_fixed_query_has_zero_trainable_parameters`; queries reused from the V7 cache, contract `6c51879f…` | PASS |
| 3 | Historical LoRA frozen | `test_02_historical_lora_requires_grad_false`, `test_03_historical_lora_checksum_unchanged_after_training`; §10.3 | PASS |
| 4 | Origin keys frozen | `test_04_historical_committed_old_key_unchanged` (and it fails when the key is mutated); §10.3 | PASS |
| 5 | Multi-Key pool works | `test_05_one_expert_can_own_multiple_keys`, `test_27_v7_checkpoint_migration_creates_origin_keys_correctly` | PASS |
| 6 | One expert cannot fill two Top-2 slots | `test_07_one_expert_cannot_occupy_two_top2_slots`; router aggregates per expert by max | PASS |
| 7 | Correctness controls `solved` | `test_08_teacher_uses_task_correctness_for_solved_decision`; `metric_adapter.py` is the only `solved` producer | PASS |
| 8 | NLL never independently controls `solved` | `test_09`, `test_10`, `test_teacher_rejects_nll_solved_threshold_configuration`; `config.py:66-85`, `:332` | PASS |
| 9 | Alternative solved expert is IGNORE, not negative | `test_11_multiple_solved_experts_best_positive_others_ignore_unsolved_negative`, `test_21` | PASS |
| 10 | BaseOnly / Reuse1 / Reuse2 get no candidate-LoRA gradient | `test_16`, `test_17`, `test_18` | PASS |
| 11 | Residual candidate does get gradient | `test_19_residual_candidate_lora_receives_gradient` | PASS |
| 12 | No gradient leaks across a mixed batch | `test_30_gradient_leakage_on_mixed_baseonly_reuse_residual_batch` | PASS |
| 13 | Alias key created lazily, only with support | `test_22_zero_support_historical_expert_gets_no_alias_key`; `key_learning.py:create_alias_keys` | PASS |
| 14 | Zero-support alias not committed | `test_pruning_retires_zero_support_alias_but_never_strands_an_expert` | PASS |
| 15 | Inference never reads ground truth | `test_24_inference_path_never_reads_ground_truth_answer` (AST scan, with negative controls) | PASS |
| 16 | Full-pool recall audit works | `compose/experiments/v8_full_pool_recall.py`; §16 | see §16 |
| 17 | V7 regression passes | 548 tests; §12 | PASS |
| 18 | Resume is deterministic | `test_25_teacher_cache_reload_deterministic`, `test_26_resume_preserves_expert_key_mapping` | PASS |
| 19 | V8-A evidence collected | §13–§18 | see §14–§18 |
| 20 | V8-B evidence collected *if run* | not run — costed and recommended in §19 | N/A (declared) |
| 21 | Method and implementation reported separately | §1 and §25 keep the two verdicts apart, per PART 37 | PASS |

---

## 25. Final Verdict

PART 37 requires the code verdict and the method verdict to be reported
**separately**, and says explicitly that a method which underperforms does not
make the implementation a failure. They are therefore stated in two subsections
that do not reference each other's conclusion, followed by the causal chain and
the go/no-go.

### 25.1 Code acceptance — PASS

Every mechanism the specification names is implemented, and each is enforced by
something that can fail rather than by a comment:

| Mechanism | Enforced by | Can it silently not work? |
| --- | --- | --- |
| Historical LoRA + committed origin keys + base frozen | optimizer whitelist built *after* the ledger snapshot; per-step gradient-footprint check; post-run checksum comparison | No — `test_02`, `test_03`, `test_04`, and `test_04` is verified to fail when the key is mutated |
| `M` decides `solved`, `L` only ranks | `metric_adapter.py` is the only producer of `solved`; `test_08`, `test_09`, `test_10`, and a test that the teacher **rejects** an NLL-threshold config | No — the forbidden configuration raises at construction |
| Alternative solved expert is IGNORE, never negative | `test_11`, `test_21` | No |
| Lazy alias keys (`|P(k,t)| > 0` only) | `test_22`, `test_22b`, pruning test | No — `test_22b` covers the origin-task case that `validate()` rejects |
| Per-sample gradient gating on mixed batches | `test_30` | No |
| Inference reads no ground truth, NLL, oracle or task label | AST scan `assert_inference_purity`, `test_24`, with negative controls that prove the scanner has teeth | No |
| Deterministic resume | `test_25`, `test_26` | No |

Suite: **608 passed** in 75.76 s; **548** of them pre-date this work and pass
unchanged. `git diff --stat 9ff2b28..HEAD` = 34 files, 11,874 insertions,
**0 deletions**, and **0 files** under `compose/v7/`, `compose/adapters/`,
`compose/eval/` or `llava/`. V7 is not merely still passing — it is
byte-unchanged, which is why its tests pass by construction.

**Three defects were found during the campaign, and all three are in the record.**
The three-valued key-target rule was not total and had already written 33
mislabelled targets into a pre-fix artefact (§9.2); `create_alias_keys` created a
key on an expert's own origin task, which the pool's own `validate()` then
refused — a defect reachable only once the all-experts scope existed (§17); and
the Task-0 parity harness composed V7's answer path with a duplicated `task0`
segment, which `_jsonl`'s tolerant missing-file handling turned into a bare
`KeyError` thirty lines later (§12.1). The first two are fixed with regression
tests. The third is fixed and **its fix has not yet been executed end to end** —
that is the one open item in this verdict, and it is a *cross-check*, not a
result: no claim in this report depends on it.

**Code acceptance: PASS**, with one declared open item (§12.1, answer parity
pending) that does not gate any other claim.

### 25.2 Method acceptance — PARTIALLY VERIFIED

The method has two halves, and they land differently.

**V8-A's retrieval claim is supported, and by the strongest evidence in the
campaign.** V8's `1 Expert : N Keys` thesis predicts that a historical expert
is hard to *find* on a new task's distribution, and that a second key encoding
"what this expert looks like here" fixes it. Measured: historical solver recall@1
is **27.4 % / 27.5 %** against **71.9 % / 79.7 %** with the task's own experts
present, and adding one untrained alias key per solving expert raises it to
**60.3 % / 80.0 %** — with **100 %** of key-selection wins (75/0 and 43/0) going
to the alias. This is a specified mechanism, measured on real runs, at the cost
of zero training. The task-prescribed prohibition on NLL thresholds is also
vindicated from the other side: the policy *beats* the NLL-oracle ceiling
(`GapClosed` 0.957 and 1.065), which is only possible because `M` and `L` are
kept apart.

**V8-A's reuse claim is real but much smaller than the headline.** The all-experts
results (94.14 % and 87.11 %) are roughly half self-reuse. Removing the task's own
experts, genuine cross-task reuse is **28.52 %** (Task 4) and **15.63 %** (Task 3)
of samples, and more than half of each history-only "reuse" figure is the frozen
base answering correctly on its own (42 of 115, 46 of 86). As a complete policy,
the history-only V8 does **not** beat V7 — 44.92 % and 33.59 % against 67.97 % —
though it does beat the only fair baseline, the frozen base alone, by +28.51 and
+15.62 points. The reuse effect also *shrinks* from Task 4 to Task 3 (28.5 % →
15.6 %), and the set no historical expert can touch *grows* (141 → 170).

**The half the method actually rests on was never run.** V8-B — training a
candidate expert on the residual, training the alias keys, and observing that the
pool grows more slowly than V7's four-per-task — is **NOT RUN**, and §20.3
records that the pool-efficiency claim is a projection until it is. Nothing in
this report should be read as evidence that V8-B works.

**Method acceptance: PARTIALLY VERIFIED.** The identification half (Goal A) and
the retrieval half of Goal B are demonstrated; the end-to-end continual-learning
benefit is unmeasured and the honest baseline-relative gain is modest.

### 25.3 The causal chain

Q1–Q6 as the specification orders them, restated and answered from this
campaign's artefacts rather than from expectation:

| # | Question | Answer | Where |
| --- | --- | --- | --- |
| Q1 | Can the old experts be reused at all? | **Yes, but less than the all-experts number suggests** — 73 and 40 samples of 256 are reached by a genuinely historical expert route; 42 and 46 more need no expert at all | §14 |
| Q2 | Does a V8 policy beat the baselines on the official metric? | **Against the NLL-oracle ceiling, yes** (94.14 / 87.11, gap closed 0.957 / 1.065). **Against V7 under a history-only scope, no** (44.92 / 33.59 vs 67.97). **Against the frozen base, yes** (+28.51 / +15.62) | §15 |
| Q3 | Would giving a historical expert a second key on the current task help? | **Yes, substantially** — historical recall@1 0.164 → 0.603 and 0.100 → 0.800, every winning key an alias | §17 |
| Q4 | When a sample is not solved, is it capability or retrieval? | **Capability, and on Task 3 this is certified** — all 12 visible historical experts were scored on all 170 Residual samples. On Task 4 it is a bound: only 8 of 16 were recalled | §16, §23 |
| Q5 | What would the pool-efficiency benefit be? | **Not measured.** V8-A trains nothing; §20 measures the opportunity and the router deficit, and leaves the reduction to V8-B | §20 |
| Q6 | What would it cost to find out? | **~3 h for a Task-1 pilot** with a 500-sample teacher, on §21's measured 1.02–1.74 s/route; 6.2–10.6 h for a 2,000-sample teacher | §19, §21 |

### 25.4 What would change this verdict

The verdict is **PARTIALLY VERIFIED**, not FAIL, because the two claims that
failed are claims the campaign was not designed to test, and the claim it *was*
designed to test passed. That distinction is falsifiable, and these are the
results that would overturn it:

* **If a V8-B run trains alias keys and historical recall@1 does not move**, then
  §17's gain was an artefact of the counterfactual's construction (the centroid
  is built from the same split it is evaluated on, which §17 states as an upper
  bound) and the method's central mechanism is unproven → **FAIL**.
* **If a V8-B run moves recall but not the official metric**, the bottleneck is
  the pair/composition rule or the selection rule, and the remedy is §23's
  branch E, not more key learning → **PARTIAL, different reason**.
* **If the history-only policy cannot be made to beat V7 even with trained keys**,
  the reuse premise is wrong for this pool and the correct conclusion is that
  V7's four-new-experts-per-task is already the right policy → **FAIL**.
* **If answer parity (§12.1) comes back DIFFER rather than MATCH**, then §5's
  claim that V8's engine reproduces V7's answers under V7's route is false, and
  every V8 metric in §15 becomes suspect → the verdict would have to be revisited
  before any V8-B work.

None of these has been observed. The first three require V8-B, which is why the
recommendation is a pilot and not a conclusion.

### 25.5 Recommendation: NO-GO for the full loop, GO for one pilot

**NO-GO** for the six-task V8-B loop as specified. The two reasons are measured,
not cautious: (a) the history-only policy underperforms V7 on both tasks
(44.92 / 33.59 vs 67.97), so a full loop started today would spend its GPU budget
to reproduce a deficit; and (b) the reuse rate *falls* from Task 4 to Task 3 while
the residual set grows (141 → 170), which is the opposite of the monotone
improvement a reuse-only method needs — each later task needs the *candidate*
half of V8 more, and that half is exactly what is unmeasured.

**GO** for one Task-1 V8-B pilot (~3 h, §19.5), and its decision rule should be
fixed before it runs:

1. **Retrieval moves.** Trained alias keys raise *task-1, history-only* solver
   recall@1 above §17's untrained centroid value on the same split. If it does
   not, §25.4's first branch fires and the method fails.
2. **The metric moves with it.** The task-1 policy closes a positive fraction of
   the gap to the frozen base's pair-diagnostic ceiling — a *history-only* gap,
   computed the way §15 computes it, not the all-experts one.
3. **The candidate earns its place.** The residual samples the candidate is
   trained on are samples no historical expert solves, measured by the §16 audit,
   not assumed from the teacher's state label.
4. **Nothing regresses.** The freeze ledger reports `UNCHANGED` for the
   historical LoRA and the committed origin keys, and V7's Task-0/1 answers are
   unchanged.

Task 1 is the right pilot task for the reason §19.5 gives: Task 0 has no history
and would test nothing new. If all four conditions hold, the full loop is a
**GO** with §21's cost model as the budget; if the first fails, stop and report a
negative result rather than resizing the pilot until it passes.
