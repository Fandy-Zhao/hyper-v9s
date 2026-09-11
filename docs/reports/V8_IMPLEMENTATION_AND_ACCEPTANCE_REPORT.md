# Hyper-LLaVA V8 — Implementation and Acceptance Report

**Method**: V8 Answer-Supervised Multi-Key Expert Pool
**Branch**: `exp/v8-answer-supervised-multikey`
**Audit base**: `main` @ `9ff2b285b2cb4ac131519d9a499ff37fd25a55f3`, working tree clean
**Written**: 2026-09-11

This report answers PART 49's causal chain Q1–Q6 with artefacts produced by real
runs. Where an experiment was not executed, the section says **NOT RUN** or
**NOT TESTED** and gives the cost reason; nothing is inferred into a result.

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
$ pytest tests/compose -q
606 passed, 3 warnings, 8 subtests passed in 75.74s (0:01:15)
```

Split:

| Suite | Collected | Result |
| --- | --- | --- |
| `tests/compose/test_v8_answer_supervised_multikey.py` (TEST 01–30) | 42 | all pass |
| `tests/compose/test_v8_generation_harness.py` | 16 | all pass |
| V7 regression (rest of `tests/compose/`) | 548 | all pass |

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
pass. `git diff --stat 9ff2b28..HEAD` reports **30 files changed, 9,617
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

**Route multiset identical — MATCH.** Per-sample answer parity (regenerate the
route through V8's engine and diff the answer text against V7's own generated
answers) is executed by the same script with `--generate`; its result is recorded
in §12.1 below.

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

| Task | Scope | BaseOnly | ≥1 solved historical single | solved only by a pair | unresolved (Residual) |
| --- | --- | --- | --- | --- | --- |
| 4 (CLEVR-Math) | all-experts | 42 (16.41 %) | 178 (69.53 %) | 21 (8.20 %) | 15 (5.86 %) |
| 4 (CLEVR-Math) | history-only | *run 2 pending* | | | |
| 3 (IconQA) | all-experts | *run 3 pending* | | | |
| 3 (IconQA) | history-only | *run 4 pending* | | | |

Read the `all-experts` row for Task 4 with the scope caveat attached: experts
16–19 are Task 4's *own* experts and did not exist when Task 4 was learned, so
"reused" there includes self-reuse. The history-only row is the honest
continual-learning number, and §20 splits the two.

What the row already establishes: **the capability is there and the teacher
finds it.** On a task whose V7 actual route scores 67.97, 69.5 % of samples are
solved by reusing a single expert that already exists in the pool, and only
5.9 % are unresolved after base, every recalled single and every valid pair have
been tried. The residual 15 samples still keep a best historical context (§17):
11 of them a single expert, 4 a pair.

---

## 15. V8-A Routing Results

**Table 1 — official metric.** `V7 Metric` is the V7 actual route on the same
split (`v7_final_pool_pair_upper_val256_20260907`), `Teacher UB` is that
campaign's answer-NLL oracle pair (the ceiling V8-A can reach without training),
`V8 Metric` is the V8 policy's route replayed through the frozen model and
scored by the official evaluator, and `GapClosed = (V8 − V7) / (UB − V7)`.

| Task | Scope | V7 Metric | Teacher UB | V8 Metric | V8 − V7 | GapClosed |
| --- | --- | --- | --- | --- | --- | --- |
| 4 (CLEVR-Math) | all-experts | 67.97 | 95.31 | **94.14** | **+26.17** | **0.957** |
| 4 (CLEVR-Math) | history-only | *run 2 pending* | | | | |
| 3 (IconQA) | all-experts | *run 3 pending* | | | | |
| 3 (IconQA) | history-only | *run 4 pending* | | | | |

The teacher's own verdict count and the official scorer's count agree exactly
(`reconciliation.difference == 0`: 241 samples solved by the teacher, 241 scored
correct by the evaluator). That is the check that the metric adapter used for
`M` and the official metric used for the headline number are the same function —
if they disagreed, the two halves of this report would be describing different
runs.

**One quantified upper-bound caveat.** STEP C scores the pool-wide candidate
union (10 experts on Task 4) rather than each sample's own Top-8, so a sample
can be solved using an expert its own recall window never contained. Three
samples (1.17 %) were solved that way
(`selected_route_outside_own_recall_window: 3` in `v8a_recall_audit.py`), so a
router restricted to each sample's own Top-8 would score at most
`(42 + 196) / 256 = 92.97 %` here rather than 94.14 %. The union is the
teacher's *search* budget, not a leak of answers — but the number is an upper
bound and the size of the bound is 3 samples, not "unknown".

**Table 2 — teacher recall.** `V7 TeacherRecall@k` is the recall of the current
single-key (`1 Expert : 1 Key`) router: the fraction of samples with a solving
expert inside the router's Top-k, measured over the solver set defined in §16.
`V8 TeacherRecall@k` is the same measurement after a current-task alias key is
created by the specified centroid initialisation — the §17 counterfactual, since
V8-A itself trains nothing. `SetExact` is discussed below.

| Task | Scope | V7 R@1 | V7 R@2 | V7 R@4 | V7 R@8 | V8 R@1 | V8 R@2 | V8 R@4 | V8 R@8 | SetExact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | all-experts | 0.719 | 0.849 | 0.950 | 0.990 | *§17* | *§17* | *§17* | *§17* | *§17* |
| 4 | history-only | *run 2 pending* | | | | | | | | |
| 3 | all-experts | *run 3 pending* | | | | | | | | |
| 3 | history-only | *run 4 pending* | | | | | | | | |

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
| 4 | history-only | *run 2 pending* | | | | | | | | | | | |
| 3 | all-experts | *run 3 pending* | | | | | | | | | | | |
| 3 | history-only | *run 4 pending* | | | | | | | | | | | |

Three readings, in order of how much they matter:

1. **Retrieval is not the bottleneck.** 71.9 % of samples that any expert can
   solve have a solving expert ranked **first** by the current single-key router,
   and 99.0 % have one inside the Top-8 window. The V7 keys already rank
   capability well; what they cannot do is say *which task* a key belongs to.
2. **Selection ordering is weaker than retrieval.** The expert the teacher
   *selects* sits first only 33.2 % of the time (`selected R@1`). The gap between
   0.719 and 0.332 is not a failure — it is the NLL tie-break and the
   lexicographic rule choosing among several solvers (§6) — but it is what the
   alias key is supposed to sharpen, since a key that encodes "this expert solves
   *this* distribution" should rank its expert above equally-capable alternatives.
3. **Capability failure, not retrieval failure, is what remains.** Only 2 samples
   (0.8 %) have a solver present in the visible order but outside the Top-8
   window (`capability_present_but_outside_recall_window: 2`; experts 12 and 13
   at ranks 12 and 9). The 15 Residual samples are cases where no *scored*
   expert solved at all. Those are lower bounds: an expert that was never
   recalled was never tried, which is exactly what a wider M or a better key
   could change — the audit states this in its own output rather than leaving it
   to be assumed.

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

## 21. Efficiency

All figures below are measured, and each cites the artifact it came from.

### 21.1 Teacher cost (Task 4, all-experts, 256 samples)

| Quantity | Value | Source |
| --- | --- | --- |
| End-to-end wall time | 3,917 s = 65.3 min | `COMPLETE.json:duration_seconds` |
| Model load | 102.7 s | `analysis.json:timings_seconds.load_model` |
| Routes generated | 2,612 (256 base + 2,140 singles + 216 pairs) | `analysis.json:generation.generated` |
| Generation cache hits | 0 | `analysis.json:generation.cache_hits` |
| Seconds per route | 1.08 s | 2,356 routes (STEP C singles + pairs) between STEP B at 341.9 s and pair scoring at 2,894.1 s |
| NLL evaluations | 2,396 live + 216 reused | `analysis.json:generation` |
| Seeded pair NLLs | 64,768, replayed exactly (`max_abs_diff 0.0`) | `COMPLETE.json:seed_verification` |
| Peak allocator memory | 14.8 GiB | `COMPLETE.json:peak_gpu_memory_bytes` |
| Resident VRAM (`nvidia-smi`) | 16,910 MiB | sampled during run 2 on the same GPU |
| GPU utilization | 14–34 % (mean ≈25 %), 85–95 W | sampled 5× at 3 s intervals |

The GPU is **not** the bottleneck: at ~25 % utilization the teacher is bound by
per-route decode/Python overhead, not by tensor compute. Two consequences worth
recording, because they decide how V8-B should be run rather than being trivia:
cost scales with the **number of routes**, so M and K_s are the cost levers
(deeper recall is linearly more expensive), and the same GPU can serve other
work alongside a run of this shape.

### 21.2 Training cost (the comparison V8-B would have to beat)

V7's own Task-4 training is 621 optimizer steps at 34.55 s/step = **5 h 58 min**
for 39,743 samples, i.e. 0.54 s/sample at gradient-accumulation 64 (from
`…/task4/logs/training.log` and `metrics/train_steps.rank*.jsonl`). V8-B adds
the teacher on top of that, which is why the pilot in §19.5 is scoped at 500
teacher samples rather than 2,000.

**Cross-check.** §19.2 estimated a 2,000-sample teacher at 6–8 h single-GPU
*before* this campaign ran, from the V7 logs and the legacy `task_run.py`
budget. The measured 1.08 s/route now predicts 2,000 × 11 ≈ 22,000 routes ≈
6.6 h. Two independent derivations agree, so the pilot's cost line is not a
guess.

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

## 24. Acceptance Checklist

Every row cites the artefact that decides it — a test name, a file:line, or a
section of this report. "Test" means it is decided by a test that runs in the
601-test suite (§11); "report" means it is decided by a measured run.

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
