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
$ pytest tests/compose -q
607 passed, 3 warnings, 8 subtests passed in 74.98s (0:01:14)
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

### 12.1 Answer parity: attempted, crashed, retry queued

The route check needs no generation, so it stands on its own. The *answer*
check — regenerate each sample through V8's engine under the identical route and
diff the text against the answers V7 actually produced — has **no result yet**,
and the report says so rather than leaving a blank. Recorded facts:

| Fact | Evidence |
| --- | --- |
| `task0_parity.json` contains `route_parity` and no `answer_parity` key | the file (mtime 07:56:12 in `experiments/runs/0911_v8a_formal/`) |
| That file therefore came from a route-only invocation | `v8_task0_parity.py:208` writes the output *once, at the end*; a crashed `--generate` run writes nothing |
| One answer-generation attempt crashed | `task0_parity.log:10-75`, `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR` inside `prepare_inputs_labels_for_multimodal` (the vision conv), i.e. the shared device was too full for cuDNN to get a workspace |
| The retry is queued, not lost | `run_deferred_gpu_work.sh:28-33` waits for the driver to release a GPU, requires >= 19,000 MiB free, then re-runs the same command with `--generate` |
| The successful route-parity result cannot be destroyed by a failed retry | the module writes atomically at the end, so a crash leaves the existing file untouched; a copy is preserved as `task0_parity_route.json` |

This is a **BLOCKER-class gap in evidence, not in code**: the parity harness is
written, tested and has already produced a route MATCH; what is missing is one
GPU slot on a contended box. §12.1's conclusion stays "route parity verified,
answer parity pending" until `task0_parity.json` acquires an `answer_parity`
key.

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
| 3 (IconQA) | all-experts | *run 3 pending* | | | |
| 3 (IconQA) | history-only | *run 4 pending* | | | |

Read the `all-experts` row for Task 4 with the scope caveat attached: experts
16–19 are Task 4's *own* experts and did not exist when Task 4 was learned, so
"reused" there includes self-reuse. The history-only row is the honest
continual-learning number, and §20 splits the two.

**This is the most important comparison in the V8-A campaign, and it goes
against the all-experts headline.** With Task 4's own experts removed, the
number of samples any historical expert can solve drops from 178 to 71
(69.53 % → 27.73 %), pair-only solves collapse from 21 to 2, and the Residual
set grows from 15 to 141. In other words **roughly six of every seven samples
V8-A appeared to "reuse" on Task 4 were being solved by the task's own experts**
— which is not reuse at all, because those experts do not exist at learning
time. The `v8a_scope_gap.py` join gives the exact split:

| Quantity | Samples | Rate | Meaning |
| --- | --- | --- | --- |
| Historical reuse (solved with the historical pool) | **115** | **44.92 %** | reachable from the pool as it stood at learning time: 42 base-only + 73 expert routes |
| — of which base-only (needs no expert at all) | 42 | 16.41 % | the base already answers correctly |
| — of which a historical expert route | 73 | 28.52 % | genuine cross-task reuse |
| Self-reuse | 127 | 49.61 % | solved only with Task 4's own (or a later task's) experts |
| Unresolved in both scopes | 14 | 5.47 % | no expert in the whole pool solves it |

The three buckets sum to 256 exactly (`bucket_sum == samples_compared`), and the
two scope totals reproduce the runners' own counts — `solved_in_all_scope: 241`
and `historical_reuse: 115` are exactly the `analysis.json` solve counts for the
all-experts and history-only runs. Those two cross-checks are why this table is
the one the report quotes: the script's first version counted `BaseOnly` samples
as unresolved and reported 57 unresolved samples instead of 14, and the
disagreement with `analysis.json` is what caught it.

So the honest answer to "can the old experts be reused?" is **yes, for 28.5 % of
this task's samples, plus 16.4 % that need no expert at all** — a real effect
worth building on, but less than half the size the all-experts number suggests.
The residual 15 samples still keep a best historical context (§17): 11 of them a
single expert, 4 a pair.

One cross-scope anomaly is reported rather than smoothed: `v7_t4_val_211` is
solved in the history-only scope but *not* in the all-experts scope, which at
first looks impossible — the history pool is a strict subset. It is not a
counter bug and it is verified from both runs' artefacts: in the history scope
the pair `{13, 0}` solved the sample, while in the all-experts scope Task 4's own
experts occupy ranks 1–4 of the Top-8 window
(`[19, 17, 16, 18, 15, 14, 13, 0]`), so `{13, 0}` never entered the K_s = 4 pair
shortlist and was never composed. A **wider pool can hide a solver from the pair
shortlist** — the same crowding effect §23 identifies as a bottleneck, and the
reason this report does not treat `historical ⊆ all` as an identity.

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
| 4 (CLEVR-Math) | history-only | 67.97 | 95.31 | 44.92 | −23.05 | −0.843 |
| 3 (IconQA) | all-experts | *run 3 pending* | | | | |
| 3 (IconQA) | history-only | *run 4 pending* | | | | |

**The history-only row is not a V8-vs-V7 result, and must never be quoted as
one.** The comparison is not like-for-like: V7's 67.97 is produced by a router
that is *allowed* to use experts 16–19 (Task 4's own, which exist in the pool
precisely because Task 4 has already been trained), while the history-only V8
policy is forbidden from using them by construction. The row measures **how far
the frozen historical pool gets on Task 4 with no task-specific experts
available** — 44.92, i.e. 23 points below what V7 achieves *with* them. Read that
way it is one of the campaign's most useful numbers: it is the ceiling for
"reuse only", and the gap between it and 67.97 is the headroom a V8-B Candidate
Expert would have to close. What it is not is evidence that V8's routing is worse
than V7's, because the two policies were not given the same pool.

The like-for-like comparison is the all-experts row (+26.17, 95.7 % of the
teacher's headroom closed), and even that is inflated by self-reuse — of its 241
solved samples, 126 are self-reuse and 115 are historical (§14). The honest
summary of Task 4 is therefore: **+26.17 like-for-like, of which the historically
reusable part is 44.92 points**.

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
| 4 | history-only | **0.274** | **0.452** | **0.740** | **1.000**† | *§17* | *§17* | *§17* | *§17* | *§17* |
| 3 | all-experts | *run 3 pending* | | | | | | | | |
| 3 | history-only | *run 4 pending* | | | | | | | | |

† **Degenerate by construction, and flagged rather than quoted.** In the
history-only run the pool holds 16 visible experts and the recall window is
Top-8, while STEP C's candidate union for the 214 unsolved samples happens to be
exactly those same 8 experts
(`tested_minus_recall_size_distribution: {"0": 256}` — the tested-minus-recall
size is 0 for all 256 samples). Every scored expert is therefore inside every
sample's own window, so `R@8 = 1.000` carries no information about the router.
The informative numbers are `R@1 = 0.274` and `R@2 = 0.452`.

**That 0.274 is the strongest single piece of evidence for the alias-key idea in
this campaign.** With the task's own experts present, the current keys rank *some*
solver first 71.9 % of the time — but that is largely the self-experts ranking
themselves well on their own distribution. Restricted to historical experts, the
same keys put a solver first only 27.4 % of the time: **the fixed-query geometry
knows which expert can do the job far less reliably than the all-experts number
suggests.** §16 shows this is not because the capability is absent. §17 measures
what a current-task alias key would buy against exactly this gap.

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
| 3 | all-experts | *run 3 pending* | | | | | | | | | | | |
| 3 | history-only | *run 4 pending* | | | | | | | | | | | |

† See §15 Table 2's footnote: `R@8 = 1.000` is degenerate here, because in the
history-only run the candidate union *is* each sample's Top-8 window.
`capability_present_but_outside_recall_window` is correspondingly 0 and
`selected_route_outside_own_recall_window` is 0 for this run — there is no
wider union for a route to hide in — so §15's upper-bound caveat does not apply
to the history-only row at all.

Three readings, in order of how much they matter. The first is the one that
changed when the history-only run finished: **the all-experts row flatters the
router, because the experts that dominate its top ranks are the task's own.**

1. **Retrieval is the bottleneck for historical experts, and the all-experts
   number hid it.** Over the whole pool, 71.9 % of solvable samples have a solver
   ranked **first**. Restricted to historical experts, the same keys put a solver
   first only **27.4 %** of the time (`solver R@1`, history-only row), and only
   45.2 % inside the Top-2. The current fixed-query geometry is good at ranking an
   expert on the distribution it was trained on and weak at ranking a *historical*
   expert on a new distribution — which is precisely the deficit a current-task
   alias key exists to close, and the reason §17's measurement matters more than
   the headline metric.
2. **Selection ordering is weaker than retrieval, in both scopes.** The expert
   the teacher *selects* sits first only 33.2 % of the time over the whole pool
   and 16.4 % restricted to history. The gap is not a failure — it is the NLL
   tie-break and the lexicographic rule choosing among several solvers (§6) — but
   a key that encodes "this expert solves *this* distribution" should rank its
   expert above equally-capable alternatives, and today it does not.
3. **Past the router, what remains is capability, not retrieval.** In the
   all-experts scope only 2 samples (0.8 %) have a solver in the visible order but
   outside the Top-8 window (`capability_present_but_outside_recall_window: 2`;
   experts 12 and 13 at ranks 12 and 9), and in the history-only scope that count
   is **0** by construction. The Residual set is what the router cannot fix: 15
   samples with the whole pool available, **141 (55.1 %) with only the historical
   pool** — samples no recalled expert solved at all. Those are lower bounds: an
   expert that was never recalled was never tried, which is exactly what a wider
   M or a better key could change — the audit states this in its own output
   rather than leaving it to be assumed.

The 141 history-only Residual samples are the load-bearing number for V8's
second half: they are simultaneously the evidence that historical reuse alone is
insufficient and the precise worklist a V8-B Candidate Expert would be trained
on. V8-A cannot say whether such an expert would learn them; it can say how many
there are and that they are not a retrieval artefact.

---

## 17. Alias Key Analysis

*This section is filled by `compose/experiments/v8a_alias_keys.py`, which runs
after the campaign on the exported teacher caches; it is the last analysis step
in the queue (`experiments/runs/0911_v8a_formal/run_alias_analysis.sh`) and had
not completed when §14–§16 were written. Its result is the §15 Table 2 `V8 R@k`
column and the answer to causal-chain Q3.*

---

## 18. Sample Case Studies

Source: `experiments/runs/0911_v8a_formal/cases_task4.json`, produced by
`compose/experiments/v8a_cases.py`, which joins the two scope runs, the frozen
Task 4 validation questions and the answers V7 actually generated in its pair
diagnostic. Totals for the run: **V7 correct 174/256, V8 (history-only) correct
115/256** — the same 67.97 % and 44.92 % as §15, recomputed here per sample by
the metric adapter rather than read from a summary, so the case list and the
headline numbers cannot drift apart.

The four classes below overlap by construction (a sample can be both "V7 miss,
V8 fix" and "residual with context"), so the counts are not a partition and are
not summed anywhere in this report.

### A. V7 missed it, V8 fixed it — 37 samples

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

`v7_t4_val_107` is the other flavour of class A: V7 is wrong, and V8's
**`BaseOnly`** state — no expert at all, `selected_experts: []` — is right. The
minimal-capacity rule (PART 8: base solves → stop) is not just an efficiency
device; here it converts a V7 miss into a hit by declining to route.

### B. Reusable, but not by a historical expert — 127 samples

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
[1]`. This class is the quantitative core of the scope critique, and it is also
the worklist V8-B would draw on: 127 samples that a *new* candidate expert
would have to cover rather than reuse.

### C. Several experts solve it — the IGNORE rule in action

The case file's `alternative_solved` counter reads `positives > 1`, which only a
selected *pair* produces, so it actually counts pair-solves (21 in the
all-experts run, 2 in history-only) and not alternative singletons. The right
measurement for "multiple solved experts" is the three-valued label itself,
taken over the history-scope records:

| Quantity | Value |
| --- | --- |
| Samples carrying ≥1 `ignore` label | **78 / 256** |
| Total `ignore` labels | 401 |
| Total `positive` labels | 75 |
| Total `negative` labels | 1,572 |
| `ignore`-per-sample histogram | `{0: 178, 1: 19, 2: 9, 3: 5, 4: 2, 5: 1, 8: 42}` |

`v7_t4_val_10` is the illustration: state `Reuse1`, `positive: [13]`,
`ignore: [0, 3, 12]`. Three experts that also solve the sample are labelled
IGNORE rather than NEGATIVE, which is PART 9's prohibition ("禁止：E5 =
negative") doing real work — labelling them negative would push their keys away
from a query they demonstrably answer. The 42 samples with eight IGNOREs are the
`BaseOnly` samples, where nothing should be pushed either way.

### D. Residual with old context, and what a new expert would face — 141 samples

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

### What the four classes say together

Class A shows the mechanism works and is worth training for. Class B and D
together are the honest limit: of the 256 samples, 115 are reachable from the
frozen historical pool and 127 are not, and the ones that are not include
samples V7 already gets right. That is the case for V8-B being a *residual*
learner on top of frozen context rather than a replacement router — and it is
also why §19.5 recommends a pilot rather than a full loop.

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

| Quantity | V7 | V8-A task 4, **all-experts** scope (measured) | Source |
| --- | --- | --- | --- |
| New experts trained for the task | 4 | 0 | §20.1 / `run_config.json` |
| New committed keys for the task | 4 | 0 | §20.1 / `COMPLETE.json` |
| Keys per expert | 1.00 (24/24) | 1.00 committed; alias-key gain measured in §17 | `v7_keys.pt` / §17 |
| Samples the base model already solved | — | 42/256 = 16.41 % | `analysis.json:solution_counts` |
| Samples served by one expert in the pool | — | 178/256 = 69.53 % | `analysis.json:solution_counts` |
| Samples served by a pair of experts in the pool | — | 21/256 = 8.20 % | `analysis.json:solution_counts` |
| Residual samples (no route found) | — | 15/256 = 5.86 % | `analysis.json:solution_counts` |
| Reuse rate (any non-empty route) | see §15 Table 2 (`V7 Recall@1 = 0.719` measures the same idea through V7's own router) | 199/256 = 77.73 % | `analysis.json:solution_counts` |
| Residual ratio | — | 5.86 % | `analysis.json:solution_counts` |

**The scope qualifier in that header is not cosmetic.** The `all-experts` run
lets the teacher route to Task 4's *own* committed experts (16–19) as well as to
the historical ones, and a task's own experts are exactly what is unavailable
while that task is being learned. The pool-wide number must therefore be split
before any part of it can be called *historical* reuse, which is what
`v8a_scope_gap.py` does by joining the two scopes per sample. The scale of the
correction is already visible in the teacher's own progress line: in the
all-experts run **178** of 214 unsolved samples stopped at a solved single, while
in the history-only run only **71** did — so of order a hundred samples that look
like reuse in the all-experts scope are in fact solved by the task's own
experts. The historical-reuse and self-reuse rates for Task 4 are reported in
§15/§23 from the joined runs; nothing in this table may be quoted as historical
reuse without that split.

The V7 comparison in that table is deliberately drawn from §15 Table 2 rather
than from a new count: V7 does route over the whole committed pool, so
"fraction of samples V7's own router sends to an expert" is already measured
there, and re-deriving it here from a different artifact would risk two numbers
for one quantity.

### 20.3 The efficiency claim V8 is *designed* to make, and its status

V8's thesis is that `new_experts_per_task` should fall because samples a
historical expert already solves should not motivate a new LoRA. That claim has
three parts, and only the first is measurable from this campaign:

1. **The opportunity exists.** 77.73 % of Task 4's samples are routed by an
   answer-supervised teacher to some expert in the committed pool, and 16.41 %
   need no expert at all. This is measured (§20.2). How much of that 77.73 % is
   *historical* rather than *the task's own* experts is the scope split, and it
   is smaller — see the split in §15/§23.
2. **V7 misses part of that opportunity.** Measured in §15 Table 2 and analysed
   in §16: the solvers exist inside the visible Top-8 window for 196/256
   samples, while V7's own router recalls them at rank 1 for 0.719 of solver
   samples. The gap between "the pool can" and "the router picks" is the room
   V8's alias keys are meant to close.
3. **V8 would therefore train fewer experts.** **Not measured.** Turning the
   reuse rate into a reduction requires choosing which samples still need a
   Candidate Expert, training it on the residual, and observing that the pool
   grows more slowly than V7's four-per-task. That is V8-B, and §19.5
   recommends a one-task pilot rather than asserting the result.

The correct reading of this section is therefore: V8-A establishes parts 1 and 2
with numbers, and leaves part 3 open. The pool-efficiency benefit is a
*projection* until a V8-B run produces a task whose committed expert count is
below V7's four.

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

## 23. Failure Analysis

"效果不好" is not an explanation. This section attributes the Task 4 result to
one of the six candidate bottlenecks the specification names (A–F), and says for
each whether it is **measured**, **measured as small**, or **not exercised by
V8-A at all**. All figures are from the two Task 4 runs; a bottleneck that only
Task 3 could reveal is marked as such rather than generalised.

| # | Bottleneck | Status on Task 4 | Evidence |
| --- | --- | --- | --- |
| A | Historical capability (the pool simply cannot do it) | **Primary constraint** | 141/256 samples (55.1 %) have no solver among all tested historical candidates; only 73 expert-route solves + 42 base-only are reachable |
| B | Candidate retrieval (the solver exists but is outside the recall window) | **Small, and zero in history-only** | `capability_present_but_outside_recall_window: 2` in the all-experts scope (experts 12, 13 at ranks 12 and 9); **0** in history-only, where the candidate union *is* the Top-8 window |
| C | Key representation (the keys rank the right expert badly) | **Measured, and the largest actionable gap** | solver R@1 = 0.274 / R@2 = 0.452 in history-only, vs 0.719 / 0.849 with the task's own experts present |
| D | Key optimisation (alias keys not learning) | **Not exercised by V8-A** | V8-A trains nothing; §9.1's hinge-gradient defect was found and fixed before the campaign, so the optimiser path is repaired but unmeasured |
| E | Pair composition (the right pair is never composed) | **Measured as small, with one verified case** | 2 pair-only solves in history-only; `v7_t4_val_211` is solved by `{13, 0}` in the history scope but never composed in the all-experts scope because the K_s = 4 shortlist was crowded by self-experts |
| F | New-expert learning (the candidate fails to absorb the residual) | **Not exercised by V8-A** | Requires V8-B; §19.5 scopes the pilot |

**Why A is the primary constraint and not B.** The two are easy to confuse,
because both show up as "the sample is not solved". They separate cleanly here:
B would mean the solver is in the pool but the router never surfaced it, whereas
in the history-only run every scored expert was inside every sample's own window
(`tested_minus_recall_size_distribution: {"0": 256}`), so the router was not
filtering anything out — the 141 Residual samples were scored and simply had no
solver. The count is still a lower bound, because only recalled candidates are
ever scored and an expert outside the recall was never tried. That is exactly
what the queued `v8_full_pool_recall.py` audit settles: it re-tests **every**
visible expert on the Residual samples, which converts "no solver among the
candidates" into "no solver in the pool" (or finds the misses). Until it
finishes, A is stated as "no solver among the tested candidates", not
"no solver exists".

**Why C is the most actionable finding.** A is a statement about the frozen pool
— nothing in V8 can change it without training, and that training is V8-B. C is
different: the capability is present (73 samples are solved once the right
expert is selected), but the current single-key geometry ranks a solver first
only 27.4 % of the time on a new task's distribution. That is the gap alias keys
are defined to close, and it is measurable without any training, which is why
§17 is the decisive follow-up rather than an optional one.

**What would falsify this attribution.** If a V8-B run produced alias keys and
recall-at-1 did not move, C would be wrong and the cause would be D (the
optimisation) or the fixed-query geometry itself — the query is frozen and
detached (§3.1), so no key can encode information the query does not carry. If
alias keys did move recall but the metric did not, the bottleneck would be E or
the selection rule. The report keeps those branches distinct rather than
treating a poor final number as evidence about any one of them.

**Failure that did occur, and is reported as such (PART 46 item 20).** Two
things in this campaign went wrong rather than merely underperforming, and both
are recorded above instead of omitted: the three-valued key-target rule was not
total, and it had already written 33 mislabelled key targets into the
pre-fix run's artefact (§9.2, quantified by `v8a_label_audit.py`); and the
Task 0 answer-parity check crashed on a contended GPU and has produced no result
(§12.1). No run was re-run to make those disappear, and no artefact was
rewritten.

---

## 24. Acceptance Checklist

Every row cites the artefact that decides it — a test name, a file:line, or a
section of this report. "Test" means it is decided by a test that runs in the
607-test suite (§11); "report" means it is decided by a measured run.

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
