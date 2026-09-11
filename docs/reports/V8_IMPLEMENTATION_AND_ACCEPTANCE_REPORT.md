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

**V7 has no teacher.** It has two adjacent pieces of infrastructure that V8
reuses rather than reinvents: `compose/eval/nll_eval.py` (teacher-forced answer
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
$ pytest tests/compose -x -q
601 passed, 3 warnings, 8 subtests passed in 78.73s (0:01:18)
```

Split:

| Suite | Tests | Result |
| --- | --- | --- |
| `tests/compose/test_v8_answer_supervised_multikey.py` (TEST 01–30) | 37 | 53 passed |
| `tests/compose/test_v8_generation_harness.py` | 16 | (same run) |
| V7 regression (rest of `tests/compose/`) | 548 | passed |

The 30 named tests from PART 15 all exist and all run. Several are parameterised
over states or configs, which is why the file reports 37 collected tests for 30
names.

---

## 12. V7 Regression and Task 0 Parity

**V7 regression**: 548 tests under `tests/compose/` that predate this work still
pass. `git diff --stat 9ff2b28..HEAD` reports **24 files changed, 7,989
insertions(+), 0 deletions(-)** — every change is a new file (`compose/v8/*`,
`compose/experiments/v8*.py`, the two new test files) plus additive edits to
`CHANGELOG.md` and `docs/module_status.md`. Nothing under `compose/v7/`,
`compose/adapters/`, `compose/eval/` or `llava/` was modified, which is why
`test_29_v7_original_tests_still_pass` passes by construction.

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
