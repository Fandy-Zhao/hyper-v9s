# V7 → V8 Implementation Mapping

Status: Phase 1 deliverable. Written **before** any V8 code was added.
Audit base: branch `main` @ `9ff2b285b2cb4ac131519d9a499ff37fd25a55f3`
(working tree clean; V8 work happens on `exp/v8-answer-supervised-multikey`).

Every "Current behavior" cell below was read from the checked-out code, not from
the V8 specification. Where the specification's assumption and the repository
disagree, the repository wins and the disagreement is called out.

---

## 0. Audit summary — 15 questions from PART 2

| # | Question | Verified answer | Evidence |
|---|---|---|---|
| 1 | Fixed Query computation | `L2Norm(concat(LayerNorm(z_visual), LayerNorm(z_text)))`, parameter-free `F.layer_norm(weight=None,bias=None)`, returns `.detach()` | `compose/v7/query.py:44-56` |
| 2 | Query dimensionality | 768 visual + 768 text = **1536**; constructor rejects anything else | `compose/v7/query.py:29-35` |
| 3 | One key per expert? | **Yes.** `nn.ParameterDict` keyed by `str(expert_id)`; `add()` raises `duplicate expert key` | `compose/v7/pool.py:62,75-76` |
| 4 | Global Top-2 implementation | Single cosine matrix over all selectable keys, `torch.topk(k=2)`, deterministic tie epsilon `-arange*1e-7`, no old/new quota | `compose/v7/routing.py:44-70` |
| 5 | historical key / LoRA freeze | `freeze_historical()` / `freeze_all()` set `requires_grad_(False)` **and** `grad=None`; commit re-normalizes then freezes | `compose/v7/pool.py:114-128`, `commit.py:107-115` |
| 6 | Candidate expert count | exactly **4** per task; `initialize_candidate_keys` raises unless `count == 4` | `compose/v7/pool.py:22-31`, `config.py:29` |
| 7 | Candidate LoRA rank / alpha | rank **8**, alpha **16.0**, dropout 0.0; 19,988,480 params/expert over 32 layers × 7 projections | `config.py:29`, `docs/reports/V7_EXPERT_LEARNING_SUMMARY.md:34` |
| 8 | Key Loss | `mean over selected **current** keys of (1 - cos(q,key))`; queries detached; Old+Old samples contribute 0; historical & unselected current keys get no attraction | `compose/v7/training.py:96-120` |
| 9 | Answer Loss | teacher-forcing target-token mean NLL; the trainer receives the LLaVA loss, the audit helper is `teacher_forcing_token_nll` | `compose/v7/training.py:78-93`, `hf_trainer.py:163` |
| 10 | Pruning | usage → full-val route → true *remove-and-reroute* per candidate → removal gain (metric & loss) → redundancy rule → iterate one removal per iteration | `compose/v7/pruning.py:47-253` |
| 11 | RMS | per-(layer,module) κ scalars, 224 entries/expert, calibrated in S4, persisted as `rms_state.per_layer_kappa` with `mode='commit_frozen'`, applied via `apply_kappa_calibration` | `compose/v7/commit.py:97-115`, `compose/eval/load_compose.py:129-132` |
| 12 | checkpoint / commit / resume | atomic temp-dir + `os.replace`; commit writes filtered `.bin`, manifest, `v7_keys.pt`; `--resume-contract-rebind` continues after a fix | `compose/v7/commit.py:42-147`, `compose/experiments/v7_task_run.py` |
| 13 | Teacher / NLL infrastructure | `compose/teacher/*` = pure-python NLL-oracle search over **pre-computed** `AnswerNLL`; `compose/eval/nll_eval.py` = live-GPU producer of `{sample_id:{set_key:mean_nll}}` | agent audit, verified at `compose/teacher/scorer.py:11`, `compose/eval/nll_eval.py:40-164` |
| 14 | Existing checkpoints | full 6-task formal run complete; per-task `committed/` contains `.bin` + manifest + `v7_keys.pt`. 23 live experts, 1 pruned (#22) | `/data/ckpt/.../v7_gpu01_cached_query_formal_20260903/task{0..5}/committed/` |
| 15 | GPU / distributed runner | `compose/v7/gpu_plan.py`, `workers.py`; S3 train via `torch.distributed.run`; eval sharded by sample across GPUs; run root `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/` | `compose/v7/gpu_plan.py`, `compose/eval/sharding.py` |

### 0.1 Specification assumptions that do NOT hold in this repository

| Spec assumption | Repository reality | Consequence for V8 |
|---|---|---|
| "V7 使用 Answer Loss" implies a per-sample correctness signal exists | **There is no `solved` / `is_solved` / per-sample correctness concept anywhere** (grep: zero hits) | V8 must introduce the whole capability layer — that is genuinely new work, not a refactor |
| Per-task official metrics decompose per sample | Only ImageNet-R has a per-sample helper (`compose/eval/metrics.py:8`). ArxivQA/IconQA/CLEVR have none persisted, **but** their official metric *is* `pred.upper()==gt.upper()` per sample → exactly decomposable. VizWiz & Flickr30k use corpus-level Bleu/METEOR/ROUGE/CIDEr `Average` → **not** decomposable | V8 needs a metric adapter with two regimes: exact decomposition (tasks 0,1,3,4) and a documented per-sample **proxy + correlation analysis** (tasks 2,5) |
| VizWiz is a VQA soft-score task | In this pipeline VizWiz is routed to `eval_caption` → `Average` (COCO caption metric) | VQA soft-score logic from the spec does not apply; caption handling does |
| Teacher must be built from scratch | `compose/teacher/*` is mature but is **NLL-only**; it never produces a prediction | V8 reuses its cache/validity/hash machinery, not its decision rule |
| Inference must be free of answers | Already true and already audited — `eval_task.py` records explicit `answer_features_used/oracle_used/task_id_lookup_used` provenance booleans | V8 keeps this contract and extends it to the Multi-Key router |

---

## 1. Per-module mapping

Legend for **Reuse in V8?**: `as-is` = import unchanged · `wrap` = call behind a V8 adapter · `extend` = subclass/new sibling · `new` = V8 writes its own.

### 1.1 Query — `compose/v7/query.py`

| Field | Value |
|---|---|
| Current behavior | `FixedMultimodalQuery` = `L2Norm(concat(LN(z_v), LN(z_t)))`, 1536-D, zero parameters, `forward` returns a **detached** tensor. `full_train_task_center` = L2-normalized mean of all train queries, asserts `N == num_train_samples`. |
| Reuse in V8? | **as-is** |
| Required change | None. V8 must import the identical class so "fixed query unchanged" is provable by object identity of the module hash; `FixedQueryProvenance.module_hash = "v7_fixed_layernorm_concat_l2_v1"` is carried into every V8 cache provenance record. |
| Risk | Low. The `.detach()` is load-bearing: it guarantees no gradient path into the CLIP backbone even if a caller passes a live tensor. A V8 re-implementation would lose that guarantee — do **not** re-implement. |

### 1.2 Pool — `compose/v7/pool.py`

| Field | Value |
|---|---|
| Current behavior | `V7ExpertKeyPool` is an `nn.ParameterDict` where **the dict key is the expert id**; therefore expert and key are structurally fused (1:1). `metadata: Dict[int, {...}]` holds `origin_task`, `lifecycle`, `rms_state`. Lifecycle ∈ {current, historical, pruned}. |
| Reuse in V8? | **new** (`compose/v8/pool.py::MultiKeyExpertPool`) — V7 pool stays untouched |
| Required change | Decouple expert from key: `expert_id -> [key_id, ...]`. New metadata split: expert record (`origin_task`, `lora_path`, `rank`, `alpha`, `lifecycle`, `rms metadata`, `creation_task`) and key record (`key_id`, `expert_id`, `task_id`, `key_type ∈ {origin, task_alias}`, `lifecycle`, `support_count`, `teacher_gain`, `validation_gain`, `parameter`). |
| Migration | V7 `E_i -> K_i` becomes V8 `E_i -> K(i,origin)` with the **identical tensor** (already L2-normalized at commit) and identical `rms_state`. No re-normalization, no re-training. |
| Risk | **Medium-high.** The V7 1:1 invariant is assumed by `routing.py`, `pruning.py`, `inference.py`, `commit.py` and by `eval_task.py --v7-key-state` (which does `v7_pool.metadata[value]["origin_task"]` on a raw expert id). V8 must not mutate `V7ExpertKeyPool`; migration is a read-only export. |

### 1.3 Routing — `compose/v7/routing.py`

| Field | Value |
|---|---|
| Current behavior | `GlobalTop2Router.forward(queries, excluded)` → cosine over all selectable keys, `topk(k=2)`, padded to the production 4-slot `ComposeSelection`, plus `route_types` (OldOld/OldNew/NewNew) and a cross-task pair counter. Ties broken by a deterministic id-order epsilon. |
| Reuse in V8? | **wrap** — V8 needs a sibling, not a patch |
| Required change | Two-stage: (1) cosine over **all active keys**; (2) `max` aggregation **per expert**; (3) `topk` over **distinct expert ids**. Must additionally return `winning_key_id`, `winning_key_task`, `winning_key_type`, `winning_key_score`. The dedup invariant (`one expert cannot occupy two Top-2 slots`) must be a unit test, because it is exactly the bug the V7 data structure makes impossible to express. |
| Risk | **Medium.** V7 never has to dedup, so no existing test covers it. A naive `topk` over keys followed by `unique` would return **fewer than 2** experts for some samples; the correct construction is max-aggregate *before* topk. V8 keeps the V7 tie epsilon (re-expressed per expert id) so run-to-run determinism is preserved. |

### 1.4 Training — `compose/v7/training.py`

| Field | Value |
|---|---|
| Current behavior | `V7StepEngine.compute()` routes → `use_selection(...)` → answer loss → `selected_current_key_loss`. `backward()` returns **False** (skips backward entirely) when a micro-batch is Old+Old, then asserts historical key AND historical LoRA received no gradient. Unselected current keys are asserted gradient-free. |
| Reuse in V8? | **extend** |
| Required change | V8 replaces "Old+Old ⇒ whole-batch no-op" with **per-sample gradient gating** on three states (BaseOnly / Reuse / Residual) in one mixed batch. The existing `assert_historical_key_gradients_frozen` / `assert_historical_lora_frozen` / `adapter_checksums` helpers are reused verbatim as the freeze audit. |
| Risk | **High — this is the highest-risk area of V8.** V7's safety came from *batch-level* exclusion; V8 must achieve it at *sample level* while samples share one forward pass and one loss tensor. The spec's PART 20.5 explicitly demands a gradient-leakage test; V8 implements `L_answer_residual = Σ r_i·L_i / max(Σ r_i, 1)` and a dedicated mixed-batch test asserting candidate-LoRA grad is exactly zero for BaseOnly/Reuse-only batches, and non-zero once any Residual sample is present. |

### 1.5 HF trainer — `compose/v7/hf_trainer.py`

| Field | Value |
|---|---|
| Current behavior | Strict optimizer whitelist (current LoRA + current keys only); base/vision/projector/embedding/historical excluded; JSONL diagnostics; DDP graph discovery via a zero-valued key anchor. |
| Reuse in V8? | **extend** |
| Required change | Add alias-key parameters to the whitelist **only** for the current task, and add the per-sample gate. Emit the `TRAINABLE PARAMETER AUDIT` block required by PART 4. |
| Risk | Medium. The V7 "zero-graph anchor" trick exists so DDP can discover parameters in no-op steps; V8's mixed batches always have some live gradient, but a **pure** BaseOnly batch would otherwise have no graph — V8 keeps an anchor for that case. |

### 1.6 Pruning — `compose/v7/pruning.py`

| Field | Value |
|---|---|
| Current behavior | `CandidatePruner.evaluate(...)` = usage → full val route → for each candidate *actually remove it and reroute* → removal gain `metric`/`loss` → redundancy(`cosine ≥ threshold` AND `contribution ≤ min`) → remove one per iteration → `reason` strings. Requires the scorer to return `{"metric","loss"}`. |
| Reuse in V8? | **extend** |
| Required change | Two new decision surfaces: (a) **alias-key pruning** (`support_count`, teacher-recall gain, cosine redundancy vs. the same expert's other keys); (b) candidate-expert pruning reusing the V7 loop unchanged. All thresholds move into config (no magic numbers). |
| Risk | Medium. The V7 loop is O(pool²) generations; V8 adds alias keys on top. Alias pruning must be **cheap** (routing-level, not generation-level) or the cost explodes. |

### 1.7 Commit — `compose/v7/commit.py`

| Field | Value |
|---|---|
| Current behavior | Filters `.bin` tensors by expert id regex, rewrites manifest (`status=frozen`), rewrites `rms_calibration` to the retained set, writes `v7_keys.pt`, atomic `os.replace`, validates the temp dir before publishing. |
| Reuse in V8? | **new sibling** (`compose/v8/commit.py`) |
| Required change | Must publish the *split* expert/key structure (`experts[]` + `keys[]`), keep the V7-compatible `v7_keys.pt` written from the origin keys for backward-compatible readers, and preserve `rms_state` byte-identically. |
| Risk | Medium. `_validate_commit_directory` hard-requires `v7_keys.pt` with the V7 schema; V8 either keeps emitting it (preferred, for V7 regression) or must relax the validator. |

### 1.8 Inference — `compose/v7/inference.py`

| Field | Value |
|---|---|
| Current behavior | Rejects any current candidate in the pool, freezes everything, exposes `forward(z_v,z_t)` and a cross-task pair logger. |
| Reuse in V8? | **new sibling** (`compose/v8/inference.py`) |
| Required change | Multi-Key aggregation. Everything else (committed-only, no task id, no answer, no NLL) is preserved. |
| Risk | Low, **if** the "no ground truth" property is tested structurally rather than by review. V8 adds an AST-level test that the inference module never names `answer`/`target`/`ground_truth`/`nll`. |

### 1.9 Metric layer — `compose/eval/metrics.py`, `compose/eval/formal_ucit_eval.py`

| Field | Value |
|---|---|
| Current behavior | `imagenet_r_exact_match` (per-sample, ImageNet-R only); `_score_answers(root, task_id, eval_task_id, answers, annotation_file)` = the official corpus scorer (`Accuracy` for tasks 0,1,3,4 via `llava.eval.eval_deepseek_r1`; `Average` for tasks 2,5 via `llava.eval.eval_caption`). |
| Reuse in V8? | **wrap** |
| Required change | New `compose/v8/metric_adapter.py` with `MetricResult(score, solved, details)` and `evaluate(prediction, target, metadata)`. For tasks 0,1,3,4 the per-sample signal is `pred.upper() == gt.upper()` — verified at `llava/eval/eval_deepseek_r1.py:52` — so the adapter's aggregate **must reproduce `Result.text` Accuracy exactly**; V8 asserts this as a validation gate. For tasks 2,5 the corpus metric is not decomposable, so V8 uses an explicitly-named training-time proxy plus a correlation analysis (PART 9). |
| Risk | **Medium.** A silently wrong `solved` definition would invalidate every downstream number. Mitigated by the "adapter aggregate == official Result.text" reproduction test on real predictions. |

### 1.10 Teacher — `compose/teacher/*`

| Field | Value |
|---|---|
| Current behavior | Pure-python complexity-penalised oracle over **pre-computed NLL**; decision rule is NLL+cardinality penalty. `MIN_RECALL_AUDIT_RATIO` is declared and never used; `OracleConfig.include_eos` is unwired; `teacher.py:190` uses strict `>` while `oracle_set.py:61` uses `>=` for the same pair threshold. |
| Reuse in V8? | **wrap (cache/validity only)** |
| Required change | V8's teacher needs a *prediction*, which NLL-by-itself cannot produce. So V8 reuses: `compose/teacher/cache.py` (versioned sharded cache), `compose/teacher/validity.py` (split/leakage guards, incl. `validate_oracle_split` rejecting any name containing `test`), `stable_hash`, and the `OracleConfig` pair-shortlist parameters (`top_k_for_pair=4`, `max_pairs=6`) — and **replaces** the decision rule with base-first / correctness-gated / NLL-ranked selection. |
| Risk | Medium. Reusing the NLL rule would directly violate PART 46 item 1. V8 keeps the teacher package unmodified and imports only the mechanical parts; the V8 decision function lives in `compose/v8/teacher.py` with its own tests. |

### 1.11 Evaluation harness — `compose/eval/eval_task.py`, `nll_eval.py`, `sharding.py`, `load_compose.py`

| Field | Value |
|---|---|
| Current behavior | `eval_task` supports three routing modes: compose-router, `--v7-key-state`, and `--selection-manifest` (`{sample_id: {"global_top2":[a,b]}}`, requires exactly 2 distinct experts). Generation is greedy, `num_beams=1`, `max_new_tokens=128`. `nll_eval` produces teacher-forced per-sample NLL for arbitrary *labelled* candidate sets (`set_key`). Measured: 0.62–0.91 samples/s, peak 14.2–14.9 GB. |
| Reuse in V8? | **wrap** |
| Required change | `--selection-manifest` can already express any **2-expert** policy, which covers V7-router and V8-router arms with zero harness changes. It **cannot** express 0- or 1-expert policies, which the V8 teacher needs (BaseOnly, Reuse1, single-expert evaluation). V8 therefore adds a thin per-sample runner that sets `manager.set_default_selection(ids, gates)` itself and reuses `load_compose_model` + `process_images` + `tokenizer_image_token` unchanged. |
| Risk | Low-medium. `set_default_selection` is already per-record inside the eval loop, so per-sample variation is a supported usage — but the file-order→`image_id` coupling in the caption scorer (`llava/eval/eval_caption.py:31-47`) means any V8 caption path must join on `question_id`, never on row order. |

### 1.12 Provenance / resume

| Field | Value |
|---|---|
| Current behavior | `mtime`/`sha256` fingerprints of every frozen input are recorded before and after; `assert_temporal_boundary` rejects future experts; `validate_oracle_split` rejects anything named `test`. |
| Reuse in V8? | **as-is** |
| Required change | Extend the provenance record with: V7 pool version, base checkpoint hash, query cache version, teacher schema version, seed, generation settings. |
| Risk | Low. |

---

## 2. What V8 must NOT touch

Per PART 3.4 and PART 46, the V8 branch leaves these byte-identical:

- `compose/v7/**` — all 21 modules.
- `compose/teacher/**`, `compose/eval/**` — read-only reuse.
- `compose/adapters/**`, `compose/experts/**`, `compose/lora/**` — the composition and RMS machinery.
- Every existing run root under `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/`.

The only shared-file change permitted is additive and backward-compatible; each
such change carries a V7 regression test.

---

## 3. Consequence for the V8-A design

Two facts from this audit drive the experiment plan:

1. **Arm A of PART 34 is free.** `route_manifest()` (`compose/v7/workflow.py:114`)
   already emits exactly the manifest format `eval_task` consumes, and every
   V7 stage already has cached selections and `answers.jsonl` under
   `evaluation/`. The V7-router column needs generation only where the pool
   differs from what V7 evaluated.
2. **A large router gap already exists in V7's own data.** The prior run
   `v7_final_pool_pair_upper_val256_20260907` evaluated *all 253 expert pairs*
   by teacher-forced NLL and then generated the argmin-NLL pair per sample:

   | task | V7 actual route | NLL-oracle pair | gap (pts) |
   |---|---|---|---|
   | 0 ImageNet-R | 83.20 | 90.62 | +7.42 |
   | 1 ArxivQA | 87.50 | 91.02 | +3.52 |
   | 2 VizWiz | 37.77 | 38.61 | +0.84 |
   | 3 IconQA | 67.97 | 85.94 | **+17.97** |
   | 4 CLEVR-Math | 67.97 | 95.31 | **+27.34** |
   | 5 Flickr30k | 35.79 | 36.51 | +0.72 |

   Caveat that motivates V8 exactly as specified: this oracle selects by **NLL**,
   which PART 46 forbids as a solved criterion. It is evidence that *routing* is
   leaving performance on the table, not evidence that the historical experts
   are *correct* on those samples. V8-A re-asks the same question with a
   correctness-gated teacher.

   Task selection for V8-A therefore falls out of the data: **task 3 (IconQA)
   and task 4 (CLEVR-Math)** — the two largest gaps, both exact-match tasks with
   a clean decomposable capability signal.

---

## 4. Blockers found during the audit

| # | Blocker | Why | Impact | Proposed minimal fix |
|---|---|---|---|---|
| B1 | No per-sample official metric for ArxivQA/IconQA/CLEVR exists in the repo | All four go through corpus `Result.text` parsing | The teacher cannot decide `solved` without it | Derive it — the official metric *is* case-insensitive exact match on a single-answer annotation (`eval_deepseek_r1.py:52`), so the decomposition is exact. Gate: adapter aggregate must equal official `Result.text` on real predictions. |
| B2 | VizWiz/Flickr30k official `Average` is corpus-level (Bleu/METEOR/ROUGE/CIDEr) | Per-image scores are computed then discarded (`eval_caption.py:141-145`) | PART 8.1 `is_solved` is undefined for these tasks | Declare an explicitly-named training-time capability proxy + validation-correlation study (PART 9). Caption tasks are **not** used for V8-A task selection. |
| B3 | `--selection-manifest` cannot express 0- or 1-expert policies | Hard-requires exactly 2 distinct ids (`eval_task.py:304-305`) | BaseOnly and Reuse1 arms are unreachable through the existing CLI | V8 adds a thin per-sample runner reusing `load_compose_model`; no change to `eval_task.py`. |
| B4 | `compose/v7/commit.py::_validate_commit_directory` hard-requires a V7-schema `v7_keys.pt` | V8 pools have a split expert/key schema | V7 regression would break if V8 is made the default | V8 commit writes **both** the V8 state and a V7-schema `v7_keys.pt` derived from origin keys. |
| B5 | ~15.5 GB free per GPU vs. 14.9 GB measured peak | Other users hold 8–11 GB on every GPU | V8-A can OOM mid-run | Capacity-wait loop (the existing pattern in `oracle_generation_eval.py:116-134`), sharding across the least-loaded GPUs, resumable per-shard caches. |

All five have minimal fixes that do not alter the V8 method definition.
