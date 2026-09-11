# Module Status

## Overview
Status of each module in the HiDe-LLaVA project as of 2026-08-03.

## Modules

| Module | Path | Purpose | Status | Tests / Checks | Notes |
| --- | --- | --- | --- | --- | --- |
| LLaVA Core | `llava/model/` | MLLM architecture (vision/text towers, LLaMA, projector) | Stable | Import test | Forked from LLaVA v1.5, modified for dual-tower + routing |
| LLaVA Train | `llava/train/` | Training loop, trainer, MOE entry point | Active | Dry-run only | `train_MOE.py` is main entry; depends on `compute_routing_weights.py` in root |
| LLaVA Eval | `llava/eval/` | Per-dataset evaluation (8 datasets) | Stable | Full eval needs GPU | Covers VQAv2, GQA, VizWiz, TextVQA, OCRVQA, ScienceQA, ImageNet, Grounding |
| LLaVA Serve | `llava/serve/` | Gradio web UI, controller, worker | Stable | Manual | Not needed for research; inherited from LLaVA |
| Hyper PEFT | `Hyper/peft/` | Custom PEFT framework | Active | Import test | Modified from HuggingFace PEFT; adds HyperMOELora |
| HyperMOELora | `Hyper/peft/tuners/clitmoelora.py` | CLIP-guided multi-expert LoRA with task routing | Active | Integration test only | Core innovation; Gaussian stats + expert management |
| Compose Foundation | `compose/` | Independent fixed-selection LoRA expert composition for LLaVA | Engineering validated; formal unseen composition failed | 159 tests + 12-run P1 matrix on GPUs 4--7 | Arithmetic-mean RMS and validation-only scalars execute correctly, but C2/C3 fail the frozen unseen B+C synergy/accuracy gates |
| Compose V7 | `compose/v7/`, `compose/experiments/v7_task_run.py` | Full-data Global Key--Expert co-evolution | Three-GPU formal run active | 39 focused tests + real 7B 3-rank two-step gate | S3-only DDP on GPU0--2; rank-audited sparse execution; six tasks remain sequential |
| Compose V7 Fixed-Query Cache | `compose/v7/query_cache.py`, `compose/v7/cache_to_s1.py`, `compose/v7/cached_selections.py`, `compose/eval/precompute_v7_queries.py`, `compose/eval/v7_cache_live_gate.py`, `compose/eval/v7_cache_precheck.py`, `compose/eval/v7_twin_run_compare.py` | Full precompute + downstream reuse of V7 fixed queries / task centers | Code complete (adaptation); Phase A independent re-audit PASS (18/18 splits + 6/6 centers bit-exact); cache-vs-live gates PASS at final HEAD; Phase B smokes in flight (cache smoke resumed on GPU1 after external termination, live twin on GPU0); twin-run comparator added | 28 cache-audit + 7 cache-to-s1 + 7 cached-selections + 16 twin-run-compare CPU tests; `tests/compose` 500/502 (2 = pre-existing `java`-less env limits) | S1/S3/S4/S5/S6 + 21-cell eval consume cache rows (`encoder_calls=0`), sample_id sequence fail-closed, content binding; `v7_twin_run_compare.py` diffs two complete task0 roots stage-by-stage (S1 rows / S3 steps / RMS / pruning trajectory / commit) with named verdicts and atomic audit JSON |
| Compose Oracle | `compose/oracle/` | Per-sample NLL, fixed empty/single/pair audit, cache, and matched controls | Validated | Deterministic repeated smoke + 6,000-sample formal audit | Stop condition C triggered after rank-16 capacity control |
| Compose Candidate Pool / Set Router | `compose/expansion/`, `compose/router/` | Two-slot candidate experts and answer-free multi-label Query-Key routing | Diagnostic only | Unit tests + gate-limited seed-0 P2/P3 smokes | P2 `FAIL_SLOT_SPECIALIZATION`; P3 `ROUTER_QUERY_INSUFFICIENT`; no full continual benchmark authorized |
| Instance Router | `llava/model/routing/instance_router.py` | Modality-aware per-instance fusion | Active | Integration test only | MLP-based router with Gaussian prior initialization |
| Scripts - Train | `scripts/Hyper/Train_*/` | Shell scripts for sequential task training | Stable | `bash -n` syntax | Multiple training orders: UCIT, UCIT_AIRFCV, UCIT_IFRCAV, UCIT_LlaVANext, CoIN |
| Scripts - Eval | `scripts/Hyper/Eval_*/` | Shell scripts for evaluation | Stable | `bash -n` syntax | Per-task eval scripts + aggregated Eval_all.sh |
| Scripts - Metrics | `scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py` | Aggregate continual learning metrics | Stable | Python compile | Computes average/task-by-task performance |
| Analysis Tools | `tools/` | CKA, Gaussian analysis, data cleaning | Reference | Manual | Standalone scripts; see `tools/README.md` |
| Sample Instructions | `sample_instructions/` | Example JSON instruction files | Reference | N/A | One sample per UCIT task |
| Config | `config.json` | Model configuration for LLaVA | Stable | JSON valid | Must be placed in LLaVA checkpoint directory |
| Requirements | `requirements.txt` | Full pip dependency list | Stable | N/A | Mix of conda/pip; some paths are machine-specific |

## Submodule Details

### llava/model/ (LLaVA Core)
- `llava_arch.py` — LlavaMetaModel with dual-tower support
- `language_model/llava_llama.py` — LLaMA-based causal LM
- `language_model/mpt/` — MPT model variant (legacy, not actively used)
- `multimodal_encoder/` — CLIP vision/text towers
- `multimodal_projector/` — MLP projector
- `routing/instance_router.py` — Instance-level modality router
- `llava_arch copy.py` — **DEPRECATED** copy, moved to `docs/deprecated/`

### Hyper/peft/ (Custom PEFT)
- `tuners/clitmoelora.py` — HyperMOELora (core contribution)
- `tuners/lora.py` — Base LoRA with 8bit/4bit support
- `tuners/` — Also includes AdaLoRA, IA3, prefix/prompt tuning (inherited)
- `utils/` — Config, save/load, hub utilities

### scripts/Hyper/ (Training & Evaluation)
- `Train_CoIN/` — 8-task CoIN benchmark training
- `Train_UCIT/` — 6-task UCIT benchmark training
- `Train_UCIT_AIRFCV/` — Alternative task order A
- `Train_UCIT_IFRCAV/` — Alternative task order B
- `Train_UCIT_LlaVANext/` — LLaVA-NeXT variant (experimental)
- `Eval_CoIN/` — 8-task CoIN evaluation
- `Eval_UCIT/` — 6-task UCIT evaluation
- `Eval_UCIT_AIRFCV/` — Alternative order A evaluation
- `Eval_UCIT_IFRCAV/` — Alternative order B evaluation

### compose/ (Compose Foundation)
- `model/` — clean Compose LLaVA config/model and vision-only multimodal preparation
- `adapters/` — independent LoRA experts, explicit gate normalization, active-row grouped execution, context-local sample selection, and manager
- `experts/` — ExpertPool metadata and adapter-only checkpoint manifests/weights
- `train/` — clean v1/UCIT preprocessing, Compose Trainer save hook, and standalone entry
- `eval/` - strict Compose/PEFT loading, fixed-selection generation, metrics, and run summaries
- `oracle/` - per-sample NLL, stable candidate sets, cache, deterministic evaluator, and capacity comparison
- `scripts/Compose/` - full/smoke training, evaluation, Oracle, and matched baseline launchers
- `expansion/` and `router/` - two-slot candidate pool and answer-free multi-label Query-Key router, currently diagnostic-only
- `tests/compose/` - 159 tests plus 8 subtests covering foundation, Oracle, arithmetic-RMS composition, scheduling, candidate slots, and multi-label routing
- `v8/` - answer-supervised multi-key layer: frozen historical LoRA/keys, per-task alias keys, correctness-decided reuse (BaseOnly/Reuse1/Reuse2/Residual), optimizer whitelist and gradient gating. Additive: it reads the V7 pool and never writes to it. See `docs/reports/V7_TO_V8_IMPLEMENTATION_MAPPING.md`
- `v8/` + `experiments/v8_task_run.py` - V8-A: the Answer-Supervised Expert Teacher over the committed V7 pool, no training, per-sample 0/1/2-expert generation with an append-only answer cache and an official-metric gate

## Risks
- `llava/model/llava_arch copy.py` is a stale copy — archived to deprecated
- `scripts/Hyper/Train_UCIT/*.bak_*` files (4 backup files) — archived to deprecated
- `gaussian.py` in root — non-runnable reference snippets, not a module
- `compute_routing_weights.py` in root — runtime dependency of `train_MOE.py`, cannot move
- `nohup.out` (23MB) in root — should be `.gitignore`'d
- `flash_attn-*.whl` (~1GB) in root — should be moved to external storage
# 2026-08-11 Compose seed42 rank study

- Status: in progress on branch `exp/0811-rank-study-seed42`.
- Scope: controlled rank, task0 bootstrap, oracle, frozen-RMS, clustering,
  and specialization experiments under
  `experiments/runs/compose_ucit_rank_study_seed42/`.
- Training boundary: formal seed42 residual IDs, Query features, cluster
  assignments, historical snapshots, generation settings, and evaluators
  remain frozen.
- Runner change: S6 now forwards resolved `lora.rank` and `lora.alpha` to
  `train_compose` and prints the required preflight contract before launch.
- GPU policy update: after the user restriction issued on 2026-08-11, all
  remaining A/B/C/D/F GPU work is serialized or batched exclusively on
  physical GPUs 4-7. The prior 0-3 scheduler and transition watchers were
  stopped; the interrupted task1-rank16 S6 has no completion marker and will
  restart through the idempotent four-GPU path on 4-7.
- RMS recovery: task1-rank32 exhausted 24 GiB during S9 hook recomputation.
  RMS now uses micro-batch 1 and releases unused CUDA cache before its exact
  fp64 all-reduce; the validation boundary, moments, reduction, and kappa
  protocol are unchanged. The repaired S9 completed with 896 finite entries
  and roughly 16.0 GiB observed peak before the pipeline resumed at S11.
- Safety: no seed43/44 experiment, no formal seed42 overwrite, no commit.

# 2026-09-11 V8 alias-key gradient fix + training loop

- Status: implemented, tested, committed on `exp/v8-answer-supervised-multikey`.
- `compose/v8/key_learning.py`: `alias_key_loss`'s ranking term was a constant
  (`float(...)` in the hinge), so `L_key` had no usable gradient at the centroid
  initialisation. The hinge now stays attached to the current key; the hardest
  competitor is still detached so frozen historical keys receive no update.
  Loss values unchanged.
- `compose/v8/trainer.py` (new): `V8TaskTrainer` assembles the parts that were
  previously only tested in isolation — freeze enforcement and ledger capture
  before the optimizer exists, mixed (never filtered) batches with per-sample
  gating, candidate LoRA + current-task alias keys in two parameter groups,
  per-step gradient-footprint audit, `leakage_probe`, and
  `finalize()` = prune → re-verify ledger → commit → checkpoint.
  Scope boundary documented in the module docstring: candidate *redundancy*
  pruning is planned and reported (`candidate_redundancy`) but not applied
  automatically, because the redundancy threshold is unvalidated on this pool.
- Tests: `tests/compose` 606 passed (V8 58, V7 regression 548). New: chained
  pipeline integration test, hinge guard test on exact basis-vector geometry,
  two trainer tests on the real `ComposeLinear` path.
- `compose/v8/teacher.py`: the Reuse1 key-target rule was not total. STEP C
  scores the pool-wide candidate list on every unsolved sample, so
  `tested_singles` is a superset of the sample's own `recall` (10 vs 8 on the
  formal Task 4 run); building the three-valued rule over `recall` and
  defaulting the rest to NEGATIVE labelled an out-of-recall *solver* negative —
  PART 9's forbidden `E5 = negative` — and would have stripped a
  `POSITIVE` from the selected expert had it come from outside the recall. The
  rule now runs over `tested_singles ∪ recall`. Measured pre-fix on the smoke
  run: `v7_t4_val_10` → expert 3 `negative`, should be `ignore`.
- `compose/experiments/v8a_label_audit.py` (new): recomputes the rule from each
  record's stored evidence (`recall`, `tested_singles`, `single_values`,
  `solved_threshold`, `selected_experts`) and reports mismatches per run, so the
  pre-fix run 1 and the post-fix runs 2–4 are compared from artifacts rather
  than by assertion. No artifact is rewritten.
- V8-B is still **not run**: cost is ~12–14 GPU-hours per task (V7's own training
  is 5 h 58 min for 621 steps; the teacher would be 6–8 h single-GPU at the
  declared 2,000-sample budget). Recommendation in report §19.5 is a one-task,
  500-sample pilot rather than the full loop.

# 2026-09-11 (later) V8 alias-key origin-task fix, parity path fix, second campaign

- Status: implemented, tested, committed on `exp/v8-answer-supervised-multikey`.
- `compose/v8/key_learning.py`: `create_alias_keys` created an alias for **every**
  expert with positive support, including experts whose `origin_task` *is* the
  current task. `MultiKeyExpertPool.validate()` rejects exactly that key
  ("alias key ... sits on the origin task; use the origin key"), so the function
  could build a pool that cannot validate. Unreachable until the all-experts
  scope existed, because every earlier caller passed a strictly historical pool.
  An origin-task expert is now skipped with that reason recorded. Regression test
  `test_22b_current_task_expert_gets_no_alias_key`; the old behaviour is verified
  to raise by constructing the bad pool directly.
- `compose/experiments/v8_task0_parity.py`: the answer-parity branch composed
  V7's answer path as `<diagnostic_root>/task0/generation_accuracy/task0/actual/…`
  — one `task0` too many, since `generation_accuracy/task0/…` hangs off the run
  *root* (the same layout read correctly by `v8_task_run.py:800` and
  `v8a_cases.py:97`). `_jsonl()` returns `[]` for a missing file, so the empty
  dict surfaced 30 lines later as `KeyError: 'v7_t0_val_0'`. Fixed, and the
  missing-file and zero-overlap cases now raise with the path in the message.
  **Re-run end to end on 2026-09-11: MATCH** — 256/256 identical answers,
  `metric_equal: true`, byte-identical official result text on both branches
  (`result_text_sha256 d1ab9369…`, `83.2`), over the same 256-sample route
  multiset (`{0,1}: 2, {0,3}: 135, {1,2}: 109, {2,3}: 10`). No open items remain
  in the code verdict.
- Campaign: all four V8-A teacher runs complete (Tasks 3 and 4, all-experts and
  history-only scopes), 256 samples each, 7.02 GPU-hours total, seed verification
  `MATCH` on all four. Post-campaign audits (label / recall / scope-gap / cases /
  alias counterfactual) written to `experiments/runs/0911_v8a_formal/`.
- `compose/experiments/v8_full_pool_recall.py`: the Residual audit read
  `recall.json`'s `visible_expert_ids` as the run's scope. That field holds the
  full committed pool (`v8_task_run.py:585`); the scope is in
  `excluded_expert_ids`, which every other consumer reads. The Task-4 history-only
  audit consequently tested Task 4's own experts 16–19 and the future-task experts
  20/21/23, and 21 of its 22 "retrieval failure" verdicts were samples solved only
  by an excluded expert. Scope resolution (`resolve_scope`) and plan construction
  (`audit_plan`) are now separate and covered by
  `test_22c_full_pool_audit_honours_the_history_only_scope`; output carries
  `schema_version: 2`. The v1 artefact is kept as evidence and the re-run writes
  `_v2`. **Corrected Task-4 result: 1 retrieval / 29 capability of 30 audited**
  (240 generations against the 8 untested visible experts), Wilson 95 % upper
  bound 16.7 % of the 141 Residual. **Task 3's audit is vacuous**: all 170
  Residual samples had already been scored against all 12 visible experts, so
  the plan is empty for every sample; the script now detects this before loading
  the model (`vacuous: true`, `generated: 0`, zero GPU seconds) instead of
  emitting a vacuous "0 retrieval failures".
- Tests: `tests/` 609 passed (V8 61, V7 regression 548). Report §1 and §25 added;
  §12.1, §14–§18, §20–§21, §23 rewritten with two-task data.
- Headline correction recorded in report §14/§25: the all-experts V8-A numbers
  (94.14 / 87.11) are roughly half self-reuse; genuine cross-task reuse is
  73/256 (Task 4) and 40/256 (Task 3).

# 2026-09-11 (final) V8 method-semantics convergence: Capability Discovery Oracle

- Status: implemented, tested, **not committed at the time of writing**; branch
  `exp/v8-answer-supervised-multikey`. No long run started (PART 13).
- Scope: audit + minimal correction against three finalised decisions — (1) the
  training teacher is a Capability Discovery Oracle and may not let the old keys'
  recall decide who receives answer supervision; (2) a Residual sample's old
  historical context must also train the current-task alias key; (3) inference
  stays a fixed sparse Top-2 with no weighting, no dynamic K, no dense MoE.
- **`compose/v8/teacher.py` — search universe was recall-bounded (the audit's
  main finding).** STEP C looped over `candidates = union of per-sample recall`,
  so an expert no sample recalled was never scored: capability discovery
  depended on key quality, and a badly-ranked capable expert produced no solver
  verdict, hence no alias key, hence stayed badly ranked — a self-confirming
  failure that hid its own deficit. On the formal Task-4 run that universe was 10
  of 16 visible historical experts, so **every Residual count in the V8-A
  campaign is a lower bound**. `visible_experts` is now a *required* argument (no
  caller can fall back to the old scope by omission) and STEP C iterates it
  directly; `recall` survives only as `router_top_m` / `router_top_m_diagnostic`
  / `router_rank`, the quantity the keys are judged by. `assert_full_history_coverage`
  makes `tested == visible` a hard invariant for every non-BaseOnly sample (a
  BaseOnly sample must have tested nothing), and `TeacherResult.coverage_report()`
  writes the auditable trace into `teacher_result.json`. Pair search is unchanged
  and still `bounded` by default (top-4 singles → ≤6 pairs) with `exhaustive` as
  explicit configuration; it is not claimed to be exhaustive.
- **`compose/v8/config.py` + `key_learning.py` — Residual context role was
  inverted.** Residual labelled *every* tested expert `NEGATIVE`, including the
  context expert, and `create_alias_keys` derived support from
  `positives_by_sample()` alone. So a context-only expert got **no alias key at
  all**, and any key it had was actively **repelled from** exactly those queries —
  the inverse of DECISION-2. Four-valued roles now: `TARGET_POSITIVE` (legacy
  string `"positive"` kept so old artefacts still parse), `TARGET_CONTEXT_POSITIVE`
  (`"context_positive"`), `TARGET_NEGATIVE`, `TARGET_IGNORE`; on a Residual every
  non-context tested expert is now IGNORE, never NEGATIVE, and
  `assert_no_ignore_is_negative` rejects context∩negative on one sample.
  `AliasSupport(k,t) = SolverPositiveQueries ∪ ContextPositiveQueries`, created
  lazily at `alias_support_threshold`, with `support_source ∈ {solver_only,
  context_only, mixed, none}` and `solver/context_support_count` per key.
  `L_key` splits the positive term: `lambda_solver_positive = 1.0`,
  `lambda_context_positive = 1.0`, `lambda_rank = 0.1`, `ranking_margin = 0.2`;
  `lambda_pos` survives as a read-only property + config-dict rename shim.
- **`compose/v8/selection.py`**: `STATE_CARDINALITY[Residual] (0,1,2) → (0,1)`,
  `MAX_RESIDUAL_ACTIVE_EXPERTS = 2`. A Residual row composes context + candidate;
  two historical contexts would demand a three-expert cardinality inference
  cannot reproduce.
- **`compose/v8/trainer.py`**: `build_key_targets` is now called per batch with
  `sample_ids=batch.sample_ids` instead of once per epoch over all records, so
  the state/gradient table is literally true of the batch that ran; a
  zero-target batch skips `backward()` via a `loss.requires_grad` guard (a
  constant-zero loss has no graph — this was a real crash the new tests caught).
  `TrainReport` gains `key_batches` / `key_role_samples`.
- **`compose/v8/cache.py`**: `CACHE_VERSION = 2`, `SUPPORTED_CACHE_VERSIONS =
  (1, 2)`. v2 adds `historical_experts_visible` / `historical_experts_tested` to
  every record; a v1 file still loads with those defaulted to empty — which is
  the honest reading, since that record *was* produced by a recall-bounded
  search. An unknown version raises and the file is left untouched; nothing is
  upgraded in place.
- **`compose/experiments/v8_task_run.py`**: `visible_experts` passed explicitly
  (`v8_task_run.py:572`); `recall.json` gains `schema_version`,
  `teacher_visible_expert_ids` (because the legacy `visible_expert_ids` has
  always held the whole active pool and the scope lives in
  `excluded_expert_ids` — the trap that §16.1's audit fell into) and
  `router_top_m_diagnostic: true`; `train/val` separation stated. New
  `metric_report()` (`v8_task_run.py:1184`) emits **two** metrics:
  `teacher_oracle_metric` (route chosen with the answer in hand, cardinality
  varies per sample — this is what 94.14 / 87.11 are) and
  `actual_top2_inference_metric` (the deployable fixed Top-2, `--top2-inference-eval`
  opt-in, and `null` with a reason when not measured). The deployable route
  (`top2_inference()`, `:825`) is supervised-free end to end.
- Tests: `tests/` **621 passed** (V8 multi-key 57, generation harness 16, **V7
  regression 548 — unchanged**, which is the evidence that no V7 path was
  touched). Twelve new oracle tests A–L, incl. D = the full
  `teacher_result → key_targets → create_alias_keys → alias_key_loss → tensor
  gradient` flow for the context expert alone in a solver-free batch, testing the
  *direction* of the step and not only that a gradient exists; I/J pin the
  fixed-Top-2 chain and its supervision-free purity; K pins the oracle/deployable
  metric distinction.
- Artefacts: `experiments/runs/0911_v8a_formal/*` were **not** rewritten. Their
  numbers keep their v1 meaning and are labelled PRE-ORACLE-SEMANTICS in report
  §14 / §17 / §19.1. Report: new **§26**; §9 table corrected (it was the report
  that was wrong, not the code); §11/§19.1/§1 updated.
- V8-B still **not run**. Revised plan (report §26.10): full suite → oracle smoke
  (`--limit 32`, ~15–25 min) → Phase-0 re-analysis of cached generations →
  Task-1 V8-B pilot (~12–14 GPU-h) → six-task loop (~35–40 GPU-h). No step past
  the suite was executed.
