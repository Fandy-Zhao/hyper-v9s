# Changelog

## 2026-09-11 — V8: alias-key gradient fix, the missing training loop, and pipeline tests

- **Fixed a defect that made alias-key learning inert.** `alias_key_loss` built
  its ranking hinge from `float(...)` values, so `L_rank` was a constant with no
  gradient; the only differentiable term left, `L_pos`, has its exact stationary
  point at the centroid `create_alias_keys` initialises keys with. A freshly
  created key therefore had a gradient of ~0 (measured `5.16e-08`, float noise)
  and could not be trained. The hinge now stays attached to the current key,
  while the hardest competitor remains **detached** so a historical key receives
  no update. Loss values are unchanged (verified: `ranking` 0.405291 before and
  after); gradients are now `3.3e-02` / `4.7e-02` on the same probe.
- **New `compose/v8/trainer.py` — the V8 current-task training loop.** Until now
  the V8 primitives were all tested in isolation but nothing assembled them, so
  V8-B had no entry point. `V8TaskTrainer` enforces the freeze and captures the
  frozen ledger *before* building the optimizer, runs mixed (never filtered)
  batches through `residual_answer_loss`, trains candidate LoRA and current-task
  alias keys in two parameter groups, audits the gradient footprint each step,
  exposes `leakage_probe`, and `finalize()` prunes, re-verifies the ledger,
  commits and checkpoints.
- Tests: `tests/compose` now reports **606 passed** (V8 58, V7 regression 548).
  Added a chained pipeline integration test (teacher verdicts → canonical cache
  → lazy alias keys → ledger → one gated step → pruning → commit → `UNCHANGED`),
  a hinge-arithmetic guard test on exact basis-vector geometry, and two trainer
  tests on the real `ComposeLinear` path. Every historical tensor is asserted
  bit-identical after a training epoch and every historical key keeps
  `grad is None`.
- Report §9.1 records the defect, its cause, the fix and the evidence, so the
  finding is attributed to a bug rather than to "key geometry is insufficient".

## 2026-09-11 — V8 Answer-Supervised Multi-Key: core, TEST 01–30, and the V8-A runner

- New package `compose/v8/` (config, query, pool, routing, selection, teacher,
  key_learning, gating, pruning, commit, checkpoint, cache, audit, inference,
  metric_adapter) plus the experiment runner `compose/experiments/v8_task_run.py`.
  V7 is untouched: no file under `compose/v7/`, `compose/adapters/`, or
  `compose/eval/` was modified, and no existing checkpoint was written to.
- Tests: `tests/compose/test_v8_answer_supervised_multikey.py` (TEST 01–30) and
  `tests/compose/test_v8_generation_harness.py`. 92 pass together with the V7
  regression file `test_v7_global_coevolution.py`.
- V8-A runs the Answer-Supervised Expert Teacher over the committed V7 pool
  (`…/v7_gpu01_cached_query_formal_20260903/task5/committed`, 23 experts, 23
  origin keys) with **no training and no alias keys**, so it tests Goal A
  (correctness-decided expert reuse) alone; with one key per expert the
  multi-key router reduces exactly to V7's, so Goal B is out of scope for it
  and the report says so.
- `compose/v8/generate.py` supplies the per-sample 0/1/2-expert routing that
  V7's evaluator cannot express — `compose/eval/eval_task.py` hard-requires two
  *distinct* experts on `--selection-manifest`, which is recorded as a BLOCKER
  with this module as its minimal fix. Batch size stays 1 and the prompt is a
  byte-identical copy of the V7 evaluator's (asserted by test) so V8 numbers are
  comparable to the V7 baseline they are measured against.
- Two silent-failure bugs found by the runner's own guard rails, both fixed:
  `RouteGenerationCache` defined `__len__` without `__bool__`, so a fresh cache
  was falsy and every write was skipped — a run reported `generated=31` and left
  no `generation_cache.jsonl`; and the seed spot check built its expert list as
  `route.split("_")[1:]`, which turns `"e10_13"` into `[13]`, so every seeded
  *pair* was silently re-measured as a single expert and a correct run aborted.
  `experts_from_route` is now the checked inverse of `route_key`.
- Smoke (task 4 CLEVR-Math, first 4 validation samples, GPU 3): states
  `{'BaseOnly': 1, 'Reuse1': 3, 'Reuse2': 0, 'Residual': 0}` — minimal
  cardinality, no sample needed a pair; V8 policy 100.0 vs V7 actual-route
  67.97; seed verification MATCH, max |diff| 0.0 across 8 live recomputations of
  the frozen V7 pair diagnostic.

## 2026-09-03 — DDP smoke lifecycle closed; DISTRIBUTED_RMS_EQUIVALENCE PASS (0903 spec §22/§24)

- DDP cache smoke (`v7_gpu01_ddp_cache_smoke_task0_20260903`, world 2 ×
  batch 1 × GA 32 = global 64, GPU0+1, started 09:34) closed 10:47 with a
  full committed/ (9 pruning jobs; rank/coverage/checksum audits green) —
  TWO_GPU_DDP_CACHE_TRAIN_SMOKE PASS.
- DISTRIBUTED_RMS_EQUIVALENCE certified PASS on the real pair: world-2 RMS
  recompute of the single-GPU cache root's *identical* post-S3 model
  (`v7_gpu01_rms_recompute_task0_20260903`, checkpoint_hash `6380bb4d…`
  equal on both sides) vs the recorded single-process RMS through the new
  `--gate-mode distributed-rms` comparator (`compare_audit_v3_
  distributed_rms.json`): rms_calibration 900 leaves + rms_summary 10
  leaves bit-identical, rms_statistics 13,441 leaves max_rel 5.9e-8;
  execution mode/world_size + calibration_sha256 recorded informational.
- Batch-32 live twin rerun launched 10:58 on GPU0 (PID 3971155, root
  `v7_gpu01_smoke_task0_live_twin_batch32_20260903`) for the §8a/§26 gate
  re-issue on the batch-32 pair.

## 2026-09-03 — DISTRIBUTED_RMS recompute gate mode + live-encoder batch passthrough (0903 spec §24/§8a)

- `compose/eval/v7_twin_run_compare.py` gains `--gate-mode distributed-rms`:
  a single-stage RMS-only gate certifying a world-2 RMS recompute of the
  *identical* post-S3 checkpoint against the recorded single-process RMS
  run (spec §24 leg 2).  Execution-context leaves
  (`execution.mode`/`world_size`) and the self-derived `calibration_sha256`
  are exempted informational on both sides; `checkpoint_hash` and every
  value leaf stay hard-compared — a recompute over a different checkpoint
  FAILs.  Design basis (recorded in PROJECT_STATE): two independently
  smoked roots consume disjoint S3 sample windows per world size
  (length-grouped sampler), so their weights legitimately differ and no
  full-root value comparison can certify the gate; the same-checkpoint
  recompute isolates the world-size contrast.  Real-artifact precheck:
  calibration/statistics files differ only in provenance.checkpoint_hash
  (same schema, no world-dependent ordered arrays).  6 new CPU tests.
- `compose/experiments/v7_task_run.py` gains `--query-features-batch-size`
  (default None -> legacy built-in 16, byte-identical): explicit live-
  encoder CLIP batch override required by the §8a batch32 twin remediation
  (cache-production / bounded gates encode at batch 32).  All three live
  encoder sites now assemble through the pure, unit-tested
  `_live_query_features_command`; cache/formal paths (encoder_calls=0)
  untouched.  6 new CPU tests.

## 2026-09-03 — V7 comparator semantic fixes + twin-run divergence localization (0903 spec §26/§28)

- `compose/eval/v7_twin_run_compare.py` semantic fixes (semantic
  comparator v2; tests updated to 21, all PASS):
  - S1 gate compares per-*sample_id* rows (id-set equality is hard; file
    order is an emission artifact — cache emits declared order, live
    emits encode order — recorded as informational
    `id_sequence_identical`).
  - json structure walk relocates machine-path leaves
    (output_dir/annotation_file/prediction_file) root-relatively and
    counts them as `relocated_path_leaves`; RMS compare passes the per-root
    paths.
  - COMMIT gate is semantic: committed/ file set hard, binary files byte
    sha256 hard, json scalars numeric with `COMMIT_FLOAT_ATOL` 2e-4, and
    v7_keys.pt compared per expert via torch.load with `KEY_ATOL` 2e-4 —
    both bounds are the *measured* upstream-noise envelope (answer_nll_full
    1.7e-4, redundancy contribution 7e-5, key diffs 7.6e-5..1.2e-4 from
    the batch-shape effect), not free parameters; sha256 mismatches stay
    informational.
- Root-caused the two remaining cache-vs-online FAIL classes to a single
  control-setup defect (report §8a, five-step evidence chain): the live
  twin's S1 ran the legacy encoder at its default batch **16** while cache
  production / the bounded gate encode at batch **32**; fp16 CLIP conv
  kernels are cuDNN-chosen per batch shape → deterministic visual-half
  row diffs ≤1.87e-3 → S2 center/keys share a 7.6e-5 offset → pruning val
  rows 170/195 (+ reroute 107) boundary Top-2 flips.  S2 determinism is
  proven (DDP-vs-single cache `candidate_keys.pt` 4/4 bit-equal on the
  same cache rows); the V7 method and the formal path (encoder_calls: 0)
  are untouched.  Remediation: re-run the live twin with `--batch-size 32`
  after the DDP smoke frees GPU0/1.
- RMS_CACHE_EQUIVALENCE PASS on the real twin pair (3 RMS files
  numeric-equal, calibration/statistics byte-identical); commit keys
  numeric sub-gate PASS (≤1.16e-4); compose_experts reroute flip correctly
  still FAILs (route-id structure diff).  Audit v2:
  `v7_gpu01_smoke_compare_task0_20260903/compare_audit_v2.json`.

## 2026-09-03 — V7 Phase B twin-run comparator + smoke affinity correction (0903 spec §28)

- Added `compose/eval/v7_twin_run_compare.py` (+ 16 CPU tests): read-only
  gate evidence machinery that diffs two complete task0 run roots stage by
  stage and emits the named spec verdicts — S1_QUERY_ROWS_EQUIVALENCE
  (per-sample_id cosine bound 1-1e-6), S3_TRAIN_STEPS_EQUIVALENCE (float
  fields 1e-4, id/route fields exact, timing recorded not compared),
  RMS_CACHE_EQUIVALENCE / DISTRIBUTED_RMS_EQUIVALENCE (same comparator,
  `--gate-mode distributed`; RMS value bound 1e-4 rel + 1e-6 abs,
  checkpoint-sha equality recorded separately), PRUNING_TRAJECTORY_
  EQUIVALENCE (identical pruning job sets + per-job selections/nll/
  official_metric), COMMIT_STATE_EQUIVALENCE (committed/ file set +
  v7_keys.pt byte sha256).  Any FAIL exits non-zero.
- GPU-affinity correction (my launch error, not an orchestrator bug):
  adaptive `--gpus N` means *physical GPU id N* (CLI > env precedence),
  so the live twin smoke had been running on physical GPU1 beside the
  cache smoke instead of GPU0.  Killed the misplaced process tree,
  relaunched with `--gpus 0` on GPU0 (PID 3389457); gpu_plan.json
  available_gpu_ids=[0] verified.
- Cache-mode fixed smoke externally terminated ~08:52 (its exit-watcher
  fired; last artifact job_4.done 08:23, no s5 marker; second external
  termination of that run after 07:48 — neither session's doing).  Its
  root resumed on GPU1 from an `ea816a3` git worktree (stored
  run-contract git_sha binding requires the original HEAD; `--config`
  must point at the main-repo absolute path and cwd must be the worktree
  — the first attempt failed exactly on that contract mismatch).

## 2026-09-03 — V7 Phase A independent cache re-verification (0903 spec §2-3)

- Added `compose/eval/v7_cache_precheck.py`: read-only, CPU-only gate tool
  that recomputes every count, sample-id/tensor hash and contract binding
  from the *live* declared split files, the on-disk cache binaries and the
  current runtime module constants / live CLIP backbone provenance, then
  emits the spec §3 QUERY CACHE PRECHECK block + JSON evidence.
- Phase A re-audit at HEAD `ea816a3`: 18/18 splits PASS (230,780 queries:
  declared == saved == unique, missing/unknown = 0, 1536-D fp32 all-finite,
  L2-norm tolerance, sample-id set hash and query tensor hash rehash-match,
  live source sha256 + content hash + backbone/impl content binding green);
  6/6 task centers recomputed from the full cached train rows bit-identical
  (max_abs_diff 0.0).  FULL_SAMPLE_QUERY_COVERAGE = YES ·
  QUERY_CONTRACT_MATCH = YES · QUERY_CACHE_READY = YES.  Producer `c93f51e`
  vs runtime `ea816a3` git drift recorded (content-bound decision record).
- Cache-vs-live gate re-run at the *final* HEAD `ea816a3` on physical GPU0
  (task0/task1.train, 128 real samples each, evidence run root
  `v7_gpu01_cache_live_gate_20260903_rerun_head`): QUERY_NUMERICAL_
  EQUIVALENCE PASS (exact_bit_equal, max_abs_diff 0.0, cosine_min
  0.99999976) and TOP2_ROUTING_EQUIVALENCE PASS (rate 1.0, score diff 0.0).
- Full `tests/compose` regression at HEAD `ea816a3`: 500 passed / 2
  env-limited failures unchanged (`java`-less caption scorers) — the
  EXISTING_V7_REGRESSION baseline is reconfirmed at the formal HEAD.
- Phase A evidence: `artifacts/v7_query_cache/reverification_20260903_080136.json`
  (gitignored; mirrored in the cache run root for reports).

## 2026-09-03 (evening) — V7 downstream cache adaptation (0903 spec §4-28)

- Added the S1 cache adapter `compose/v7/cache_to_s1.py`: emits the legacy
  byte-schema `features/<split>.json` payload straight from the binary cache
  (streamed, `encoder_calls: 0`, `query_origin` recorded, sample-id sequence
  enforced fail-closed, content binding on backbone/impl hashes, producer vs
  runtime git recorded).  Real task0 emission dry-run: 45.9 s for 23,742
  train + 256 val rows (vs multi-minute dual-GPU live CLIP).
- `v7_task_run.py`: S1 now switches to cache emission under
  `--query-cache-manifest`; S3/S4/S5 blocks assert the payloads' cache origin
  (`_assert_s1_payload_origin`) and refuse live re-encoding; live paths are
  byte-identical when the flag is absent.
- Added final-evaluation cache reuse: `compose/v7/cached_selections.py`
  precomputes committed-pool Global Top-2 selection manifests from cache test
  rows (same router, same device as the legacy live eval, `encoder_calls: 0`);
  per-task S6 and the 21-cell evaluator (`v7_formal_ucit_eval.py`) consume
  them via `eval_task --selection-manifest` with no CLIP query encoder; the
  exactly-3-GPU restriction relaxes to ≥2 distinct GPUs in cache mode only.
- RMS (activation RMS, no query consumption) and pruning (consumes the
  cache-derived payload rows) inherit reuse without method changes.
- 14 new CPU tests (`test_cache_to_s1.py` 7, `test_cached_selections.py` 7);
  full `tests/compose` 500 passed (the only 2 failures remain the documented
  pre-existing `java`-less caption-scorer parity tests).
- Commits: `af716f0` (adapter + evaluation reuse), `1083eee` (Phase A
  report + producer tooling).  GPU equivalence gates + smokes are pending an
  idle GPU0/1 window (blocked by other users' jobs at report time).

## 2026-09-03

- Added a standalone Fixed-Query full precompute pipeline on GPU0+GPU1
  (`compose/eval/precompute_v7_queries.py` orchestrator/worker/gate +
  `compose/v7/query_cache.py` cache library): parity sharding (`index%world`),
  per-rank tmp partials, 10-point merge audit, tmp->fsync->atomic-rename official
  writes, contract-hash-bound resume/invalidation, and full-train task centers.
- Cached all declared V7 queries (230,780 = 211,244 train + 1,536 val + 18,000
  test across 6 tasks, 1536-D fp32, detached, exact V7 math) plus 6 task centers
  under `v7_fixed_query_cache_gpu01_20260903/`; main-flow files untouched.
- Passed a bounded 128-sample single-GPU vs dual-GPU gate bit-exactly
  (cosine_min 0.99999976 >= 1-1e-6) and a Top-2 route-consistency smoke against
  the dormant formal 2-GPU run's S2 candidate pool (agreement 1.0, 0 score diff).
- Fixed a throughput bottleneck (serial PIL resize -> decode-thread pre-resize
  with identical transformers math, proved bit-identical): ~17 -> ~56-65
  samples/s per worker (3.5x).
- Added 28 CPU tests for shard/merge/atomic/contract/audit logic
  (`tests/compose/test_query_cache_audit.py`), all passing with the 48-test
  query regression; report in
  `docs/reports/V7_FIXED_QUERY_CACHE_GPU01_REPORT.md`; machine-readable
  manifest mirrored to `artifacts/v7_query_cache/query_cache_manifest.json`.

## 2026-09-03

- Added the formal V7 three-rank DDP path on GPU0--2. Only S3 training is
  distributed; six tasks remain strictly sequential with global batch 63
  (`1 x 21 x 3`) and unchanged learning rates.
- Added sparse-Key DDP graph anchoring, rank-local logs, cross-rank
  Key/current-LoRA and optimizer audits, global sampler coverage/padding
  accounting, rank0-only V7 checkpoint writes, and per-rank resume counters.
- Added the resumable six-task launcher and a three-worker task-free V7
  evaluator for exactly 21 lower-triangle UCIT cells.
- Passed 39 focused tests and a real 7B Task0 three-GPU/two-step gate.

## 2026-09-02

- Hardened the final V7 pre-training gate: split isolation now distinguishes
  harmless reused source IDs from real image+question/normalized-record
  leakage, and every resumable stage is bound to Git/config/data/annotation/
  previous-checkpoint hashes. Formal runs now persist and print the
  single-process effective global batch and final artifact provenance, while
  rejecting unsupported implicit DDP execution.

- Completed the V7 formal implementation repair: uncapped full-data formal
  mode with observed-sample coverage, answer-only NLL mask parity, unified RMS
  and preprocessing contracts, iterative remove-and-reroute pruning, strict
  Top-2 commit, split provenance, official UCIT validation path and a separate
  resumable six-task launcher.
- Added explicit fixed-query CLIP backbone/hash provenance, schema-locked
  `1/sqrt(2)` pair scaling, accumulation-aware gradient hooks, optimizer and
  scheduler resume restoration, and transactional atomic commit directories.
- Expanded final regression coverage to `396 passed + 8 subtests`; final-HEAD
  GPU smoke was resource-blocked by external allocations. Formal launch remains
  fail-closed until audited validation files are installed.

- Fixed the V7 dynamic Global Top-2 training boundary to pad each two-expert
  route to the unified four-slot `ComposeSelection` contract without changing
  the active experts or gates.
- Added a regression test reproducing the real 7B GPU2 first-step failure.
- Made historical LoRA checksums dtype/shape-aware and byte-exact for BF16,
  fixing Task1 frozen-history audit initialization on the real 7B checkpoint.

## 2026-09-01

- Added Hyper-LLaVA V7 full-data global Key–Expert co-evolution as explicit
  `v7_global_coevolution` training/inference mode.
- Added fixed parameter-free 1536-D multimodal Query, four rank-8 current
  Candidates, per-sample historical+current Global Top-2, selected-current
  Key/LoRA training and frozen-history audits.
- Added validation remove-and-reroute pruning, RMS-bound retained commit,
  atomic resume state, committed-only inference and machine-readable route,
  loss, gradient, usage, pair and cross-task diagnostics.
- Added 17 acceptance tests, a 30+30 step two-task CPU smoke and bounded real
  ImageNet-R fixed-query preparation smoke.
## 2026-08-15
- Added a resumable, four-GPU no-router oracle evaluator for the frozen V6.2 Formal UCIT seed42 run: exhaustive empty/single/pair fixed selections, original UCIT scoring, target-answer sample oracle, exact stage-snapshot reuse proofs, cross-task and pair-synergy analysis, frozen-RMS audit, formal-artifact fingerprints, and a fail-closed completeness report.

## 2026-08-05
- Completed the V6 UCIT engineering closure batch (Stage E0-E12, 13 commits, HEAD 41e70bd): unified empty/single/pair ComposeSelection, expert lifecycle registry with two-phase commit transactions, 16-stage task state machine, dual-mode Query-Key Router, answer-supervised teacher with Top-M retrieval, answer-teacher-driven residual buffers, 1/2-slot candidate pools, validation + transactional commits (0/1/2), Router calibration, RMS statistics, independent-load snapshots, acceptance tests (339 passed + 14 subtests) and a real two-task UCIT dry run (ImageNet-R -> ArxivQA, seed 42): Task 1 committed expert 10 (validation gain +0.136, 500-sample accuracy 27.4%); Task 2 all-empty teachers (no residual) with commit 0 by design (50.6% backbone-only). 18/18 acceptance criteria PASS.
- Produced the handoff package (`artifacts/v6_ucit_handoff/`, 14 files) with locked config, exact/resume commands, schemas and known issues; `ready_for_six_task_run = true`. Batch stops here per the task book: no third task, no six-task run, no push.

## 2026-08-03
- Added arithmetic-mean per-layer RMS composition, validation-only C3 scalar selection, layer contribution/cosine audits, and a resumable one-process-per-GPU scheduler with an explicitly authorized 4--7 fallback.
- Completed the 12-run P1 formal matrix over checkpoint seeds 42/43/44 (analysis seeds 0/1/2), using physical GPUs 4--7 for 3.434 recorded GPU-hours; all runs completed without OOM.
- Applied the frozen unseen-composition gate: both independent and residual B+C fail C2/C3 synergy, bootstrap, and best-single accuracy requirements, yielding `FAIL_COMPOSITION`.
- Added two-slot candidate-pool and answer-free multi-label Query-Key router implementations. Gate-limited seed-0 diagnostics yield `FAIL_SLOT_SPECIALIZATION` and `ROUTER_QUERY_INSUFFICIENT`; the full continual benchmark remains prohibited.
- Expanded the Compose suite to 159 passing tests plus 8 subtests and added complete configs, logs, metrics, gate decisions, reports, and a reproduction script under `outputs/compose_p1_p3_20260803T090000Z/`.

## 2026-07-30
- Made Compose gate normalization explicit with `none`, `l1`, and `l2` modes; the default now preserves supplied gates and default selections use unit gates.
- Changed mixed-sample Compose execution to run each expert only on rows with a positive gate and scatter weighted deltas back with autograd-safe `index_add_`.
- Added normalization, validation, mixed top-1/top-2 execution, inactive-expert, gradient, rank-2 input, and bf16 coverage, bringing the Compose suite to 30 passing unit tests.
- Validated grouped bf16 forward/backward execution against a per-sample reference on an RTX 4090 with zero observed output and input-gradient difference.
- Added a strict standalone Compose/PEFT evaluation path, supervision and reload audits, and matched full UCIT Task1 training; Compose reaches 90.2000% versus PEFT 90.1667% with identical adapter parameter counts.
- Trained and strictly reloaded a two-expert functional-proxy pool while proving all pre-existing Expert 0 tensors and fixed-input logits remain exactly unchanged.
- Added fp32 per-sample teacher-forced NLL, stable empty/single/pair candidate enumeration, deterministic Oracle caching, and matched selection scoring/evaluation controls; the Compose suite now has 42 passing tests.
- Completed the 6,000-sample Oracle and exactly parameter-matched rank-16 control. Negative mean/median synergy and stronger rank-16 NLL/accuracy trigger stop condition C, so Set Router development is stopped for this pool.

## 2026-07-29
- Restricted Compose adapter injection to the seven LLaMA decoder projections per layer (224 for the 32-layer foundation model), with duplicate and excluded-module validation.
- Made Compose expert checkpoints strict: one expert now saves exactly 448 decoder tensors plus tensor, parameter, and file-size metrics; reload rejects missing, unexpected, or boundary-external keys.
- Added post-truncation zero-supervision protection and min/mean/max supervised-token diagnostics to Compose data collation.
- Replaced implicit `llava` config dispatch with explicit LLaVA-to-Compose conversion and core-dimension validation.
- Expanded Compose coverage to 16 unit tests and completed the post-review two-step GPU smoke and strict output-equivalent checkpoint reload.
- Added an independent `compose/` package with clean LLaVA model classes and multimodal sequence preparation that does not import Hyper PEFT or routing state.
- Added independent LoRA experts, fixed sample-level top-1/top-2 composition, injection/management APIs, ExpertPool metadata, and adapter-only checkpoint save/load.
- Added a standalone Compose training entry and UCIT Task1 full/smoke DeepSpeed scripts.
- Added 10 Compose unit tests and completed a two-step single-GPU ZeRO-2 Task1 smoke run with finite losses (`2.3188`, `0.8084`).
- Changed the top-level `llava` model export to a compatible lazy import so utility imports do not eagerly initialize Hyper-LLaVA.
- Added ADR and implementation report for Compose Foundation.

## 2026-07-26
- Initialized project governance skeleton (AGENTS.md, PROJECT_STATE.md, ROADMAP.md, CHANGELOG.md)
- Created directory structure: `docs/`, `docs/decisions/`, `docs/deprecated/`, `docs/reports/`, `experiments/runs/`, `tools/`
- Created `docs/architecture.md` and `docs/module_status.md`
- Moved root-level analysis scripts to `tools/`
- Archived deprecated/backup files to `docs/deprecated/0726-cleanup/`
- Cleaned root `__pycache__/`

## 2025-07 — 2026-06 (prior history, reconstructed from git log)
- `0252db1` — Hyper-LLaVA original backup (2026-07-26)
- `ecfc76c` — C1 test finish
- `34c51e1` — C1 Test
- `b961d1a` — C1-sample/sample-rule added
- `4dd0793` — Initial file upload
- `2c09394` — README.md update
- Earlier commits — LLaVA base, Hyper PEFT, training scripts, model architecture
- ACL 2025 paper: HiDe-LLaVA accepted (arXiv:2503.12941)
- New work: FCIT (Federated Continual Instruction Tuning, ICCV 2025)
- New survey: Comprehensive Survey on Continual Learning in Generative Models (2025.06)
