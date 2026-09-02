# Changelog

## 2026-09-02

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
