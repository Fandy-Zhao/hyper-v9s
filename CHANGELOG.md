# Changelog

## 2026-07-30
- Made Compose gate normalization explicit with `none`, `l1`, and `l2` modes; the default now preserves supplied gates and default selections use unit gates.
- Changed mixed-sample Compose execution to run each expert only on rows with a positive gate and scatter weighted deltas back with autograd-safe `index_add_`.
- Added normalization, validation, mixed top-1/top-2 execution, inactive-expert, gradient, rank-2 input, and bf16 coverage, bringing the Compose suite to 30 passing unit tests.
- Validated grouped bf16 forward/backward execution against a per-sample reference on an RTX 4090 with zero observed output and input-gradient difference.

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
