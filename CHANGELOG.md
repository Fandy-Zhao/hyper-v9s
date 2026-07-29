# Changelog

## 2026-07-29
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
