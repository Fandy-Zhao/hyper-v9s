# AGENTS.md

## Project Goal
HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model (ACL 2025 Main). This project implements a task-specific expansion and task-general fusion framework for continual instruction tuning of MLLMs, using Centered Kernel Alignment (CKA) similarity to guide layer-wise adaptation, and Gaussian-based modality routing for image/text fusion. Built on LLaVA v1.5 and CoIN.

## Hard Rules
1. Do not modify unrelated files.
2. Do not create temporary files in the repository root.
3. Put temporary analysis, logs, and experiment records under `experiments/runs/MMDD_short-name/`.
4. Every meaningful code change must update at least one status document: `PROJECT_STATE.md`, `CHANGELOG.md`, `docs/module_status.md`, or a relevant module README.
5. Do not directly delete deprecated files. Move them to `docs/deprecated/MMDD-short-name/` and add a `README.md` explaining why they were deprecated, what replaces them, and whether they can later be deleted.
6. Before edits, report `git status --short`, current branch, files read, and the plan.
7. After edits, report `git status --short`, `git diff --stat`, and relevant tests or smoke checks.
8. Final reports must include changed files, reasons, tests, risks, and next steps.

## Workflow
Issue -> Branch -> Plan -> Diff -> Commit -> Report

## Branch Policy
- `feat/MMDD-short-name` — new feature or experiment
- `fix/MMDD-short-name` — bug fix
- `docs/MMDD-short-name` — documentation only
- `chore/MMDD-short-name` — maintenance, cleanup, refactor
- `exp/MMDD-short-name` — exploratory experiment
- `refactor/MMDD-short-name` — code restructuring

Do not develop directly on `main` or `master`. Current active branch: `zzf`.

## Commit Policy
- `feat(scope): short description`
- `fix(scope): short description`
- `docs(scope): short description`
- `chore(scope): short description`
- `refactor(scope): short description`
- `test(scope): short description`
- `exp(scope): short description`

Scope examples: `llava`, `hyper`, `scripts`, `routing`, `eval`, `train`, `data`.
Avoid vague messages such as `update`, `fix bug`, `change files`, `final`, or `temp`.

## Documentation Policy
- `PROJECT_STATE.md`: current project status, active branch plan, known risks.
- `ROADMAP.md`: planned milestones and backlog.
- `CHANGELOG.md`: date-ordered user-visible and engineering changes.
- `docs/architecture.md`: system architecture and major dependencies.
- `docs/module_status.md`: module ownership, status, and validation notes.
- `docs/decisions/ADR-MMDD-short-name.md`: architectural or technical decisions.
- `docs/reports/MMDD_short-name.md`: task reports, audits, or integration summaries.
- `experiments/runs/MMDD_short-name/notes.md`: experiment notes, command logs, and results.

## Deprecated File Policy
Move deprecated files to `docs/deprecated/MMDD-short-name/`. Include a `README.md` with original location, deprecation reason, replacement, and deletion guidance. Do not directly delete files without archiving.

## Test Policy
This is a deep learning research project. Standard checks:
- **Python syntax / import smoke test**: `python -c "from llava.model import LlavaLlamaForCausalLM; from Hyper.peft import HyperMOELoraConfig; print('imports OK')"`
- **Config validation**: verify `config.json` is valid JSON and contains required fields (`mm_vision_tower`, `mm_text_tower`, `model_type`)
- **Training dry-run**: run `train_MOE.py` with `--num_train_steps 2` on a small data slice to validate the training loop
- **Shell script validation**: `bash -n` on all `.sh` scripts under `scripts/`
- Full training/eval runs require GPU cluster (multi-GPU DeepSpeed) and UCIT dataset — not runnable in lightweight CI

## Project-Specific Modules
| Module | Path | Purpose |
| --- | --- | --- |
| LLaVA Core | `llava/model/` | MLLM architecture: vision tower, text tower, projector, LLaMA backbone |
| LLaVA Train | `llava/train/` | Training loop, trainer, memory-efficient variants, MOE training entry |
| LLaVA Eval | `llava/eval/` | Per-dataset evaluation scripts for UCIT benchmark |
| LLaVA Serve | `llava/serve/` | Model serving (gradio, controller, worker) |
| Hyper PEFT | `Hyper/peft/` | Custom PEFT framework: HyperMOELora, multi-expert LoRA with task routing |
| HyperMOELora | `Hyper/peft/tuners/clitmoelora.py` | Core innovation: CLIP-guided MOE LoRA with task-specific expert expansion |
| Instance Router | `llava/model/routing/instance_router.py` | Modality-aware instance-level routing for image/text fusion |
| Scripts | `scripts/Hyper/` | Training and evaluation shell scripts for CoIN and UCIT benchmarks |
| Analysis Tools | `tools/` | CKA similarity, Gaussian analysis, data cleaning utilities |
| Sample Instructions | `sample_instructions/` | Example instruction JSON files for each UCIT task |
