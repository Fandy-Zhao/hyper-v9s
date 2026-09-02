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
