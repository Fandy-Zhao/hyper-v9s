# Module Status

## Overview
Status of each module in the HiDe-LLaVA project as of 2026-07-26.

## Modules

| Module | Path | Purpose | Status | Tests / Checks | Notes |
| --- | --- | --- | --- | --- | --- |
| LLaVA Core | `llava/model/` | MLLM architecture (vision/text towers, LLaMA, projector) | Stable | Import test | Forked from LLaVA v1.5, modified for dual-tower + routing |
| LLaVA Train | `llava/train/` | Training loop, trainer, MOE entry point | Active | Dry-run only | `train_MOE.py` is main entry; depends on `compute_routing_weights.py` in root |
| LLaVA Eval | `llava/eval/` | Per-dataset evaluation (8 datasets) | Stable | Full eval needs GPU | Covers VQAv2, GQA, VizWiz, TextVQA, OCRVQA, ScienceQA, ImageNet, Grounding |
| LLaVA Serve | `llava/serve/` | Gradio web UI, controller, worker | Stable | Manual | Not needed for research; inherited from LLaVA |
| Hyper PEFT | `Hyper/peft/` | Custom PEFT framework | Active | Import test | Modified from HuggingFace PEFT; adds HyperMOELora |
| HyperMOELora | `Hyper/peft/tuners/clitmoelora.py` | CLIP-guided multi-expert LoRA with task routing | Active | Integration test only | Core innovation; Gaussian stats + expert management |
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

## Risks
- `llava/model/llava_arch copy.py` is a stale copy — archived to deprecated
- `scripts/Hyper/Train_UCIT/*.bak_*` files (4 backup files) — archived to deprecated
- `gaussian.py` in root — non-runnable reference snippets, not a module
- `compute_routing_weights.py` in root — runtime dependency of `train_MOE.py`, cannot move
- `nohup.out` (23MB) in root — should be `.gitignore`'d
- `flash_attn-*.whl` (~1GB) in root — should be moved to external storage
