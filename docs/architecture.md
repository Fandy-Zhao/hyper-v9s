# Architecture: HiDe-LLaVA

## Overview

HiDe-LLaVA extends LLaVA v1.5 with a hierarchical decoupling strategy for continual instruction tuning. The architecture has three main tiers:

```
┌─────────────────────────────────────────────────┐
│                 HiDe-LLaVA                       │
├─────────────────────────────────────────────────┤
│  LLaVA v1.5 Base (llava/)                        │
│  ├── Vision Tower (CLIP-ViT-L/14@336)            │
│  ├── Text Tower (CLIP-ViT-L/14 text encoder)     │
│  ├── Vision Projector (MLP 2x GELU)              │
│  ├── LLM Backbone (LLaMA-7B)                     │
│  └── Modality Router (instance-level fusion)     │
├─────────────────────────────────────────────────┤
│  Hyper PEFT Framework (Hyper/peft/)               │
│  ├── HyperMOELora (task-specific expert LoRA)    │
│  ├── Gaussian Statistics (image/text per expert) │
│  ├── Routing Weight Computation                  │
│  └── Task Embedding / Expert Selection           │
├─────────────────────────────────────────────────┤
│  Training & Eval Pipeline (scripts/Hyper/)        │
│  ├── Sequential Task Training (CoIN/UCIT)        │
│  ├── CKA Similarity Analysis                     │
│  └── Multi-benchmark Evaluation                  │
└─────────────────────────────────────────────────┘
```

## Key Components

### 1. LLaVA Core (`llava/model/`)

- **`llava_arch.py`**: `LlavaMetaModel` — abstract base combining vision tower, text tower, and multimodal projector. Handles dual-tower feature extraction and modality alignment.
- **`language_model/llava_llama.py`**: `LlavaLlamaForCausalLM` — the full causal LM wrapping LLaMA with multimodal inputs.
- **`multimodal_encoder/`**: CLIP-based vision and text encoders.
- **`multimodal_projector/`**: MLP projector mapping vision features to LLM embedding space.
- **`routing/instance_router.py`**: `InstanceModalityRouter` — lightweight MLP that predicts per-instance image/text fusion weights, initialized from Gaussian prior statistics.

### 2. Hyper PEFT (`Hyper/peft/`)

- **`tuners/clitmoelora.py`**: `HyperMOELoraModel` + `HyperMOELoraConfig` — the core innovation. Multi-expert LoRA where each expert corresponds to a previously learned task. CLIP-guided routing selects which experts to activate. Gaussian statistics (mean, covariance) are maintained per expert for both image and text modalities.
- **`tuners/lora.py`**: Standard LoRA implementation (shared base).
- **`peft_model.py`**: PEFT model wrappers with save/load utilities.

### 3. Training Pipeline (`llava/train/`)

- **`train_MOE.py`**: Main training entry for MOE-based continual learning. Orchestrates dataset loading, model setup with HyperMOELora, and sequential task training. Calls `compute_routing_weights.py` at each stage to compute adaptive image/text fusion weights.
- **`llava_trainer.py`**: Custom trainer extending HuggingFace Trainer with LLaVA-specific loss and logging.

### 4. Analysis Tools (`tools/`)

- **CKA Similarity**: `test_CKA_sim.py` — computes Centered Kernel Alignment between model layers to guide which layers to expand vs. fuse.
- **Gaussian Analysis**: `calbc.py`, `calpd.py`, `calwd.py`, `clcalbc.py` — analyze expert overlap via Bhattacharyya coefficient, Poincaré ball distance, and Wasserstein distance.
- **Gaussian Verification**: `verify_gaussian.py` — validates the diagonal Gaussian assumption via PCA and marginal distribution plots.

## Data Flow

```
Image + Instruction
       │
       ▼
┌─────────────┐    ┌─────────────┐
│ Vision Tower │    │ Text Tower  │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│   Projector │    │  Tokenizer  │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └──────┬───────────┘
              ▼
    ┌──────────────────┐
    │ Instance Router  │ ◄── Gaussian prior (per-expert image/text stats)
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  HyperMOELora    │ ◄── Task embedding → expert selection
    │  (LLaMA Layers)  │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │   LM Head        │
    │   (Text Output)  │
    └──────────────────┘
```

## Key Design Decisions

1. **Task-specific expansion + task-general fusion**: New tasks get dedicated LoRA experts (expansion), while shared knowledge is maintained through a frozen base + selective routing (fusion).
2. **CKA-guided layer selection**: Not all layers need task-specific adaptation — CKA similarity analysis identifies which layers benefit most from expansion.
3. **Gaussian modality routing**: Instead of fixed image/text weights, each expert maintains Gaussian statistics per modality, enabling adaptive fusion per instance.
4. **Sequential training**: Tasks are trained one at a time (continual learning setting), with previous experts frozen to prevent catastrophic forgetting.

## Dependencies

- **LLaVA v1.5**: Base MLLM architecture (Apache 2.0)
- **CoIN**: Continual instruction tuning framework
- **DeepSpeed**: Distributed training (ZeRO-2/3)
- **CLIP-ViT-L/14@336**: Vision and text encoders (OpenAI)
- **LLaMA-7B**: Language model backbone
- **PEFT**: Parameter-efficient fine-tuning (customized fork in `Hyper/peft/`)
