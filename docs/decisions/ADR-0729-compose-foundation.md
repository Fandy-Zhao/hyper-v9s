# ADR-0729: Compose Foundation Architecture

- Status: Accepted
- Date: 2026-07-29
- Branch: `feat/0729-compose-foundation`

## Context

The repository's current LLaVA path is inseparable from Hyper-LLaVA at import and runtime: `llava_arch.py` imports Hyper PEFT, multimodal preparation performs Gaussian/Poincare routing, and `LlavaLlamaForCausalLM` initializes task IDs, fixed expert counts, anchors, statistics, and an instance router. The ordinary data entry also contains a live `breakpoint()`. Reusing these classes would make Compose another mode of Hyper-LLaVA instead of an independent method.

## Decision

1. Add a standalone `compose/` package. Compose model classes inherit directly from Transformers LLaMA classes and do not inherit the repository's Hyper-enabled LLaVA classes.
2. Keep multimodal responsibilities in `compose/model/multimodal_arch.py`: CLIP vision loading, projector loading, image encoding, image-token insertion, mask/position/label rebuilding, and image propagation during generation.
3. Use independent `LoRAExpert` modules inside `ComposeLinear`. A `ComposeSelection` carries per-sample `[batch, top_k]` expert IDs and normalized gates. The foundation deliberately permits only top-1 or top-2.
4. Propagate explicit sample selections through a context-local runtime. Fixed training/inference selections can also be installed as layer defaults; there is no task-ID binding or learned router.
5. Manage metadata separately through `ExpertPool`, and save only Compose expert parameters plus a versioned JSON manifest. Hugging Face/DeepSpeed still owns optimizer, scheduler, and Trainer state.
6. Copy only the minimal vision tower/projector adapter behavior needed by Compose. Change `llava/__init__.py` to a compatible lazy export so importing `llava.constants`, conversations, or training utilities does not eagerly load Hyper-LLaVA.
7. Copy the UCIT v1 preprocessing path into Compose and remove the debug breakpoint rather than importing `llava/train/train.py`.

## Consequences

### Positive

- Compose imports and checkpoints are independent from `Hyper/peft`, Gaussian statistics, Poincare routing, and `InstanceModalityRouter`.
- Top-1 and top-2 composition have explicit, testable batch semantics.
- Existing Hyper-LLaVA model and training implementations remain available as the baseline.
- Adapter checkpoints are small enough for continual-learning experiments and can be loaded without serializing the 7B base model.

### Costs and Risks

- The clean multimodal adapter duplicates a small part of LLaVA and must be kept aligned if the base vision interface changes.
- The initial checkpoint emits a Transformers warning because the source config says `model_type=llava`; weights are structurally compatible and the two-step run validates loading, but a future model conversion tool could emit a native Compose config.
- Only fixed selection is implemented. Automatic routing and the full expert lifecycle require later ADRs and evaluation plans.
