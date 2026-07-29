# ADR-0729: Compose Decoder-only Injection Boundary

- Status: Accepted
- Date: 2026-07-29
- Branch: `feat/0729-compose-foundation`

## Context

The initial Compose injector matched terminal module names across `model.named_modules()`. LLaMA decoder projections, the CLIP vision tower, and the multimodal projector reuse names such as `q_proj`, `k_proj`, and `down_proj`, so name-only matching injected 296 layers even though the intended expert definition was the decoder's seven projections per layer. This also allowed adapter-only checkpoints to persist a broader and unstable model boundary.

## Decision

1. Compose Foundation injects only `get_model().layers[*]`.
2. Every decoder layer must contain exactly `self_attn.{q_proj,k_proj,v_proj,o_proj}` and `mlp.{gate_proj,up_proj,down_proj}` as ordinary linear layers before injection.
3. The complete seven-name set is mandatory. Partial target lists, duplicate injection, missing projections, non-linear targets, and any Compose layer outside this boundary are errors.
4. Vision towers, multimodal projectors, embeddings, and `lm_head` are excluded even when their leaves share target names.
5. Expert checkpoint keys must correspond exactly to the validated decoder layer set and each expert contributes only `lora_A.weight` and `lora_B.weight` per injected layer.

For the 32-layer LLaVA-1.5 7B foundation model, this yields 224 injected projections and 448 tensors per expert.

## Consequences

### Positive

- The trainable expert boundary now matches the architecture description and is stable across vision-tower implementations.
- Checkpoint size and tensor count are deterministic and auditable.
- Accidental reinjection and silent partial injection fail before the model is modified.

### Costs and compatibility

- The Foundation API no longer supports arbitrary linear-module target lists.
- Earlier 296-layer Compose smoke checkpoints are incompatible with strict loading and remain diagnostic artifacts only.
- A future architecture with different decoder projection names requires a new explicit boundary decision and tests.
