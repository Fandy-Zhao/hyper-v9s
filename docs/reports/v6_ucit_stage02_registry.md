# Hyper-LLaVA V6 Stage 02 — Compose Expert Registry

Status: **PASSED**

Stage commit: the atomic commit containing this report (self-reference; the authoritative hash is recorded in `git log` and the completion response). Push status: not attempted at report-generation time and recorded in the completion response.

## Architecture boundary

Stage 02 adds a metadata-only `ExpertRegistry`, scoped `ExpertActivationContext`, and the sole Compose-to-Hyper `AdapterBridge`. Dependency direction is `compose -> Hyper runtime contract`; no Hyper module imports Compose, and no `Hyper/`, original UCIT evaluator, CLIP Gaussian router, checkpoint loader, or task script was modified.

The bridge stores no tensor. It recognizes the general Hyper LoRA A/B container contract, supports empty and one active expert, separates active from trainable IDs, snapshots/restores exact runtime and `requires_grad` state, and checks DDP ID agreement when distributed state is initialized. Multiple IDs can be represented for Stage 03 preparation, but forward raises `NotImplementedError` instead of silently selecting one expert.

## Metadata and registry

`ExpertMetadata` is versioned and JSON-safe. Canonical fields cover adapter identity, rank/alpha, status, creation/checkpoint provenance, active/trainable flags, support/reuse/contribution counters, parent ID, version, and extension data. Allowed Stage-02 statuses are registered, frozen, trainable, and archived. Unknown loaded fields are preserved under `extra`. Legacy ExpertPool fields and the historical `training` spelling remain readable.

`ExpertRegistry` preserves registration and selection order, rejects duplicate/unregistered/archived selection, keeps active and trainable sets independent, prevents unregistering checkpoint-bound experts, and round-trips through state dictionaries and atomic non-overwriting JSON files. Registry checkpoints add source commit/model/config hash/timestamp/run ID and contain no model weights.

## Validation

- Python compile and targeted Stage-02 tests: passed.
- Full `tests/compose` regression: 74/74 passed.
- Real Hyper tiny-layer compatibility: max absolute output difference 0.0; selected expert gradients present; all other experts frozen with no gradient.
- Context nesting and exception restoration: passed.
- JSON/state_dict/checkpoint round trips and SHA-256: passed.
- V6-off import and forward behavior: unchanged.
- Hyper compatibility hooks: none; the experiment wrapper monkeypatches only the training module's local `get_peft_model` reference and leaves the frozen entry unchanged.

## GPU and GB24

GPU 0–3 were occupied by another user's openpi processes. Physical GPU 4–7 were selected, with process-local mapping 0→4, 1→5, 2→6, 3→7. Training used world size 4, per-device batch 2, gradient accumulation 3, effective global batch 24. Evaluation used physical GPU 4. No other process was stopped. No OOM occurred.

## Smoke

ImageNet-R smoke completed 30 optimizer steps with train loss 0.7668777044. The frozen evaluator produced all 128 predictions and 53.91% accuracy; this is a closure check, not a formal metric.

Two launcher failures and one evaluator naming-dispatch failure occurred before useful work; every log/config was retained. Retry2 used `torchrun` to preserve explicit CUDA visibility while Trainer still used DeepSpeed zero2. The evaluator retry used a verified symlink containing the established `llava` basename convention; evaluator code was not changed.

## Mini2

Accuracy matrix (percent):

| after task | ImageNet-R | ArxivQA |
|---|---:|---:|
| ImageNet-R | 47.66 | — |
| ArxivQA | 50.00 | 78.91 |

Metrics: MAA 56.0575, MFN 64.4550, MFT 63.2850, BWT 2.34. Against Stage 01 GB24, the first three matrix positions are identical except ArxivQA is 78.91 versus 80.47 (2/128 fewer correct; -1.56pp). Metric differences are MAA -0.39, MFN -0.78, MFT -0.78, BWT 0.00. This is reasonable mini-sample fluctuation and does not indicate baseline regression.

## Known limitations and Stage 03 prerequisites

Stage 02 intentionally does not execute multi-expert delta composition and implements no RMS composition, teacher, router, residual buffer, candidate/lifecycle, shadow update, or V6 full trainer. Stage 03 may build single/dual expert execution and RMS calibration on the registry/bridge contract after this commit; no later stage was started here.
