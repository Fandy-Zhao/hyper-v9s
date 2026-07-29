# Compose Foundation Post-review Fixes

## Branch

`feat/0729-compose-foundation`

## Summary

This pass corrected the mismatch between Compose Foundation's intended decoder-only experts and its initial repository-wide name-based injection. It also made checkpoint loading exact, prevents training samples from silently losing all answer supervision after truncation, and converts source LLaVA configuration explicitly without model-type dispatch warnings or Hyper imports.

## Files and behavior changed

- `compose/adapters/`: traverses only LLaMA decoder layers, requires the seven canonical projections, rejects duplicate injection, and reports excluded-module counts.
- `compose/experts/`: validates the exact decoder key set during save and load and records adapter tensor count, parameter count, and checkpoint byte size.
- `compose/train/data.py`: checks labels after pad/truncate, raises with batch position, sample ID, and lengths for zero-supervision samples, and records aggregate supervised-token metrics.
- `compose/model/` and the training entry: explicitly convert raw `llava`/`llama` config dictionaries into `ComposeLlavaConfig`, validate core dimensions, and print architecture/injection/expert diagnostics.
- `compose/train/trainer.py`: records the number of trainable LoRA-B tensors and finite gradients through autograd hooks, including under ZeRO-2.
- `tests/compose/`: covers exact 224-layer injection, duplicate rejection, exact 448-tensor checkpoints, corrupted checkpoint rejection, output-equivalent reload, zero-supervision truncation, and warning-free config conversion.

## Validation

- `python -m py_compile ...`: passed for all changed Python modules and tests.
- `python -m unittest discover -s tests/compose -v`: 16/16 passed in Conda environment `hyper`.
- `pytest`: not installed in `hyper`; explicitly skipped after detection.
- `bash -n` for Task1 full and smoke scripts: passed.
- Fresh import isolation: passed; importing Compose did not load `Hyper.peft`.
- `git diff --check`: passed after line-ending normalization.

## Two-step GPU smoke

- Server/GPU: `ubuntu`, one NVIDIA GeForce RTX 4090, DeepSpeed ZeRO-2.
- Output: `/data/ckpt/zhaozhuofan/compose/smoke/task1_two_step_0729_injection_fix`.
- Core config: hidden size 4096, intermediate size 11008, 32 layers, 32 attention heads, 32 key/value heads.
- Injection: 224 expected and 224 actual; vision tower 0, multimodal projector 0, LM head 0.
- Experts: one registered and trainable expert, ID 0.
- Loss: `2.3188` then `0.8274`; mean `1.5730931460857391`.
- Supervised tokens after truncation: min 3, mean 3.6667, max 5, zero-supervision 0.
- Trainable LoRA-B tensors: 224; finite-gradient LoRA-B tensors observed: 224.

## Checkpoint validation

- Decoder layers in manifest: 224.
- Adapter tensors: 448 (224 LoRA-A and 224 LoRA-B).
- Adapter parameters: 19,988,480.
- Weight file size: 40,129,746 bytes.
- Non-zero, finite LoRA-B tensors after two steps: 224/224.
- First and second strict reload: 448 expected, 448 loaded, 0 missing, 0 unexpected.
- Fixed-input logits before and after forced reload were bit-identical; maximum absolute difference was 0.0.

## Risks and next step

The validation is intentionally bounded to two steps and does not establish Task1 quality. Older 296-layer Compose checkpoints are now rejected by design. The environment still reports unrelated DeepSpeed `libaio` and PyTorch deprecation warnings. The next experimental step is a full Task1 run into a fresh output path, followed by evaluation, before any Task2 composition or learned router work.
