# V7 two-GPU throughput and DDP correctness report

Date: 2026-09-03
Branch: `feat/0903-v7-throughput-equivalence`
GPUs: physical GPU0 and GPU1, two RTX 4090 devices

## Correctness repair

Sparse Global Top-2 uses only a subset of current experts in each micro-batch.
With DDP `find_unused_parameters=True`, the reducer at an accumulation boundary
could classify experts absent from the final micro-batch as unused and discard
their earlier accumulated gradients. V7 now keeps every backward under
`no_sync()` and performs one explicit SUM/world-size reduction at the window
boundary, before Trainer gradient clipping. A global presence reduction keeps
Adam `grad=None` semantics identical on all ranks.

Packed execution uses the labels after multimodal image-token expansion. For
batch sizes above one, answer loss is the sum of per-sample answer-token means
and key loss is the sum of per-sample selected-current-key losses. Batch size
one retains the original model loss path exactly.

## Benchmark

Task0 formal data, four optimizer steps, GC on, BF16, TF32, workers=4:

| Config | Global batch | samples/s | Runtime | Result |
|---|---:|---:|---:|---|
| b1/a32 | 64 | 1.046 | 244.76 s | distributed audit pass |
| b2/a16 | 64 | 1.352 | 189.38 s | distributed audit pass |

Speedup: 29.25%. The first optimizer window contains the same 64 unique sample
IDs in both configurations. Mean inter-step DataLoader wait was about 0.03 s,
so workers=8 was not selected. Sampled GPU memory was approximately 23 GB per
device; b2 completed without OOM.

## Formal execution decision

Selected execution config: world size 2, b2/a16, global batch 64, workers 4,
gradient checkpointing on. Learning rates, epoch count, scheduler, warmup,
weight decay, seed, Top-2 routing, RMS, pruning, validation and test protocols
remain unchanged. The global batch change from the interrupted three-GPU run's
63 to 64 was explicitly approved by the user, so the new run uses a distinct
contract and a fresh root; no old optimizer state is resumed.

Formal root:
`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_ucit_formal_2gpu_b2a16_w4_seed42_20260903`
