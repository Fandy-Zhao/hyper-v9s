# V7 Query Gate Fast Recovery Report

## Decision

The same-topology online control is bit-identical to the saved query cache on
all 23,742 ImageNet-R train samples. The prior failure is classified as
batch-topology-dependent floating-point drift, not cache corruption or
sample-ID misalignment. Existing S3, RMS and five-job pruning PASS evidence is
reused. Formal GPU0/GPU1 training is released.

## Reproducibility

- Baseline SHA: `6c06e62d542b45fd0fc91d396830e29ddc365644`
- Formal start SHA: the commit containing this report; the launcher persists
  its exact value to the formal run root as `formal_start_sha.txt`.
- Formal start timestamp: persisted by the detached launcher log at start.
- Formal cache: unchanged
  `artifacts/v7_query_cache/query_cache_manifest.json`
- Control output:
  `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_query_gate_same_topology_control_20260903`

## Original mismatch

The original online control used one contiguous batch-32 stream with a final
batch of 30. Cache production used two `index % 2` shards with 11,871 samples
each and final batches of 31+31. Exactly 32 rows were non-bit-identical:

`v7_t0_train_23680` through `v7_t0_train_23711` (declared indices
23680--23711).

This is exactly the topology-changed set (`TOPOLOGY_FAILURE_SET_MATCH=YES`).
Seventeen of those rows crossed the existing cosine threshold. The original
aggregate values were max absolute diff 4.89369035e-4, mean absolute diff
3.0294270e-8, max relative diff 1.9982227, minimum cosine 0.99999591627 and
mean cosine 0.999999998191. The original evidence remains preserved.

## Same-topology control

- Physical GPUs: 0,1
- Local mapping: physical GPU0 -> cuda:0; GPU1 -> cuda:1
- World size: 2
- Sharding: declared sample ids assigned by `index % 2`
- Batch size: 32
- Samples per shard: 11,871
- Tail batches: 31+31
- Query dtype/output: fp32, 1536-D
- Backbone, preprocessing and query implementation: unchanged from producer
- Saved and recomputed tensor content hash: `556b2e79...`

Full-split numerical result:

- max_abs_diff: 0.0
- mean_abs_diff: 0.0
- max_relative_diff: 0.0
- min_cosine_similarity: 0.9999999999999991
- mean_cosine_similarity: 1.0000000000000002
- num_over_tolerance: 0
- exact bit-equal rows: 23,742 / 23,742

Routing against one common committed pool:

- Top1 differences: 0
- Top2 set differences: 0
- Top2 order differences: 0
- Selection-count differences: none

## Boundary-route analysis

The earlier comparator exposed only the first metadata mismatch, so the phrase
"one routing-flip sample" was too narrow. Comparing each cross-topology run
with its own committed key pool finds three boundary route differences:

- `v7_t0_train_10552`: cache/same Top2 expert 3, margin_2_3
  3.5762787e-7; single-stream Top2 expert 0, margin 1.7881393e-7.
- `v7_t0_train_19227`: cache/same order [3,2], margin_2_3
  3.4451485e-5; single-stream order [2,3], margin 3.4570694e-5.
- `v7_t0_train_19991`: cache/same Top2 expert 3, margin_2_3
  4.1723251e-7; single-stream Top2 expert 2, margin 1.1920929e-7.

The cross-topology selection-count delta is `{0:+1, 2:+1, 3:-2}`. For every
sample, same-topology online equals cache. Therefore
`ROUTING_FLIP_CAUSE=NUMERICAL_BOUNDARY_UNDER_DIFFERENT_BATCH_TOPOLOGY`.

## Reused downstream evidence

- S3 cached-query training and gradient invariants: PASS
- Two-GPU DDP cache smoke, world2 x batch1 x GA32: PASS
- RMS cache equivalence: PASS
- Distributed RMS equivalence on identical checkpoint: PASS
- Five-job iterative pruning trajectory: PASS
- Evaluation cache equivalence: PASS
- Full query cache coverage: PASS (230,780 rows)
- Effective global batch: 1 x 2 x 32 = 64, exact

## Gate

- ORIGINAL_QUERY_GATE = FAIL
- FAILURE_CAUSE = BATCH_TOPOLOGY
- SAME_TOPOLOGY_QUERY_GATE = PASS
- CACHE_CORRECTNESS = PASS
- TOP2_EQUIVALENCE = PASS
- SELECTION_COUNT_EQUIVALENCE = PASS
- S3 = PASS
- RMS = PASS
- PRUNING = PASS
- GLOBAL_BATCH_64_EXACT = YES
- FORMAL_TRAINING_READY = YES

Machine-readable evidence:

- `artifacts/v7_query_gate/failing_query_sample_ids.json`
- `artifacts/v7_query_gate/routing_flip_sample_id.json`
- `artifacts/v7_query_gate/same_topology_control/audit.json`
