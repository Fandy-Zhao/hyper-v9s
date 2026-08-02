# Hyper-LLaVA V6 Stage 06 — Keyed 0/1/2 Expert Set Router

Status: **PASSED_WITH_NEGATIVE_RESULT**. The engineering acceptance criteria passed, while the formal routed UCIT result is below the frozen Hyper baseline. Git: this report is contained by the atomic Stage 06 commit; source parent `907806c938869d45eae379ef6a4b1cf33c3bf000`.

## Frozen design and provenance

- Formal mode: `continual_anchor`; composition: `direct_sum`; supervision: historical-only Oracle-Direct; no test Oracle.
- Candidate retrieval uses the Stage 05 128-D normalized Query and registered Expert Keys with `M=min(8, visible_expert_count)`. A cardinality head predicts 0/1/2, learned single scores rank members, and a symmetric pair MLP scores unordered pairs.
- Pair thresholds were preregistered as `0.50/0.65/0.80`. All three produced the same validation decisions; `0.80` was frozen once using validation regret/precision only.
- Stage 06 config-file SHA256: `0d15fa157442aa9c7758b94aed2fb183061983df56072a41c47ee7fc0564a292`; frozen training-config hash embedded in the checkpoint: `cd4952a089c6ae3df531bf81a1edfb766073236f3a8efe02653018cfb82fe8b2`; Stage 05 checkpoint config hash: `cc4ea261d7964f2b2bba0159c4ed76f730f66d3eb564805d56adcfcdb922397b`.
- Dataset manifest hash: `2982f01bc8f8d29a3a6ab648286d1ee399f76319c3031649add99db4dd5af3cf`; Oracle cache hash: `779af3046f12b6293d94a329ee837216598a94dbb5f51e0f6fb5874c6f328d5f`.
- Query/Key checkpoint SHA256: `8b8e3837eef4ac4cf52b20652730d5cf6acfe801f48bec5cbfe8be6836f02439`; Router checkpoint SHA256: `b42c0b3d8074a944dd3a8ea80290c1a2ecb2767ce7e12e0d21d25f23c629b287`.
- GPU: preferred 0–3 were occupied by another user; formal work used 4–7, never more than four GPUs. Environment: `{"cuda":"11.8","peft":"0.4.0","python":"3.10.20","pytorch":"2.3.1+cu118","transformers":"4.33.3"}`.

## Technical validation

- Full Compose suite: `134 passed, 8 subtests passed`; the eight Stage 06 test files separately pass (`9 passed`), covering cardinality, pair symmetry, set inference/losses, deterministic bounded anchor replay, temporal masks, answer isolation, checkpoint round-trip, and DDP state consistency.
- Router test manifests explicitly record `answer_features_used=false`, `oracle_used=false`, and `task_id_lookup_used=false`; temporal violations are `0`.
- Empty, Single, and Pair execution paths are unit-tested. Direct pair composition remains covered by the frozen Compose runtime tests. V6-off and Stage 04 Oracle scorer regressions pass in the full Compose suite.
- Large checkpoints, feature caches, route manifests, logits, and generation outputs remain under `/data/ckpt/zhaozhuofan/v6_ucit_staged/` and are not committed.

## Continual-anchor validation

On 192 frozen validation rows, candidate recall upper bound is `1.000000`, exact set accuracy `0.447917`, and cardinality accuracy `0.583333`.

| Cardinality | Precision | Recall | F1 |
|---|---:|---:|---:|
| Empty | 1.000000 | 0.347826 | 0.516129 |
| Single | 0.500000 | 1.000000 | 0.666667 |
| Pair | 0.000000 | 0.000000 | 0.000000 |

- Member micro/macro F1: `0.478571 / 0.312882`; pair both-members accuracy: `0`; pair false-positive rate: `0`.
- Predicted 0/1/2 rates: `0.166667 / 0.833333 / 0`; average active experts: `0.833333`.
- Predicted-minus-Oracle NLL regret: `0.206364`; predicted-minus-best-single NLL: `0.151812`; selected-set score regret: `0.208447`; accuracy regret: `0.020833`; pair-only regret: `0.390767`; worst-10% regret: `1.576802`.
- Router-only latency: `0.001476 s/sample`, throughput `677.67 samples/s`, peak allocated memory `12,601,344 bytes`.

OracleSet is an offline train/validation upper bound only. PredictedSingleOnly and PredictedSet are identical for the formal test run because no Pair was predicted. Hyper original route is the immutable Stage 01 baseline.

## Mini2, Mini3, and full UCIT

- Mini2 matrix: `[[91.90], [91.60, 57.77]]`; MAA/MFN/MFT/BWT = `80.4233 / 74.8350 / 74.6850 / -0.3000`.
- Mini3 matrix: `[[91.90], [91.60, 57.77], [90.57, 85.40, 54.46]]`; MAA/MFN/MFT/BWT = `78.6167 / 68.0433 / 76.8100 / 13.1500`.
- Full 6x6 lower triangle:

```text
91.90
91.60  57.77
90.57  85.40  54.46
87.80  86.90  54.68  69.97
87.80  85.77  53.74  63.87  38.07
87.13  93.43  59.35  64.87  47.07  56.29
```

Formal MAA/MFN/MFT/BWT = `75.3510 / 68.0233 / 61.4100 / +7.9360`. Frozen Hyper baseline = `83.1269 / 73.2367 / 78.4817 / -6.2940`; deltas are `-7.7759 / -5.2133 / -17.0717 / +14.2300`. The positive BWT must not be read as overall superiority: it is driven by weak diagonals followed by higher later-task scores.

Across all 63,000 test decisions, route counts 0/1/2 are `6 / 62,994 / 0`, average active experts is `0.999905`; the six Empty decisions are all in stage-6 Flickr30k and every other decision is Single. Generation peak memory is `16,309,958,656 bytes`; `44,308` immutable Stage 01 answers were reused exactly; generation time including reuse is `0.133966 s/sample`; route-only latency is `0.000903 s/sample`.

## Failures, retries, and limitations

- The first ArxivQA task-2 generation attempt was stopped after identifying that already reusable frozen answers were still entering model setup; pending-only execution was implemented and the same cell then completed. No scored result from the failed attempt was retained.
- VizWiz task-3 caption scoring initially failed because Java was absent from `PATH`; the existing immutable merged answers were rescored after adding the conda Java path, without regeneration.
- A formal validation rerun serialized the unchanged Router state under a new checkpoint hash. Route manifests were regenerated against that hash and decisions were compared field-for-field with the original manifests.
- The first regeneration launch exposed a CLI/script version mismatch for `--feature-cache`; after syncing the reviewed CLI, one duplicate task-owned launch was stopped and a single run was used. No other user's process was touched.
- Final specification review found that the first `continual_anchor` implementation advanced task-by-task but omitted historical replay. Before acceptance, it was replaced by deterministic seed-42, train-only, Oracle-member-stratified replay. A second audit caught Pair rows being counted in two per-expert buckets; selection was tightened so the union itself hard-caps every Oracle member at 32, then unit-tested and formally retrained. Validation metrics were unchanged. Only stage-6 Flickr30k routes changed, so that cell was backed up, regenerated after each correction, and officially rescored; deterministic replay produced zero text mismatches on unchanged routes. The accepted checkpoint and route manifests use the final union-capped hashes above.
- The learned router is almost completely collapsed to Single on test (six Empty, zero Pair), so the full test matrix does not empirically exercise Pair composition and cannot establish Pair usefulness. The low Stage 05 retrieval quality remains an input-side bottleneck. These are negative method results, not engineering acceptance failures, and no test metric was used to retune the Router.

Machine-readable metrics are in `experiments/runs/v6_ucit_staged/stage06_set_router/metrics/`; formal scripts, frozen config, and preregistration manifest are adjacent. Full logs and large artifacts remain in the corresponding `/data/ckpt` tree.
