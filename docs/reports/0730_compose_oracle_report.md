# Compose Fixed-set Oracle Report

## Definition

The audit uses per-sample teacher-forced NLL in fp32 after multimodal token
expansion. For two experts, every sample is evaluated with four stable
candidates: empty, Expert 0, Expert 1, and the L2-normalized pair with gates
`[1/sqrt(2), 1/sqrt(2)]`. Multimodal preparation is shared, but each candidate
executes a separate decoder forward; no base-transformer reuse is claimed.

## Correctness and determinism

- The Compose unit suite passes 42/42 tests.
- Two independent four-sample GPU smoke runs produced byte-identical JSONL
  caches with SHA-256
  `172919d07c802650145ca380671db33f2c6aff604c8c48a03433a38934708c83`.
- The formal cache contains 6,000 unique samples: 3,000 ImageNet-R and 3,000
  IconQA.
- Each sample records all raw candidate NLLs, target token count, selected
  indices, checkpoint IDs, commit, and configuration hash; full logits are not
  stored.

## Formal audit

| Metric | Value |
| --- | ---: |
| Samples | 6,000 |
| PairOracleRate | 30.6000% |
| PositiveSynergyRate | 31.3333% |
| MeanSynergy | -0.03851718 |
| MedianSynergy | -0.00094279 |
| Average active experts | 1.289667 |
| Mean NLL: empty | 1.53596694 |
| Mean NLL: Expert 0 | 0.75015122 |
| Mean NLL: Expert 1 | 1.11013597 |
| Mean NLL: L2 pair | 0.16488876 |

Relative-synergy pair acceptance is 31.3333%, 30.4000%, 29.8833%, 29.5167%,
and 28.6167% at thresholds 0%, 1%, 2%, 3%, and 5% respectively.

The run completed in 21m06s on four RTX 4090 GPUs. Each process evaluated
1,500 samples at 1.19--1.24 samples/s and peaked at approximately 14.84 GB.

## Matched-capacity result

The rank-16 control has exactly 39,976,960 adapter parameters, equal to the
two rank-8 experts combined. Its mean NLL is 0.11385490, compared with
0.16488876 for the L2 pair. The L2 pair beats rank-16 NLL on only 22.9500% of
samples, and `MeanPairMinusRank16` is +0.05103385. Direct summation improves
the pair mean NLL slightly to 0.15508867 but remains worse than rank-16.

Generation reaches 80.9667% for the L2 pair, 82.3833% for direct summation,
85.4333% for the per-sample Oracle best singleton, and 86.5333% for rank-16.
Thus the extra-capacity control explains the apparent pair benefit and both
normalization variants underperform a single matched-capacity adapter.

## Stop decision

Stop condition C is triggered. The deterministic Oracle implementation is
validated, but this two-expert experiment does not establish useful
composition. Set Router implementation and training stop here; no Router
interface is added by this task.

## Paths

- Formal cache: `/data/ckpt/zhaozhuofan/compose/oracle/task1_task4_experts01_rank8_seed42/oracle_cache.jsonl`
- Formal metrics: `/data/ckpt/zhaozhuofan/compose/oracle/task1_task4_experts01_rank8_seed42/metrics.json`
- Smoke caches: `/data/ckpt/zhaozhuofan/compose/oracle/smoke/task1_task4_run{1,2}_0730`
