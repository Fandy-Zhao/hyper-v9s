# Compose Next-stage Report

## 1. Branch and commits

- Branch: `exp/0730-compose-task1-oracle`
- Base: `a5a0a575658ac74ae7bfe6bf461c39afddb186ef`
- No push, pull request, or merge was performed.

## 2. Scope

The work validates Compose execution semantics, matched Task1 parity,
multi-expert isolation, a deterministic per-sample fixed-set Oracle, and a
rank-matched capacity control. It does not implement or train a Set Router.

## 3. Gate normalization change

`ComposeSelection` supports explicit `none`, `l1`, and `l2` modes. User gates
are never silently changed. Single-expert evaluation uses gate 1 with `none`;
Oracle pairs use two unit gates with `l2`, producing exactly
`[1/sqrt(2), 1/sqrt(2)]`.

## 4. Grouped execution implementation

Each expert runs only on selected batch rows and scatters its weighted delta
back into the batch. Rank-2/rank-3 inputs, heterogeneous top-1/top-2 batches,
bf16 forward/backward, and unused-expert non-execution are covered. Duplicate
expert IDs are rejected.

## 5. Task1 full training

Compose Expert 0 trained for 375 steps with 19,988,480 parameters. Mean loss
was 0.20838464, supervision zero count was 0, and 224/224 LoRA-B tensors had
finite gradients.

## 6. Task1 evaluation

Compose achieved 2706/3000 exact matches (90.2000%) on UCIT Task1. Strict
double reloads loaded 448/448 tensors and produced exactly equal 32,000 fixed
sample logits.

## 7. Standard PEFT parity

Matched rank-8 PEFT trained for the same 375 steps with 19,988,480 parameters,
mean loss 0.20884625, and 2705/3000 exact matches (90.1667%). The one-sample
difference passes the single-expert parity gate. No usable Hyper-LLaVA Task1
checkpoint was found, so no Hyper result is reported.

## 8. Multi-expert isolation

Expert 1 is an IconQA visual-reasoning functional proxy. The two-expert pool
contains 896 tensors and 39,976,960 parameters. All 448 Expert 0 tensors and
its fixed-input logits remain exactly unchanged after Expert 1 training.

## 9. Per-sample NLL

Teacher-forced shifted cross-entropy is computed per sample in fp32 while
ignoring `IGNORE_INDEX`; zero-target samples raise. Multimodal preparation is
performed once per sample batch and candidate decoder forwards remain
separate.

## 10. Oracle implementation

The two-expert candidate order is empty, Expert 0, Expert 1, and the L2 pair.
Two smoke runs produced byte-identical caches. The formal 6,000-sample audit
completed in 21m06s with approximately 14.84 GB peak memory per process.

## 11. Rank-16 control

The accepted configuration has rank 16, alpha 32, scale 2, 448 tensors, and
exactly 39,976,960 parameters, matching the two rank-8 experts exactly. It
trained for 842 steps on the same 53,857 mixed records with the same global
batch, epoch, optimizer, learning rate, maximum length, and seed. Mean loss
was 0.25027549; end-to-end runtime was 1h39m55s and peak training memory was
23,718 MiB. All 224 LoRA-B tensors received finite gradients. Two strict
reloads loaded 448/448 tensors with no missing or unexpected keys and produced
exactly equal 32,000 fixed-sample logits (maximum absolute difference 0).

On all 6,000 audit samples, rank-16 mean teacher-forced NLL is 0.11385490.
For comparison, the rank-8 pair has mean NLL 0.16488876 under L2
normalization and 0.15508867 under direct summation. Rank-16 generation
accuracy is 5192/6000 (86.5333%).

## 12. Main results

| Metric | Value |
| --- | ---: |
| PairOracleRate | 30.6000% |
| PositiveSynergyRate | 31.3333% |
| MeanSynergy | -0.03851718 |
| MedianSynergy | -0.00094279 |
| Average active experts | 1.289667 |
| PairBetterThanRank16Rate | 22.9500% |
| MeanPairMinusRank16 | +0.05103385 |
| Base accuracy | 18.4000% |
| Single 0 accuracy | 52.1500% |
| Single 1 accuracy | 47.9667% |
| Oracle best-single accuracy | 85.4333% |
| L2 pair accuracy | 80.9667% |
| Direct-sum pair accuracy | 82.3833% |
| Rank-16 accuracy | 86.5333% |

The rank-8 pair and rank-16 single adapter each contain exactly 39,976,960
parameters. Direct summation improves mean NLL over L2 by 0.00980009, but is
still 0.04123376 worse than rank-16 on average.

Inference measurements use total GPU-seconds divided by 6,000 for per-sample
device latency; four-way runs additionally retain parallel wall time in their
aggregate JSON. Peak memory is the maximum per-device allocation.

| Selection | Device latency (s/sample) | Peak memory (bytes) |
| --- | ---: | ---: |
| Base | 0.222754 | 15,099,399,168 |
| Single 0 | 0.877019 | 15,099,399,168 |
| Single 1 | 0.680591 | 15,099,399,168 |
| L2 pair | 0.988705 | 15,099,399,168 |
| Direct-sum pair | 1.028196 | 15,101,430,784 |
| Rank 16 | 0.785937 | 15,101,430,784 |

## 13. Stop-condition decision

Stop condition C is triggered. Mean and median synergy are negative, the L2
pair loses to rank-16 NLL on 77.05% of samples, and both L2 and direct-sum
generation underperform the exactly parameter-matched rank-16 adapter. The
fixed-set Oracle is valid, but useful expert composition is not established.

## 14. Completed and incomplete items

Completed: Stages A--E, matched Task1 baselines, two-expert isolation,
deterministic Oracle cache, rank-16 training and strict reload, matched
teacher-forced and generation controls, and the final 6,000-sample comparison.
Intentionally not implemented: Set Router or any of its proposed interfaces,
because stop condition C failed the composition gate.

Final validation passes Python compilation, 42/42 unittest cases, syntax
checks for every `scripts/Compose/*.sh` launcher, Compose/Hyper import
isolation, and `git diff --check`. The `hyper` environment does not contain
pytest; no unapproved dependency was installed, and the complete required
suite is covered by unittest.

## 15. Risks

- The two experts are functional proxies trained on task datasets, not proven
  atomic semantic experts.
- Only two experts are audited, so the four-expert 11-candidate invariant is
  unit-tested rather than exercised in a full GPU benchmark.
- Oracle gains are concentrated in a minority of samples and may be explained
  by capacity or generation behavior; the matched control supports that
  explanation here.
- The result covers one two-task pool and does not prove that all future expert
  decompositions will fail.
- Generation latency was measured on shared RTX 4090 servers; total GPU-seconds
  support method comparison, but should not be treated as isolated production
  serving latency.

## 16. Recommendation on whether to implement Set Router

Do not implement the Set Router for this expert pool. First redesign or retrain
experts so a repeated fixed-set audit shows positive mean and median synergy
and beats an exactly parameter-matched single adapter on NLL and accuracy.
