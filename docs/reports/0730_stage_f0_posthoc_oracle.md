# Stage F0 Post-hoc Oracle Audit

## Scope and definitions

This is an offline audit of the existing 6,000-sample caches. No model was trained or evaluated again. Best single and best pair are selected per sample by teacher-forced NLL; Oracle generation accuracy uses the prediction attached to that NLL-selected configuration.

## Oracle A

- Mean NLL: `0.10853885`
- Generation accuracy: `88.3333%`
- Accuracy gain over best single: `+2.9000` percentage points
- Mean NLL improvement over best single: `+0.01783272`
- Selection rates: `{"base": 0.016, "pair_direct_sum": 0.5268333333333334, "pair_l2": 0.12966666666666668, "single0": 0.04683333333333333, "single1": 0.2806666666666667}`

## Oracle B

- Mean NLL: `0.09401556`
- Generation accuracy: `89.8167%`
- Accuracy gain over rank16: `+3.2833` percentage points
- Mean NLL improvement over rank16: `+0.01983935`
- Selection rates: `{"pair_direct_sum": 0.37, "pair_l2": 0.0835, "rank16": 0.5465}`

## Pair-exclusive value

- PairExclusiveNLLRate: `40.4500%`
- PairExclusiveCorrectRate: `1.4000%` (`84` samples)
- Rank16CorrectPairWrong: `5.3833%` (`323` samples)
- Rank16WrongPairCorrect: `3.4500%` (`207` samples)
- Pair/rank16 error-set Jaccard: `0.53138815`

## Synergy distribution

- Mean / median / std: `-0.00862003` / `0.00021651` / `0.10745805`
- P1/P5/P10/P25/P50/P75/P90/P95/P99: `-0.43603436, -0.14558347, -0.06574606, -0.00041835, 0.00021651, 0.00564039, 0.04349204, 0.09444700, 0.25792948`
- Positive / negative subset mean: `0.024577087674241303` / `-0.0743711155044772`
- Maximum positive / negative gain: `1.17142290` / `-1.94136980`
- Negative mean characterization: `tail_concentrated_or_not_widespread`; worst 10% of negative samples account for `55.6941%` of negative magnitude.

| Scope | Mean | Median | Std | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 | Positive mean | Negative mean | Max positive | Max negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | -0.00862003 | 0.00021651 | 0.10745805 | -0.43603436 | -0.14558347 | -0.06574606 | -0.00041835 | 0.00021651 | 0.00564039 | 0.04349204 | 0.09444700 | 0.25792948 | 0.02457709 | -0.07437112 | 1.17142290 | -1.94136980 |
| UCIT/IconQA | -0.02409386 | -0.00002759 | 0.13798789 | -0.58490853 | -0.23237938 | -0.12102841 | -0.03177784 | -0.00002759 | 0.00276341 | 0.05824133 | 0.11465111 | 0.28092027 | 0.04509962 | -0.07481977 | 1.17142290 | -1.94136980 |
| UCIT/ImageNet-R | 0.00685380 | 0.00087112 | 0.05979071 | -0.19414501 | -0.02117375 | 0.00001002 | 0.00015791 | 0.00087112 | 0.00646374 | 0.03221616 | 0.07408455 | 0.23023937 | 0.01499537 | -0.07161716 | 0.50993794 | -1.20234489 |

## Requested subset comparisons

| Scope | Subset | Samples | Mean synergy | Mean pair minus rank16 NLL |
| --- | --- | ---: | ---: | ---: |
| overall | pair_positive_synergy | 3987 | 0.024577087674241303 | -0.013521919925331983 |
| overall | pair_negative_synergy | 2013 | -0.0743711155044772 | 0.08978248409000233 |
| overall | pair_better_than_rank16 | 2721 | 0.018934776483017262 | -0.04374718004758504 |
| overall | pair_not_better_than_rank16 | 3279 | -0.03148573765282174 | 0.07497905539504574 |
| UCIT/ImageNet-R | pair_positive_synergy | 2718 | 0.014995374059120737 | -0.00985934736107776 |
| UCIT/ImageNet-R | pair_negative_synergy | 282 | -0.07161715995433507 | 0.11786828406490603 |
| UCIT/ImageNet-R | pair_better_than_rank16 | 1889 | 0.011487360007168473 | -0.029315353200564 |
| UCIT/ImageNet-R | pair_not_better_than_rank16 | 1111 | -0.0010245143726134768 | 0.05564163112039562 |
| UCIT/IconQA | pair_positive_synergy | 1269 | 0.04509962321868393 | -0.021366578892741732 |
| UCIT/IconQA | pair_negative_synergy | 1731 | -0.07481976684193536 | 0.08520698114781698 |
| UCIT/IconQA | pair_better_than_rank16 | 832 | 0.035843634323015294 | -0.07651367153078545 |
| UCIT/IconQA | pair_not_better_than_rank16 | 2168 | -0.0470957095459543 | 0.08488859338818977 |

All source hashes are preserved in `posthoc_oracle.json`; every sample, NLL, prediction, answer, and derived selection is preserved in `per_sample_audit.csv`.

## Pre-registered decision

- Stop condition triggered: `false`
- Oracle B accuracy gain: `+3.2833` pp (threshold `< 0.5`)
- Oracle B mean NLL improvement: `+0.01983935` (threshold `< 0.005`)
- PairExclusiveCorrectRate: `1.4000%` (threshold `< 2%`)
- Decision: 当前 pair 对 rank16 存在超过预注册停止门槛的局部互补；仅保留为后续实验基线，不实现 Router。

Regardless of this gate, no rank16/pair Router or Set Router is implemented.
