# Dual-LoRA Synergy Root-Cause Diagnosis — Stage 01: Two-Dimensional Oracle Weight Search

Date: 2026-08-05
Branch: `exp/compose-root-cause-diagnosis`

## 1. Question

Is the fixed combination weight (1,1) the main cause of the dual-LoRA
composition failure? For each of the four expert pairs on seed 42, scan
(alpha, beta) over {0.0, ..., 1.5}^2 in three modes and compare the best
point against the best single expert and the rank-16 control.

## 2. Setup

Pairs (checkpoint, dataset, expert ids):

| pair | checkpoint | dataset | experts | best single ref | rank16 ref |
|---|---|---|---|---|---|
| A+IndependentB | assembled/seed42/a_independent_b | A_plus_B | (0,1) | expert_a / independent_b | rank16_ab |
| A+ResidualB | seed42/residual_b | A_plus_B | (0,1) | expert_a / residual_b | rank16_ab |
| IndependentB+C | assembled/seed42/independent_b_c | B_plus_C | (1,2) | independent_b / expert_c | upper_bc |
| ResidualB+C | assembled/seed42/residual_b_c | B_plus_C | (1,2) | residual_b / expert_c | upper_bc |

Modes:

- A raw: y = base + alpha*u_1 + beta*u_2
- B RMS-calibrated: y = base + pair_scale * (alpha*kappa_1^l*u_1^l + beta*kappa_2^l*u_2^l), kappa from the VAL split (never test); the (1,1) point is the current RMS-calibrated combination
- C first-expert-fixed: the alpha=1 slice of mode A (extracted, no extra forwards)

Fixed points included in every surface: (1,0), (0,1), (1,1), (0.5,0.5),
(1/sqrt(2),1/sqrt(2)), current RMS combination, current formal setting.

Per-point metrics: accuracy, answer-token NLL, Brier, ECE, mean/median
synergy, positive synergy rate, worst-10% mean synergy, conditional
marginals G_2|1 and G_1|2, accuracy delta vs best single, accuracy gap vs
rank-16. Per-sample oracle (diagnostic only): per-sample best point and the
fraction needing both non-zero experts.

## 3. Results

PENDING — coarse grids running (4 pairs × 256 points × 2 modes on GPUs 4-7).

## 4. Stage gate

The combination method is extended to seeds 42/43/44 only if, on seed 42:

- accuracy exceeds the best single expert by >= 1.0 pp; or
- accuracy ties while Brier and ECE both improve with no increase in error count.

## 5. Classification

Determined by the best-point geometry (sections 情况 A-D of the task).

Artifacts: `artifacts/dual_lora_stage01/grid_results.parquet`,
`per_sample_oracle.parquet`, `figures/`.
