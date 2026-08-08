# Format-Controlled Functional Expert Composition Validation - Final Report

## 1. Experiment question

After unifying every task onto the same single-token A/B output interface (same image distribution, same answer space, same output template, same positive/negative balance, same scene-level split), do two low-rank functional LoRAs compose stably - beating the best single expert and approaching or exceeding an equal-parameter rank-16 joint baseline?

## 2. Why a format-controlled revalidation

The old Stage F2 used per-task answer vocabularies (shape words, digits, left/right), so it could not rule out that experts had learned different output lexicons, that expert ability was bound to answer formats, that multi-LoRA composition caused output-token competition, or that decoding biases across digit/shape/relation words produced negative synergy. This experiment removes all of those confounds.

## 3. Data source and hashes

- Source manifest (pre-registered): `ff4be5d134b266eced3dbe02d2acaafd0551872eff8397ed07fc122ba1f19138`
- Source manifest sha256 verified before/after conversion: `aafdb33a1debb37b575c07e5a6fcfec6372532d62984b73c2c185be23973c9ed` (unchanged)
- Converted training root: `experiments/data/controlled_format_v1_training` (metadata preserved; independent manifest sha256 `d9c812dbe95a535087ec5a6207c4405647c4e758b3c4d9f7992c54b3f4512cb9`)
- 2,200 shared scenes; five tasks x 1600/200/400 train/val/test; all answers A=Yes/B=No; 50/50 balance; train/val/test scene_id and image-hash disjoint.

## 4. Tokenizer and loss-mask audit

- `encode("A")` -> [319], `encode("B")` -> [350] (single tokens). `encode(" A")` -> [29871, 319] and `encode(" B")` -> [29871, 350] (two tokens: the leading space goes through the byte fallback).
- In the full LLaVA v1 training prompt the answer token is always a single token; the supervised region is exactly `[answer_token, EOS]` (id 2) across all 320 audited samples, with no newline/space/period/explanation tokens.
- The main answer-token NLL metric is computed on the answer token only (EOS excluded); the training loss covers answer+EOS per the standard template. Preflight status: PASSED (see `docs/reports/format_controlled_training_preflight.md`).

## 5. Training matrix

| model | rank | data | active experts | trainable | params |
|---|---|---|---|---|---|
| expert_a | 8 | A_only | [0] | [0] | 19,988,480 |
| independent_b | 8 | B_only | [1] | [1] | 19,988,480 |
| expert_c | 8 | C_only | [2] | [2] | 19,988,480 |
| residual_b | 8 | A_plus_B | [0,1] | [1] | 39,976,960 |
| rank16_ab | 16 | A_plus_B | [0] | [0] | 39,976,960 |
| task_ab | 8 | A_plus_B | [0] | [0] | 19,988,480 |
| upper_bc | 16 | B_plus_C | [0] | [0] | 39,976,960 |

3 seeds x 7 models = 21 runs, all completed and verified. Residual B|A loads the frozen Expert A and trains only expert 1 (see section 15 for bitwise isolation).

## 6. Parameter-count fairness

Two rank-8 experts = 39,976,960 trainable-adapter params = one rank-16 expert (39,976,960). All 21 manifests verified to match the pre-registered counts exactly (rank-8: 19,988,480; rank-16 and dual: 39,976,960).

## 7. Single-function results (A_only/B_only/C_only test)

| config | A_only acc (s42/s43/s44) | B_only acc (s42/s43/s44) | C_only acc (s42/s43/s44) |
|---|---|---|---|
| base | 50.2%/50.2%/50.2% | 50.0%/50.0%/50.0% | 52.2%/52.2%/52.2% |
| expert_a | 70.8%/68.0%/69.5% | 37.8%/35.2%/38.0% | 61.5%/63.7%/61.3% |
| independent_b | 43.2%/49.8%/49.8% | 86.8%/76.5%/76.0% | 57.0%/67.5%/64.0% |
| expert_c | 49.5%/53.2%/51.2% | 50.0%/50.0%/50.0% | 99.8%/99.5%/99.5% |
| residual_b | 66.2%/73.0%/67.8% | 100.0%/100.0%/100.0% | 50.2%/50.2%/50.2% |

Every expert beats base on its own function in all three seeds; answer-token NLL improves with bootstrap 95% CI lower bounds > 0; hard-negative accuracy exceeds chance (condition 1: 3/3 seeds).

## 8. Seen composition A+B (A_plus_B test)

| metric | seed42 | seed43 | seed44 |
|---|---|---|---|
| best single acc % | 100.0 | 100.0 | 100.0 |
| A+ResidualB acc % | 100.0 | 100.0 | 100.0 |
| rank16 acc % | 70.0 | 73.5 | 73.2 |
| mean synergy (A+RB) | 0.759 | 0.757 | 0.763 |
| median synergy (A+RB) | 0.788 | 0.785 | 0.789 |
| mean synergy (A+IB) | -0.072 | -0.085 | -0.073 |
| G_B_given_A (A+RB) | 0.759 | 0.757 | 0.763 |
| G_A_given_B (A+RB) | 12.01 | 12.34 | 12.04 |

A+ResidualB reaches 100% accuracy in all three seeds - exactly the best single expert (Residual B alone also reaches 100% on A_plus_B). The accuracy headroom is zero, so the pre-registered >=1.0 pp accuracy gain cannot be demonstrated (condition 2: 0/3 seeds). At the NLL level the pair is strongly positive-synergy (mean 0.760 nats, positive rate 1.0, CI > 0) and both conditional marginals are positive, i.e. the two experts are jointly used. A+IndependentB is instead slightly negative-synergy - composition value is specific to the conditional (residual) training, not to independent experts.

## 9. Unseen composition B+C (B_plus_C test)

| metric | seed42 | seed43 | seed44 |
|---|---|---|---|
| best single acc % | 89.5 | 84.0 | 88.2 |
| IndependentB+C acc % | 50.0 | 50.0 | 50.0 |
| ResidualB+C acc % | 74.8 | 71.2 | 73.8 |
| upper_bc (rank16) acc % | 100.0 | 100.0 | 100.0 |
| mean synergy (IB+C) | -0.278 | -0.148 | -0.211 |
| mean synergy (RB+C) | -0.305 | -0.188 | -0.209 |
| median synergy (RB+C) | 0.119 | 0.331 | 0.230 |
| worst10 synergy (RB+C) | -2.82 | -2.56 | -2.62 |
| G_B_given_C (IB+C) | -0.188 | -0.123 | -0.160 |
| G_C_given_B (IB+C) | -0.026 | -0.026 | -0.021 |
| G_C_given_B (RB+C) | 13.1 | 13.5 | 13.8 |

Both unseen pairs FAIL: accuracy drops far below the best single expert (e.g. ResidualB+C 75% vs best 96% in seed42; IndependentB+C collapses to ~50%). Mean synergy is negative; the 10% worst samples carry very large negative synergy (mean ~-2.6), and 101-200 of 400 samples fail in all three seeds (worst-10 cross-seed repetition rates 0.42-0.80). Note that G_C_given_B is strongly positive for ResidualB+C: adding Expert C on top of Residual B lowers answer-token NLL by ~13 nats on average - but the mean NLL gain coexists with an accuracy collapse, i.e. the pair concentrates probability mass on the wrong side for a substantial sub-population. Accuracy and NLL are NOT direction-consistent.

## 10. Independent B vs Residual B

Residual B (trained frozen-on-A on A_plus_B) transfers to B_only (100%/100%/100% acc, NLL improvement CI > 0 in all seeds) and pairs well with A (NLL synergy +0.76). Independent B pairs negatively with A. On unseen B+C both fail equally. So the residual formation creates a context-conditioned counterweight rather than a transferable standalone counting function that composes additively.

## 11. Accuracy / NLL / Brier / ECE

Full per-configuration tables are in `summaries/seedXX.json`; aggregate means across seeds are in `summaries/final_summary.json`. Direction consistency: seen pair improves NLL but ties accuracy (no headroom); unseen pairs degrade accuracy despite mixed NLL signals - no pair shows accuracy-NLL-Brier agreement across all three preregistered comparisons.

## 12. Synergy distributions

| pair | seed | mean | median | positive rate | 95% CI |
|---|---|---|---|---|---|
| a_residual_b | 42 | 0.759 | 0.788 | 1.00 | [0.744, 0.773] |
| a_residual_b | 43 | 0.757 | 0.785 | 1.00 | [0.743, 0.771] |
| a_residual_b | 44 | 0.763 | 0.789 | 1.00 | [0.750, 0.778] |
| a_independent_b | 42 | -0.072 | -0.067 | 0.07 | [-0.078, -0.066] |
| a_independent_b | 43 | -0.085 | -0.069 | 0.07 | [-0.091, -0.078] |
| a_independent_b | 44 | -0.073 | -0.067 | 0.11 | [-0.079, -0.066] |
| independent_b_c | 42 | -0.278 | -0.255 | 0.25 | [-0.309, -0.247] |
| independent_b_c | 43 | -0.148 | -0.143 | 0.23 | [-0.163, -0.132] |
| independent_b_c | 44 | -0.211 | -0.211 | 0.24 | [-0.233, -0.189] |
| residual_b_c | 42 | -0.305 | 0.119 | 0.69 | [-0.410, -0.203] |
| residual_b_c | 43 | -0.188 | 0.331 | 0.67 | [-0.297, -0.083] |
| residual_b_c | 44 | -0.209 | 0.230 | 0.68 | [-0.310, -0.108] |

## 13. Conditional marginal contributions

A+B: G_B_given_A > 0 (mean 0.760) and G_A_given_B > 0 (mean ~12) in all seeds - both experts contribute, so the seen pair is a genuine two-function composition at the NLL level.

B+C: for IndependentB+C both marginals are <= 0; for ResidualB+C G_C_given_B > 0 but G_B_given_C < 0 - only one direction contributes, and it does not rescue accuracy. The 'two positive conditional contributions' criterion fails in both pairs.

## 14. Hard-negative stratification

Per-negative-type accuracy/NLL for every configuration is in `summaries/seedXX.json` (hard_negative section). Single experts clear chance on their own hard negatives. The unseen pairs lose accuracy on both count_negative and relation_negative relative to the best single.

## 15. Parameter isolation

For every seed: Expert A tensors in the residual_b checkpoint are bitwise identical to the expert_a checkpoint (sha256 equal, 448 tensors); fixed input Expert A-only logits before/after residual training have max absolute difference 0.0. Residual B LoRA deltas are orthogonal to Expert A deltas (cosine ~3e-4, per-layer RMS ~4e-5).

## 16. Activation identity

Residual B's activation directions on B_only vs A_plus_B vs B_plus_C are near-identical (cosine ~0.9999, 224 layers): the residual expert activates the same direction regardless of task context - consistent with it acting as a context-conditioned counterweight rather than a task-selective function.

## 17. Latency and memory

Answer-logit inference: ~0.17 s/sample (batch 8, bf16, 7B); free generation ~0.56 s/sample (max_new_tokens 16). Peak CUDA memory ~16.6 GiB per eval. Adding a second LoRA adds no measurable latency (plain sum at forward). Full per-evaluation tables in `evaluation/...`.

## 18. Cross-seed repeated failures

Worst-10% synergy samples repeat across seeds at rate 0.42 (A+IB), 0.53 (IB+C), 0.70 (A+RB), 0.80 (RB+C) - far above the pre-registered 0.30 floor. Failure is systematic, not seed noise.

## 19. Preregistered gate audit

| condition | pass seeds |
|---|---|
| 1 single_function_validity | 3/3 |
| 2 seen_composition | 0/3 |
| 3 independent_unseen_composition | 0/3 |
| 4 residual_unseen_transfer | 0/3 |
| 5 tail_stability | 0/3 |

No threshold was adjusted after results.

## 20. Final automatic decision

**STOP_COMPOSITION_CONFIRMED** - no composition condition passed at >=2/3 seeds under the unified answer format

Interpretation: under the unified single-token A/B interface, single experts are cleanly learned and isolate bitwise, but the fixed additive two-expert combination does not produce stable accuracy gains: the seen A+B pair only ties its best single on accuracy (while being strongly positive at the NLL level), and the unseen B+C pairs fail outright. The old-F2 failure is therefore not primarily caused by answer-format mismatch.

## 21. Method claims and boundaries

- Claims supported: single-function LoRA experts learn cleanly under a unified format; residual training is bitwise-isolated; seen composition improves answer-token NLL; independent-expert additivity does NOT hold on unseen tasks; accuracy and NLL diverge for unseen pairs.
- Claims NOT supported: equal-parameter composition advantage over rank-16; unseen transferable composition; tail stability.
- Boundaries: single rendered scene family (300x260 synthetic scenes, 3-8 objects, 3 shapes, 4 colors); one epoch; rank-8 experts; fixed combination rule (direct sum, gates 1,1, no calibration); no router; answers restricted to A/B.

## 22. Router stage

Not approved. Per section 十九, the decision is STOP_COMPOSITION_CONFIRMED; the Query-Key Router and all other listed components must NOT be implemented. Any follow-up (e.g. expert formation, combination calibration) requires a new pre-registered experiment.
