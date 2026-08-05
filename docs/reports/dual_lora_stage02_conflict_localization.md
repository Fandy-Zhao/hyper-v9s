# Dual-LoRA Synergy Root-Cause Diagnosis — Stage 02: Layer/Module Conflict Localization

Date: 2026-08-05
Branch: `exp/compose-root-cause-diagnosis`

## 1. Question

Where does the negative synergy come from: which Transformer layers and
which LoRA target modules? No re-training — only forward-time gate masks on
the frozen experts (seed 42 first).

## 2. Per-layer diagnostics

One pair forward (gates 1,1) over the test set capturing, per sample and
per ComposeLinear layer:

- ||u_1^l||, ||u_2^l|| (RMS over tokens)
- cos(u_1^l, u_2^l) (mean over tokens)
- ||u_1^l + u_2^l||
- cancellation ratio ||u_1+u_2|| / (||u_1||+||u_2||)
- dominance ratio max/min

Stratified by sample groups (computed from recorded evaluations, no new
runs): pair-correct & singles-wrong (gained), pair-wrong & single-correct
(lost), both ok, both wrong, worst-10% synergy, cross-seed repeated failures
(failing in all three seeds).

## 3. Module/layer ablations

Module scopes (all layers): attn_all(q,k,v,o), qv, qk, vo, ffn(gate,up,down),
gate, updown, split_attn_ffn (expert1 on attention, expert2 on FFN),
split_ffn_attn (expert2 on attention, expert1 on FFN), full.

Layer scopes (all modules): low (first 1/3), mid, high (last 1/3),
x1_lowmid_x2_midhigh, x2_lowmid_x1_midhigh, pos_cos layers, neg_cos layers
(negative control).

Per scope: accuracy, NLL, Brier, ECE, synergy statistics, marginals.

## 4. Answers to be established

1. Is negative synergy concentrated in attention or FFN?
2. In low/mid/high layers?
3. Does negative cosine correlate with accuracy collapse?
4. Does RMS calibration reduce dominance without reducing cancellation?
5. Is there a simple layer/module division of labor that recovers composition?

Pass condition: a no-training module/layer scope beats the full-layer
combination by >= 3 pp on at least 2/3 seeds and is not more than 1 pp below
the best single expert.

## 5. Results

PENDING — diagnostics + ablations run after the Stage 01 grid frees GPUs.

Artifacts: `artifacts/dual_lora_stage02/layer_module_metrics.parquet`,
`failure_samples_{pair}.jsonl`, `ablation_{pair}.parquet`, `figures/`.
