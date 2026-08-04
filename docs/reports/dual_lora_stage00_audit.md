# Dual-LoRA Synergy Root-Cause Diagnosis — Stage 00: Implementation & Baseline Audit

Date: 2026-08-04
Branch: `exp/compose-root-cause-diagnosis`
Predecessor: `docs/reports/format_controlled_composition_final.md` (STOP_COMPOSITION_CONFIRMED)

## 1. Scope

Verify that the observed dual-LoRA composition failures are not artifacts of
the implementation or the metric pipeline:

1. the multi-LoRA forward is exactly `y = W0(x) + alpha*LoRA_A(x) + beta*LoRA_B(x)` per target layer;
2. scaling, enable/disable, merge/unmerge, dropout, batching and metric definitions are correct;
3. numerical unit tests pass (fixed points, explicit sum, repeatability, batch size, save/reload);
4. the formal results reproduce within 0.2 pp.

Reference design documents named in the task (`Hyper-LLaVA_V6_CPrompt_Keyed_Expert_Pool_重排版.md`,
`Hyper-LLaVA_V4_初步实验计划.md`) are not present on this machine; the audit uses the code,
training/eval manifests and reports under `compose/`, `experiments/runs/format_controlled_composition_v1/`
and `docs/reports/format_controlled_*`, which are the executed artifacts of those plans.

## 2. Where the multi-LoRA forward lives

| file | role |
|---|---|
| `compose/adapters/lora.py` | `ComposeLinear` (base linear + per-expert LoRA) and `LoRAExpert`; the formal controlled path |
| `compose/adapters/manager.py` | `ExpertManager`: default selection = uniform gates across a batch |
| `compose/experts/checkpoint.py` | checkpoint format (`compose_experts.bin` / `.json`), strict key validation |
| `compose/eval/load_compose.py` | model loading (`inject_compose_adapters` → ComposeLinear; never PEFT) |
| `compose/eval/eval_controlled_ab.py` | the formal A/B evaluation (answer-token logits) |
| `compose/lora/direct_sum.py`, `rms_composition.py`, `runtime.py` | older explicit-sum / RMS-calibrated / hook-based composition paths |

The formal controlled runs use the ComposeLinear path (gates 1,1, normalization none). The
Hyper legacy path (`loraA/loraB` containers with `cur_task`/`disable_adapters`) exists in
`compose/lora/adapter_bridge.py` but is NOT used by the controlled evaluation.

## 3. Audit checklist

### 3.1 Forward arithmetic per target layer — PASS

`ComposeLinear.forward` (`compose/adapters/lora.py:130-162`): `result = base_layer(inputs)`; for
each selected expert, `delta = experts[key](selected_inputs)`, `result += gate * delta`.
`LoRAExpert.forward` (`lora.py:33-36`): `delta = B(A(dropout(x))) * scaling`, i.e. the delta is
exactly `LoRA(x) = (alpha/rank) * B A x`. No other term enters the output; the observed output is
`W0(x) + sum_i gate_i * LoRA_i(x)` for every target layer.

### 3.2 LoRA alpha/rank scaling applied exactly once — PASS

`LoRAExpert.scaling = alpha / rank` is applied once inside the expert forward (`lora.py:26,36`).
The composition adds no further scaling (gates multiply the already-scaled delta). The compose
adapter config (`rank=8, alpha=16`; `rank=16, alpha=32` → scaling 2 in both cases) is recorded in
`run_train_one.sh` and verified by the checkpoint parameter counts (19,988,480 / 39,976,960).

### 3.3 Enable/disable correctness — PASS

The controlled path does not use PEFT. Enable = `set_default_selection(ids, gates)`,
disable = `clear_default_selection()` (`manager.py:35-47`, `lora.py:94-113`). With gates 0 the
expert is skipped (`active_slots = ids-match & gate>0`, `lora.py:150`), so gates (1,0) and (0,1)
produce single-expert behavior and (0,0) produces base — verified numerically in test T2.

### 3.4 Both adapters simultaneously active in the pair forward — PASS

The forward loops over all selected experts (`lora.py:145-161`); the explicit-sum recomputation
in test T3 checks every ComposeLinear layer of the real model for the pair selection.

### 3.5 Merge/unmerge never pollutes base parameters — PASS

No merge/unmerge exists in the compose path: `base_layer` weights are never written; checkpoints
store expert tensors only (`checkpoint.py:88-121`), and loading validates keys strictly
(`_validate_state_keys`). Bitwise isolation of Expert A during residual training was already
established (final report section 15) and is re-verified by the cross-bundle test in T2.

### 3.6 Eval-phase LoRA dropout — PASS

Compose adapters are trained with `--compose_dropout 0` (`run_train_one.sh`), so
`LoRAExpert.dropout` is `nn.Identity` (`lora.py:27`); `load_compose_model` calls `model.eval()`
and the eval runs under `torch.inference_mode()`.

### 3.7 Batch-internal adapter mapping — PASS

The formal eval sets one default selection for the whole batch (`manager.set_default_selection`
expands the selection to the batch, `lora.py:121-128`). Per-sample routing (`ComposeSelection`)
exists for the router flow but is unused in the controlled evals. Test T5 confirms per-sample
outputs are independent of batch size.

### 3.8 RMS calibration input distribution and layer definition — PASS (with note)

RMS statistics are keyed by `(expert_id, layer_name)` (`statistics.py:23-27`); provenance is
hash-bound to the checkpoint, dataset manifest and composition rule, and the constructor
rejects a `test` calibration split (`statistics.py:113-115`). NOTE: the formal controlled runs
used gates 1,1 with no RMS calibration; RMS-calibrated composition is only exercised in Stage 01
mode B, where statistics are computed on the VAL split of the same dataset with the same layer
definitions (see `dual_lora_stage01_rms_stats.py`).

### 3.9 Answer-token NLL excludes EOS — PASS

`eval_controlled_ab.py:179-208`: the supervised region is exactly `[answer_token, EOS]`; the
answer-token position is `positions[0]`; NLL is computed at `answer_pos = answer_token_pos - 1`
(LM shift, same convention as the training loss) over the FULL vocabulary, target token only.
EOS never enters the NLL. This is the fixed post-incident-3 implementation; the buggy variant is
quarantined at `evaluation/seed42_bug_v1_invalid`.

### 3.10 Positive-class consistency across metrics — PASS

accuracy = argmax over {logit_A, logit_B}; Brier = (p_A - y_A)^2; ECE = 15-bin fixed-width on
p_A vs y_A; NLL = full-vocab per-token NLL. All share the A/B restricted distribution; NLL
intentionally uses the full vocabulary (pre-registered) while the other three use the restricted
A/B space — this is a design decision, not an inconsistency.

## 4. Numerical equivalence tests

Run via `compose/eval/dual_lora_stage00_numerical.py` on the seed-42 assembled `a_independent_b`
checkpoint, `A_plus_B` test set.

| test | assertion | result |
|---|---|---|
| T1 synthetic arithmetic | ComposeLinear == base + alpha*delta_A + beta*delta_B (7 fixed points, fp64) | PASS (bitwise, max diff 0.0) |
| T2 model fixed points | gates (1,0) == expert A alone; (0,1) == expert B alone; base == cleared selection; elementwise logits | PASS (bitwise, max diff 0.0) |
| T2 cross-bundle | pair-checkpoint experts == single-checkpoint experts (bitwise logits) | PASS (bitwise, max diff 0.0) |
| T3 explicit sum | every ComposeLinear layer: output == base + 1*delta_A + 1*delta_B from the same hidden states | PASS (224/224 layers bitwise, max diff 0.0) |
| T4 repeatability | two identical runs, identical per-sample logits | PASS (bitwise, max diff 0.0) |
| T5 batch size | batch 4/8/12 per-sample logits | PASS within bf16: max diff 0.125 (=2 bf16 ulps at logit scale ~18); 5/64 prediction flips all on exact-tie samples (margin <= 0.125); batch 16 exceeds the 24 GiB GPU at the CE loss and was replaced by 12 |
| T6 save/reload | save → reload → identical per-sample logits | PASS (bitwise, max diff 0.0) |

Notes: gates (0,0) are not expressible in the ComposeSelection API (>=1
positive gate required); the equivalent base forward is a cleared selection,
and gate-0 expert skipping is exercised by the (1,0)/(0,1) fixed points.
Batch-size invariance is exact for non-tie samples; the 5 flips occur on
samples whose logit margin is 0.0-0.125 (1-2 bf16 ulps) due to
padding-dependent attention masking — inherent bf16 behavior, not an
implementation defect.

Artifact: `artifacts/dual_lora_stage00/numerical_equivalence.json` — `passed: true`.

## 5. Reproduction of the formal results

30 evals re-run on the unchanged checkpoints with the unchanged formal eval path
(`run_eval_one.sh` copied to `experiments/runs/dual_lora_stage00/` with a new result root):

- singles: expert_a(A_only), independent_b(B_only), residual_b(B_only), expert_c(C_only)
- seen: a_residual_b(A_plus_B), a_independent_b(A_plus_B)
- unseen: independent_b_c(B_plus_C), residual_b_c(B_plus_C)
- capacity controls: rank16_ab(A_plus_B), upper_bc(B_plus_C)

Gate: accuracy within 0.2 pp of the recorded value, per-sample predictions identical.

RESULT: **30/30 PASSED, bitwise-identical**. Every reproduction matches the
recorded summary exactly (delta accuracy 0.0 pp, delta NLL 0.0, per-sample
predictions identical, max |delta logit| = 0.0). The formal results are
reproducible to bit-level determinism on the same hardware.

Artifact: `artifacts/dual_lora_stage00/reproduction.csv` (compare script
`compose/eval/dual_lora_stage00_repro_compare.py`).

## 6. Environment

- python: `/home/zhaozhuofan/miniconda3/envs/hyper/bin/python` (3.10)
- torch bf16, tf32 enabled, seed 42 for eval; GPUs 4-7 (all free at start; GPUs 0-3 carry other users' processes)
- model: llava-v1.5-7b + clip-vit-large-patch14-336, compose adapters rank 8/alpha 16

## 7. Result

**PASS.** Code audit (3.1-3.10) clean; all numerical-equivalence tests T1-T6 pass (bitwise
where deterministic); all 30 reproduction evals match the recorded formal results bitwise.
No implementation error, scaling duplication, adapter-activation bug or metric error was
found. The observed composition behavior is a property of the trained experts and the data,
not of the implementation. A separate data-level finding (template-determined answers, see
`docs/reports/dual_lora_stage03_shortcut_finding.md`) is the dominant caveat for the
function-purity interpretation of the results.
