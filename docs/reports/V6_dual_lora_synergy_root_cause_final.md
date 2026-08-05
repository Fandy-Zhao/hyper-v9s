# V6 Dual-LoRA Synergy Root-Cause Diagnosis — Final Report (Stage 05)

Date: 2026-08-05
Branch: `exp/compose-root-cause-diagnosis`

> PENDING — populated as stages 01-04 complete. This file is the Stage 05
> deliverable with the four-way decision (A/B/C/D) and the V6 recommendation.

## 1. Experiment questions (from the task)

1. Is the dual-LoRA failure only a bad fixed combination weight?
2. Which layers / LoRA target modules carry the conflict?
3. Are the independent experts incompatible task solutions rather than
   composable functional experts?
4. Why does conditional-residual training work on seen combinations but
   fail on unseen ones?
5. Can minimal compatibility-aware training make two rank-8 experts beat
   the best single expert on an unseen combination?

## 2. Reproduction and implementation audit (Stage 00)

- Code audit: PASS on all 10 checklist items.
- Numerical equivalence T1-T6: PASS (bitwise where deterministic).
- Reproduction: 30/30 evals bitwise identical to the recorded formal results.
- Data shortcut finding: the A/B answer is deterministically determined by
  the instruction phrasing in every split of every task; the image is never
  needed. Function-purity conclusions are invalid on this data
  (see `dual_lora_stage03_shortcut_finding.md`).

## 3. Weight oracle (Stage 01)

PENDING.

## 4. Layer/module conflict (Stage 02)

PENDING.

## 5. Function purity (Stage 03)

PENDING (shortcut probe results available; gradient signatures pending).

## 6. Compatibility training (Stage 04)

PENDING.

## 7. Seen vs unseen combinations

PENDING.

## 8. Accuracy / NLL / Brier / ECE consistency

PENDING.

## 9. Costs

PENDING (parameter counts, GPU hours, peak memory, inference latency).

## 10. Cross-seed stability

PENDING.

## 11. Failure-sample analysis

PENDING.

## 12. Verdict on the V6 core hypothesis

PENDING — one of:

- Decision A: composition hypothesis holds.
- Decision B: conditional (compatibility-aware expert formation required).
- Decision C: seen-context only; no unseen transfer.
- Decision D: composition hypothesis rejected; stop full-LoRA composition.

## 13. Evidence, claims, and next steps

- Three strongest supporting pieces of evidence: PENDING
- Three strongest opposing pieces: PENDING
- Narrowest claim supportable now: PENDING
- What cannot be claimed: PENDING
- Minimal V6 next-version changes: PENDING
- Is the Key Router worth implementing: PENDING
- Is Probe worth restoring: PENDING
- Rank-level or layer-level experts instead: PENDING
