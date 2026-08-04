# Data Shortcut Finding: Template-Determined Answers (2026-08-04)

**Severity: blocking for function-purity claims.** Discovered during Stage 03
shortcut probing. In the entire format-controlled dataset (train 1600, val
200, test 400 per task; all five tasks; all splits), the A/B answer is
deterministically determined by the instruction phrasing alone. The image
content is not needed to answer any question.

## Evidence

Per-split template→answer purity (template = first question sentence with
digits normalized to N; colors/shapes kept):

| split | task | templates | answer-pure | samples |
|---|---|---|---|---|
| train | A_only | 48 | 48 | 1600 |
| train | B_only | 4 | 4 | 1600 |
| train | C_only | 525 | 525 | 1600 |
| train | A_plus_B | 24 | 24 | 1600 |
| train | B_plus_C | 48 | 48 | 1600 |
| val | all | — | all pure | 200 each |
| test | all | — | all pure | 400 each |

BOW logistic-regression probe on the test split (5-fold CV): **100%** on
A_only, B_only, A_plus_B, B_plus_C; 99.75% on C_only (2 samples fall back to
A_only phrasing). Image-global-mean probe: ~51.5% (chance) — no usable
visual signal beyond the text.

## Root cause

`compose/data/controlled_format_v1.py` schedules one sample per scene with
`idx` even → positive (answer A) and `idx` odd → negative (answer B), and
uses `PROMPTS[task][idx % 4]` for the question phrasing. Template index
parity therefore equals answer letter:

- A_only: "Is the {color} **object** a {shape}?" / "**object** {shape}-shaped?" → A;
  "Is the {color} **shape** a {shape}?" / "Is the {color} **one** a {shape}?" → B
- B_only: "Are there **exactly** {count} objects in the image?" /
  "Is the total number of objects **equal to** {count}?" → A;
  "Does the image **contain exactly** {count} objects?" /
  "Are there **precisely** {count} objects shown?" → B
- C_only: "**to the left of**" / "**positioned to the left of**" → A;
  "**located right of**" / "**on the right side of**" → B
- A_plus_B: "Are there **exactly** {count} {shape}s in the image?" /
  "Is the number of {shape}s **equal to** {count}?" → A;
  "Does the image **contain exactly** {count} {shape}s?" /
  "Are there **precisely** {count} {shape}s?" → B
- B_plus_C: "Are there **exactly** {count} objects to the left of ..." /
  "Is the number of objects to the left of ... **equal to** {count}?" → A;
  "Does the image **have exactly** {count} objects left of ..." /
  "Are there **precisely** {count} objects on the right of ..." → B

## Consequences for the experiment

1. **Single-function "experts" are template-polarity classifiers.** The
   observed high single-function accuracy (e.g. B_only 86.8%/76.5%/76.0%)
   can be achieved by mapping phrasing→A/B without performing the function.
   Base (50%) fails only because the synthetic phrasings were unseen in
   pretraining. The one-epoch LoRA can memorize the mapping.
2. **Function-purity conclusions are invalid on this data** (Stage 03
   classification C): "B_only 很强但存在数据捷径：专家功能纯度结论无效，
   需要重新生成数据".
3. **The composition measurements remain valid as measurements** of how
   these (lexicon-driven) LoRA experts combine, and the failure is
   reproducible, but the interpretation "composable functional experts
   fail" cannot be supported — the V6 function-composition claim has no
   valid evidence on this dataset either way.
4. Stage 04 compatibility training inherits the confound: any expert
   trained on these templates will learn the phrasing mapping for its
   training split; B+C transfer becomes a test of lexicon transfer, not
   function transfer.
5. The pre-registered validation ("negative_type consistent with polarity
   ✅ PASS") checked consistency, not polarity-phrasing independence; the
   independence condition was not part of the validation.

## Recommendation for the final decision

The V6 dual-LoRA composition claims must be judged on data where the answer
requires the image. Options: (a) regenerate the controlled data with
polarity de-correlated from phrasing (same template for both polarities,
answer = function of image), (b) restrict V6 claims to what this dataset can
support. Stage 04/05 measurements still run as pre-registered, with this
finding explicitly marked as the dominant caveat.
