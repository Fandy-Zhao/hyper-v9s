# Format-Controlled Dataset Validation Report

**Dataset:** `controlled_format_v1`  
**Path:** `experiments/data/controlled_format_v1/`  
**Generated:** 2026-07-31  
**Validation Status: ✅ ALL CHECKS PASSED**

---

## Validation Summary

| Check | Description | Result |
|-------|-------------|--------|
| 1 | All image paths exist | ✅ PASS |
| 2 | All scene metadata files exist | ✅ PASS |
| 3 | Train/val/test scene_id no overlap | ✅ PASS |
| 4 | Train/val/test image hash no overlap | ✅ PASS |
| 5 | All answers are A or B only | ✅ PASS |
| 6 | A/B balance per task per split (50±2%) | ✅ PASS |
| 7 | required_functions correctness | ✅ PASS |
| 8 | A_only: no counting/spatial terms | ✅ PASS |
| 9 | B_only: no shape/spatial terms | ✅ PASS |
| 10 | C_only: no counting terms | ✅ PASS (4 edge-case fallback warnings) |
| 11 | A_plus_B: shape + counting terms present | ✅ PASS |
| 12 | B_plus_C: spatial + counting terms present | ✅ PASS |
| 13 | Answer vs scene ground truth matches | ✅ PASS (0 mismatches) |
| 14 | negative_type consistent with polarity | ✅ PASS |
| 15 | No duplicate (question, image) pairs | ✅ PASS |
| 16 | No exact duplicate samples | ✅ PASS |
| 17 | Image distribution consistent across tasks | ✅ PASS |
| 18 | JSON loadable by training loader | ✅ PASS |
| 19 | Tokenizer answer encoding verified | ✅ PASS |
| 20 | Deterministic reproducibility (seed=730) | ✅ PASS |

### Warnings (non-blocking)

- 4 C_only samples (0.18% of total) lack explicit spatial wording — these are edge-case fallback samples where all objects in the scene share the same x-coordinate. Their required_functions are still ["C"] and their answers are derived from object-pair relations.

---

## A/B Balance

All tasks and all splits achieve exactly 50%/50% A/B balance:

| Task | Train (A/B) | Val (A/B) | Test (A/B) |
|------|-------------|-----------|-------------|
| A_only | 800/800 | 100/100 | 200/200 |
| B_only | 800/800 | 100/100 | 200/200 |
| C_only | 800/800 | 100/100 | 200/200 |
| A_plus_B | 800/800 | 100/100 | 200/200 |
| B_plus_C | 800/800 | 100/100 | 200/200 |

---

## Negative Type Distribution (Train)

| Task | count_negative | attribute_negative | relation_negative |
|------|---------------|-------------------|-------------------|
| A_only | — | 800 | — |
| B_only | 800 | — | — |
| C_only | — | 1* | 799 |
| A_plus_B | 400 | 400 | — |
| B_plus_C | 400 | — | 400 |

\* 1 edge-case fallback sample.

---

## Scene Distribution

| Property | Value |
|----------|-------|
| Total scenes | 2,200 |
| Train scenes | 1,600 |
| Val scenes | 200 |
| Test scenes | 400 |
| Object count range | 3–8 |
| Shapes | circle (3,637 avg per scene), square, triangle |
| Colors | red, blue, green, yellow (balanced ~25% each) |

---

## Image Distribution

| Shape | Total across all scenes |
|-------|------------------------|
| circle | 4,092 (33.6%) |
| square | 4,066 (33.4%) |
| triangle | 4,044 (33.2%) |

| Color | Total across all scenes |
|-------|------------------------|
| red | 3,078 (25.2%) |
| green | 3,059 (25.1%) |
| yellow | 3,035 (24.9%) |
| blue | 3,030 (24.8%) |

Object count distribution is roughly uniform across 3–8.

---

## Tokenizer Audit

| Property | Value |
|----------|-------|
| Model | llava-v1.5-7b (Vicuna/LLaMA) |
| Vocab size | 32,000 |
| Answer "A" token ID | 319 (single token) |
| Answer "B" token ID | 350 (single token) |
| A in context | 1 token |
| B in context | 1 token |

Both A and B are single tokens in the full conversation context (after "ASSISTANT: ").

---

## Manifest

| Property | Value |
|----------|-------|
| Schema version | 2 |
| Generator version | controlled_format_v1 |
| Split policy | scene_level |
| Min horizontal gap | 30px |
| Manifest SHA-256 | `ff4be5d134b266eced3dbe02d2acaafd0551872eff8397ed07fc122ba1f19138` |
| Git commit | `dc7ad7c` |

---

## Reproducibility

The dataset can be completely reproduced with:
```
python -m compose.data.controlled_format_v1 \
  --output-root <path> \
  --seed 730 \
  --train-size 1600 \
  --val-size 200 \
  --test-size 400
```

---

## Human Inspection

An HTML inspection page with 600 randomly sampled cards (20 positive + 20 negative per task per split) is available at:
[docs/reports/format_controlled_dataset_samples.html](format_controlled_dataset_samples.html)
