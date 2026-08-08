# Format-Controlled Training Preflight

- source root: `experiments/data/controlled_format_v1`
- output root: `experiments/data/controlled_format_v1_training`
- status: **PASSED**

## 1. Files and counts

| task/split | records | expected | ok |
|---|---|---|---|
| A_only/test | 400 | 400 | True |
| A_only/train | 1600 | 1600 | True |
| A_only/val | 200 | 200 | True |
| A_plus_B/test | 400 | 400 | True |
| A_plus_B/train | 1600 | 1600 | True |
| A_plus_B/val | 200 | 200 | True |
| B_only/test | 400 | 400 | True |
| B_only/train | 1600 | 1600 | True |
| B_only/val | 200 | 200 | True |
| B_plus_C/test | 400 | 400 | True |
| B_plus_C/train | 1600 | 1600 | True |
| B_plus_C/val | 200 | 200 | True |
| C_only/test | 400 | 400 | True |
| C_only/train | 1600 | 1600 | True |
| C_only/val | 200 | 200 | True |

## 2. Images

- referenced images: 2200
- missing images: 0

## 3. Metadata preservation

- missing scene_id: 0
- missing required_functions: 0

## 4. Answers and template

- answer violations: 0
- template violations: 0

## 5. Tokenizer audit (standalone)

| text | token ids | token count | ok |
|---|---|---|---|
| 'A' | [319] | 1 | True |
| 'B' | [350] | 1 | True |
| ' A' | [29871, 319] | 2 | False |
| ' B' | [29871, 350] | 2 | False |

## 6. In-prompt supervision audit (loss mask)

- samples audited: 320
- violations: 0
- supervised token sets observed: [(319, 2), (350, 2)]

## 7. Scene / image split overlap

- scene overlap: 0
- image-hash overlap: {'train': 0, 'val': 0}

## 8. Loader audit

| task | batch | input_ids | labels | supervised min | supervised max |
|---|---|---|---|---|---|
| A_only | 64 | [64, 65] | [64, 65] | 2 | 2 |
| B_only | 64 | [64, 67] | [64, 67] | 2 | 2 |
| C_only | 64 | [64, 70] | [64, 70] | 2 | 2 |
| A_plus_B | 64 | [64, 67] | [64, 67] | 2 | 2 |
| B_plus_C | 64 | [64, 73] | [64, 73] | 2 | 2 |

## 9. Converted files manifest

- manifest: `experiments/data/controlled_format_v1_training/training_manifest.sha256`
- sha256: `d9c812dbe95a535087ec5a6207c4405647c4e758b3c4d9f7992c54b3f4512cb9`

## 10. Conclusion

- status: **PASSED**
