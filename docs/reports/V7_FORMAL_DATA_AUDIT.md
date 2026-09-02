# V7 Formal Data Audit (Original UCIT Train / Test Files)

Audit head: `5819da8` (branch `exp/v7-full-data-global-key-expert-coevolution`)
Audit date: 2026-09-02
Scope: the six **original** UCIT instruction files (train + `test_3000`) that feed the
V7 six-task formal run. Validation-data construction is Phase D; this document covers
originals only and closes Phase C.

Method: one read-only audit pass over **every record of all 12 files** (213,857 train +
18,000 test records) — schema shape, per-record image existence (realpath under
`image_folder`), answer/placeholder invariants, multiplicity, image+question twin
groups and their answer consistency, sample-id uniqueness, ImageNet-R class histogram,
and train×test overlap on four axes. Raw JSON of this run is kept at
`/tmp/v7_formal_data_audit_raw.json` (audit payload sha256 `0039a43e47ef…28a9`).

---

## 1. File inventory, hashes, metric wiring

`image_folder = /data/dataset/zhaozhuofan/UCIT/datasets`

| Task | V7 idx | split | file | records | bytes | sha256 |
|---|---|---|---|---|---|---|
| ImageNet-R | 0 | train | `instructions/ImageNet-R/train.json` | 23,998 | 10,323,018 | `8aff880c4ebafefd4222439beffa6c89740020da3496a57143fca355fc3b57a4` |
| ImageNet-R | 0 | test | `instructions/ImageNet-R/test_3000.json` | 3,000 | 710,456 | `bfc603ab258d11186e776524e710764a92c05e2779ba60c3f970cccc3ec342ac` |
| ArxivQA | 1 | train | `instructions/ArxivQA/train_4w.json` | 40,000 | 37,750,597 | `d6073a9150608bb3a650d88c346eebb93d2fd5f3fb2bade569b4f64117fa0f98` |
| ArxivQA | 1 | test | `instructions/ArxivQA/test_3000.json` | 3,000 | 2,299,951 | `6a4da4dd77a2d339a41aa02c3b7a2db5292f9e2e3cbf9a1f23e3db9fb8d102c2` |
| VizWiz | 2 | train | `instructions/VizWiz/train.json` | 40,000 | 18,508,713 | `d0b97411bbe6b4b9006bd38d82c652a9d94d5d087a12da861e4ba398b6347078` |
| VizWiz | 2 | test | `instructions/VizWiz/test_3000.json` | 3,000 | 829,885 | `d7db5d9b0e298c24180f9a2b81b4c7ca76d8e935f2bbd6f37868040517ab7bfd` |
| IconQA | 3 | train | `instructions/IconQA/train.json` | 29,859 | 14,097,289 | `65b3df433ec76f44287809cd9554f77ff78106f7962e9211472dbc96d0266bb4` |
| IconQA | 3 | test | `instructions/IconQA/test_3000.json` | 3,000 | 881,447 | `f4ef8b7735606fe5ff61623213945da7ecb95f203aff73a238957e66eac74896` |
| CLEVR | 4 | train | `instructions/CLEVR/train_4w.json` | 40,000 | 17,307,438 | `085cc2ee1a1257502e5b13b698fbc9aa2d05baaabc6f56218f0fb9253eb1ca47` |
| CLEVR | 4 | test | `instructions/CLEVR/test_3000.json` | 3,000 | 762,405 | `463c53f3dc4d6297069b3396cce5609f6dd90173b0a979b9f181fbad4854e41d` |
| Flickr30k | 5 | train | `instructions/Flickr30k/train_brief_4w.json` | 40,000 | 18,305,474 | `38534b7790934ad49b6e868c4798688d44863ffb2a5611e2db1d932744b69756` |
| Flickr30k | 5 | test | `instructions/Flickr30k/test_3000.json` | 3,000 | 899,666 | `e0a244e7456f761404daa9743321e020343e3e752f6e3c699eff051c4ed4ffec` |

Hashing strategy (stated per the audit contract): **annotation/instruction JSON files are
sha256-hashed in full** (above). **Per-image byte-hashing is not performed**; instead
every referenced image is asserted to exist via `realpath` resolution under
`image_folder` with per-file byte sizes recorded (Section 3). Test files are byte-
identical inputs to the official test metrics and are never modified by the V7 chain.

Metric wiring (V7 `task_index` → official UCIT task registry, verified in
`compose/eval/formal_ucit_eval.py`):

| V7 idx | task | official id | scorer module | metric | annotation |
|---|---|---|---|---|---|
| 0 | ImageNet-R | t1 | `llava.eval.eval_deepseek_r1` | Accuracy | instruction file (`answer` exact match, case-insensitive) |
| 1 | ArxivQA | t2 | `llava.eval.eval_deepseek_r1` | Accuracy | instruction file |
| 2 | VizWiz | t3 | `llava.eval.eval_caption` | Average (Bleu1–4/METEOR/ROUGE_L/CIDEr) | `val_coco_type_3000.json` |
| 3 | IconQA | t4 | `llava.eval.eval_deepseek_r1` | Accuracy | instruction file |
| 4 | CLEVR | t5 | `llava.eval.eval_deepseek_r1` | Accuracy | instruction file |
| 5 | Flickr30k | t6 | `llava.eval.eval_caption` | Average | `val_coco_type_3000.json` |

Caption scorers need a working `java` for the COCO PTB tokenizer; `java` (OpenJDK
11.0.29) is installed in the hyper env (`$CONDA_PREFIX/bin/java`), and the six-task
launcher prepends the env `bin` dir to `PATH` (commit `55f7b86`), so `eval_caption`
subprocesses resolve `java` regardless of the invoking shell.

Official caption annotations audited (metrics denominator; 5 references per image):

| file | images | annotations | refs/image | categories | images[] id / file_name |
|---|---|---|---|---|---|
| `instructions/VizWiz/val_coco_type_3000.json` | 3,000 | 15,000 | 5 (all 3,000) | 1 (`captioning`) | 1..N positional / `VizWiz_val_%08d.jpg` |
| `instructions/Flickr30k/val_coco_type_3000.json` | 3,000 | 15,000 | 5 (all 3,000) | none | 1..N positional / `<flickr_id>.jpg` |

Test instruction rows are per-record scoring units (3,000 rows/task); `eval_deepseek_r1`
joins predictions to the annotation on `question_id`, `eval_caption` maps rows to COCO
images by file name.

---

## 2. Record schema audit

**Train files — two schema families, both with `conversations` (no top-level `text`):**

| Task | keys | `conversations` roles | qid/id field | field format |
|---|---|---|---|---|
| ImageNet-R | `conversations`,`id`,`image` | human+gpt, len 2 (23,998/23,998) | `id` (no `question_id`) | `<wnid>/<name>.jpg` image stem |
| ArxivQA | `conversations`,`id`,`image` | human+gpt, len 2 (40,000) | `id` | stringified integers, 40,000 distinct of 40,000 |
| VizWiz | `conversations`,`image`,`question_id` | human+gpt, len 2 (40,000) | `question_id` | 8-digit zero-padded image id, 20,559 distinct of 40,000 |
| IconQA | `conversations`,`image`,`question_id` | human+gpt, len 2 (29,859) | `question_id` | image dir number, 29,859 distinct of 29,859 |
| CLEVR | `conversations`,`id`,`image` | human+gpt, len 2 (40,000) | `id` | integers, 39,387 distinct of 40,000 → **613 duplicate ids** |
| Flickr30k | `conversations`,`image`,`question_id` | human+gpt, len 2 (40,000) | `question_id` | image file id, 21,632 distinct of 40,000 |

Invariants, per record, all six train files: human value starts with the literal
`<image>\n` prefix (anomalous records **0/213,857**); empty answers **0**; answers
containing `<image>` **0**; empty question text **0**. Every train row is a
question→answer pair (roles exactly `{human, gpt}`, conversation length exactly 2).

**Test files — uniform flat schema, all six tasks:** keys exactly
`answer`,`image`,`question_id`,`text` (3,000 rows each). `question_id` unique per file
(duplicates 0 in all six), answers non-empty (0 empty), no placeholder anywhere,
question text non-empty. Test `text` carries no `<image>` placeholder (canonical
`question_text` of these flat records returns the text verbatim).

Duplicate sample ids inside train: ImageNet-R 0, ArxivQA 0, IconQA 0,
VizWiz 19,441, CLEVR 613, Flickr30k 18,368. For VizWiz/Flickr30k/CLEVR the shared ids
are the **image-scoped ids repeated across sibling records of the same image** (VizWiz
and Flickr30k ids are per-image by construction; 613 CLEVR ids repeat across different
question records). Ids are rewritten to `v7_t{task}_{split}_{i}` inside the V7 chain, so
these collisions do not enter the pipeline; the *identity* axes (below) are what the
pipeline audits.

---

## 3. Per-record image existence (12/12 files, 231,857 checks)

Every record of every file was resolved as `realpath(image_folder / image)`.

| Task | split | records | distinct images | missing files | missing dirs | outside image_folder |
|---|---|---|---|---|---|---|
| ImageNet-R | train / test | 23,998 / 3,000 | 23,998 / 3,000 | 0 / 0 | 0 / 0 | 0 / 0 |
| ArxivQA | train / test | 40,000 / 3,000 | 30,534 / 2,949 | 0 / 0 | 0 / 0 | 0 / 0 |
| VizWiz | train / test | 40,000 / 3,000 | 20,559 / 3,000 | 0 / 0 | 0 / 0 | 0 / 0 |
| IconQA | train / test | 29,859 / 3,000 | 29,859 / 3,000 | 0 / 0 | 0 / 0 | 0 / 0 |
| CLEVR | train / test | 40,000 / 3,000 | 30,489 / 2,695 | 0 / 0 | 0 / 0 | 0 / 0 |
| Flickr30k | train / test | 40,000 / 3,000 | 21,632 / 3,000 | 0 / 0 | 0 / 0 | 0 / 0 |

Referenced sub-tree layout (all real): ImageNet-R `{train,test}/<wnid>/`; ArxivQA
`images/`; VizWiz `{train,val}/`; IconQA `iconqa_data/iconqa/{train,val}/<type>/…`;
CLEVR `images/{train,val}/`; Flickr30k `{train,val}/`.

---

## 4. Multiplicity and duplication audit (train)

`q/image` = distinct question strings per image; `a/image` = distinct answers per image.

| Task | images | records/image (k histogram) | q/image | twin groups (image+question, ≥2) | twin group sizes | answer-inconsistent groups | exact duplicate records |
|---|---|---|---|---|---|---|---|
| ImageNet-R | 23,998 | k=1: 23,998 | 1 | 0 | — | 0 | 0 |
| ArxivQA | 30,534 | 1:23,015 2:5,999 3:1,184 4:263 5:62 6:6 7:3 8:2 | 1..6 | 1,739 | 2:1,674 3:64 4:1 | **0** | 0 |
| VizWiz | 20,559 | 1:7,483 2:7,962 3:3,976 4:1,025 5:113 | 1 | 13,076 | 2:7,962 3:3,976 4:1,025 5:113 | 12,390 | **1,331** |
| IconQA | 29,859 | k=1: 29,859 | 1 | 0 | — | 0 | 0 |
| CLEVR | 30,489 | 1:22,613 2:6,453 3:1,236 4:164 5:21 6:2 | 1..6 | 96 | 2:96 | **0** | 0 |
| Flickr30k | 21,632 | 1:8,878 2:8,108 3:3,738 4:848 5:60 | 1 | 12,754 | 2:8,108 3:3,738 4:848 5:60 | 12,754 | **4** |

Interpretation per task family:

- **ImageNet-R / IconQA — strictly one record per image**, no twins, no duplicates.
- **ArxivQA / CLEVR — question-bank tasks**: one image carries up to 8 (ArxivQA) or 6
  (CLEVR) distinct questions. Exact **image+question twin groups** (ArxivQA 1,739 —
  3,544 records; CLEVR 96 — 192 records) are *answer-consistent in every group*
  (inconsistent = 0): the twins are duplicated (image, question) pairs that share the
  same gold answer and differ only in record id. Full-record byte duplicates: 0.
  CLEVR's 613 duplicate top-level ids are distinct records sharing an id string.
- **VizWiz / Flickr30k — caption tasks converted to one shared instruction per image**
  (every image has exactly 1 question string — the caption prompt): sibling records of
  one image are distinct caption references (VizWiz 20,559 images with k=1..5
  caption-answer rows; Flickr30k 21,632 images, k=1..5). Duplicate *caption strings*
  within an image: VizWiz **1,054 images** (1,331 full duplicate records, e.g. the
  repeated `"Quality issues are too severe to recognize visual content"` refusal class);
  Flickr30k **4 images** (e.g. `7510394.jpg` → `"A man is smoking a pipe"` repeated),
  matching exactly the 4 byte-duplicate records.

These are pre-existing dataset artifacts, reported for completeness. Their V7
consequences are neutralized by construction: identity-based leakage groups are never
allowed to straddle formal train and validation (Phase D groups whole images and whole
twin groups), and the V7 chain rewrites sample ids so duplicated ids never collide.

In-test duplication (test_3000 files only): ArxivQA 8 and CLEVR 3 exact
(image, question) twin pairs with distinct `question_id`s, all answer-consistent
(ImageNet-R/VizWiz/IconQA/Flickr30k: 0). Each member of such a pair is scored as an
independent row — a pre-existing test-set artifact that does not touch the V7 chain.

---

## 5. ImageNet-R class distribution (train, 23,998 records)

Class is taken from the record's image path — the second-to-last path component is the
ImageNet synset wnid (`ImageNet-R/train/<wnid>/<file>`), i.e. the class label source is
the **image directory name**, not the text answer.

- **200 classes, all non-empty**, sizes min **41** (`n02088466`, `n02109525`), max
  **334**, mean **119.99**, median **110**.
- Size bins (#classes): 41–60: 23 · 61–90: 51 · 91–120: 42 · 121–160: 43 ·
  161–200: 23 · 201–334: 18 (sum 200).
- Smallest classes: `n02088466` (41), `n02109525` (41), `n02091032` (44),
  `n02112137` (44), `n02128385` (44).
- **Within-class answer uniformity: every one of the 200 classes has exactly 1
  distinct answer string** (`nonuniform` classes = 0), and the file holds exactly 200
  distinct answer labels — the path wnid → gpt label mapping is 1:1. Answer lengths:
  17,527 single-token, 6,244 two-token, remainder ≤ 4 tokens (mean 1.28), all non-empty.

This distribution is the input to the Phase D stratified validation carve (two-stage
largest-remainder over the 200 classes, ≥1 seat/class, seed 42).

---

## 6. Train × test overlap audit (four axes + question strings)

Axes are computed with the pipeline's own hashing code
(`compose.v7.provenance`: `_record_identity` = sha256(image, question_text),
`_normalized_record_hash` = sha256 of the full record JSON; sample id extraction uses
the same `id`-then-`question_id` fallback the run contract uses).

| Task | source-id overlap | image+question | normalized record | image-only | question-string |
|---|---|---|---|---|---|
| ImageNet-R | 0 | 0 | 0 | 0 | 1 |
| ArxivQA | 1,475 | 0 | 0 | **1,258** | 0 |
| VizWiz | 2,646 | 0 | 0 | 0 | 1 |
| IconQA | 0 | 0 | 0 | 0 | 706 |
| CLEVR | 301 | 0 | 0 | 0 | 776 |
| Flickr30k | 0 | 0 | 0 | 0 | 1 |

Semantics and verdicts:

- **image+question overlap and normalized-record overlap are the material-leakage axes**
  (hard failures in `audit_split_isolation`) → **0/0 on all six tasks: PASS**.
- **Image-only overlap is report-only** → non-zero **only for ArxivQA: 1,258 shared
  figure images** between train and test (test rows pair them with different
  questions/answers; same figure reused across splits — pre-existing, matches the
  previously reported 1,258 figure). All other tasks split image namespaces by
  directory (`train/` vs `test|val` trees) → 0.
- **Source-id overlaps (VizWiz 2,646, CLEVR 301, ArxivQA 1,475) are numeric-collision
  artifacts** of independently sourced splits that reuse the same numeric id space
  (VizWiz: zero-padded 8-digit image ids; test samples ids that numerically collide
  with the train id space while pointing at different image files — image overlap is 0
  for VizWiz/CLEVR). Report-only per the provenance contract.
- **Question-string overlap** non-zero for the templated tasks (IconQA 706, CLEVR 776
  shared question templates; 1 for the single-instruction caption tasks and
  ImageNet-R). Text reuse without image reuse is not leakage; the pipeline's identity
  hash (image+question) is the fatal axis.

**Zero test leakage into any pre-commit stage** is separately enforced by
`bind_pipeline_data_usage` (the s0 run contract raises if any of training / key
learning / RMS / pruning sources resolve to the test file).

---

## 7. Gates summary (Phase C)

| gate | result |
|---|---|
| 12/12 files exist, parse, counts match contract (23,998 / 40,000 / 40,000 / 29,859 / 40,000 / 40,000 train) | PASS |
| per-record image existence, 231,857 checks | PASS (missing 0) |
| schema invariants (roles, conv length, placeholder prefix, empty/placeholder answers) | PASS (anomalies 0) |
| ImageNet-R 200 classes non-empty, ≥1 answer/class uniform | PASS |
| train∩test image+question & record overlap | PASS (0/0 all tasks) |
| image-only overlap reported | ArxivQA 1,258 (pre-existing; report-only) |
| test files sha256 stable & untouched | PASS |
| metric wiring + caption annotations + java availability | PASS |
| validation files present | **BLOCKED → Phase D constructs `v7_validation/` + `v7_train/`** |

Phase C closes with all original-file gates green; the remaining data work (validation
carve + formal-train files + config migration) proceeds in Phase D.
