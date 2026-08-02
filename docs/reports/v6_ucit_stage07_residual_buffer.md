# Hyper-LLaVA V6 Stage 07 — Old-Expert Sufficiency and Residual Buffer

Status: **PASSED**. The technical closed loop is complete; classifier quality is limited and is reported without claiming method effectiveness. Git: this report is contained by the atomic Stage 07 commit; source parent and buffer provenance commit `a8a9b8a53f9ce02b552e82ab55fc5792045984e1`.

## Frozen definition and provenance

- Teacher source: historical-only Oracle-Direct on train/validation only. `old_expert_sufficient=true` iff the selected historical set is nonempty, its penalized score is below Empty, and raw answer-token mean-NLL gain is at least `0.02`.
- Predicted head inputs are limited to Query, cardinality logits, Top-M similarity, similarity margin, masked similarity entropy, predicted-set score, and visible-expert count. Answer NLL, target, teacher set, task ID, post-task expert, and test metrics are not inputs.
- Threshold candidates were frozen as `0.30/0.50/0.70`; selection minimizes validation false-sufficient rate, then maximizes F1, then chooses the lower threshold. The selected threshold is `0.70`.
- Config-file SHA256: `ad99920470f181f31a0b018b1e79794c6d8b4678701694ff6c9958b9c8e25a9e`; embedded head config hash: `a934b6f328147dd9ada33fdd08325fe806d3405a1a0db78c69c69e38da64fab3`.
- Dataset manifest hash: `2982f01bc8f8d29a3a6ab648286d1ee399f76319c3031649add99db4dd5af3cf`; Oracle cache hash: `779af3046f12b6293d94a329ee837216598a94dbb5f51e0f6fb5874c6f328d5f`.
- Sufficiency checkpoint SHA256: `3d6f0f0a4190e3ba007e52d277ee1f7fd507c244c778f69b684dbc60e2dbb4d2`.
- GPU: preferred 0–3 were occupied by another user's 10–11 GiB processes; actual feature export/training used 4–5, never more than two GPUs. Environment: `{"cuda":"11.8","peft":"0.4.0","python":"3.10.20","pytorch":"2.3.1+cu118","transformers":"4.33.3"}`.

## Technical validation

- Full Compose suite: `146 passed, 8 subtests passed`; the seven specified Stage 07 test files separately pass (`12 passed`).
- Teacher definition, no-answer head signature, train-only enforcement, atomic shard round-trip, strict resume-or-reject, exact DDP merge, dedup/conflict handling, checkpoint round-trip, and teacher/predicted temporal leakage rejection all pass.
- Exported features: 768 train + 192 validation; `answer_features_used=false`, `task_id_lookup_used=false`, `test_data_used=false`, temporal violations `0`.
- Buffer audit: unique IDs equal record counts, every split is train, every teacher and predicted expert predates the record task, teacher buffer rows are teacher-insufficient, and predicted buffer rows fall below frozen threshold.
- A real second build resumed both existing shards after deterministic field-by-field comparison (creation timestamp and the later invocation's Git commit excluded), preserved the original provenance and checksums, and atomically refreshed the sampling manifest and metrics.

## Sufficiency validation

Formal six-task validation metrics at threshold `0.70`:

| Metric | Value |
|---|---:|
| Accuracy | 0.697917 |
| AUROC | 0.831461 |
| AUPRC | 0.734426 |
| Precision | 0.792453 |
| Recall | 0.471910 |
| F1 | 0.591549 |
| False sufficient rate | 0.106796 |
| False insufficient rate | 0.528090 |
| ECE | 0.040785 |
| Teacher insufficient rate | 0.536458 |
| Predicted insufficient rate | 0.723958 |

Threshold sensitivity (`0.30/0.50/0.70`) gives false-sufficient rates `0.592233/0.242718/0.106796` and false-insufficient rates `0.022472/0.202247/0.528090`. The frozen conservative rule therefore trades substantial false-insufficient behavior for lower false-sufficient risk.

ImageNet-R smoke passed (all 32 validation rows teacher-insufficient and predicted-insufficient). Mini2 and Mini3 both predict all rows insufficient; their AUROC values are `0.782051` and `0.759972`, but thresholded sufficient recall is zero. The full model improves global separation but remains poorly calibrated by task: validation false-sufficient rate reaches `1.0` on CLEVR, while sufficient recall is zero on ArxivQA and VizWiz.

## Residual Buffers

- Teacher buffer: `427` rows; SHA256 `7b5afbe0412fe261c4d909fd51816e37de40361142ec5449c7bb3cc760df007c`.
- Predicted buffer: `561` rows; SHA256 `2b272db375a5c612c2b72a3c34ef6471b3d5fc800407a8b711da349f5bbcad55`.
- Overlap: `381`; union: `607`; Jaccard: `0.627677`; teacher-only: `46`; predicted-only: `180`.
- Sampling manifest SHA256: `32fb60ae392a7895f8915641c7539085facb90d3bd2c97e02e698a1d5fb0ac92`. Every task has only 128 train rows, so the 512/task cap does not discard a candidate; retained/discarded IDs are still persisted.

| Task | Teacher insufficient | Predicted insufficient | Mean residual gain |
|---|---:|---:|---:|
| ImageNet-R | 1.000000 | 1.000000 | 0.000000 |
| ArxivQA | 0.671875 | 1.000000 | 0.112965 |
| VizWiz | 0.734375 | 1.000000 | 0.022026 |
| IconQA | 0.398438 | 0.851562 | 0.184439 |
| CLEVR | 0.265625 | 0.000000 | 0.152104 |
| Flickr30k | 0.265625 | 0.531250 | 0.099070 |

For predicted Empty rows, teacher/predicted insufficiency are both `1.0` (128 rows). For predicted Single rows they are `0.467188/0.676563` (640 rows); no Pair was predicted. Router-set mismatches have teacher insufficiency `0.715026`, versus `0.395288` when the predicted and teacher sets match.

Teacher buffer answer types are 79 single-token, 204 short-phrase, and 144 long-answer; predicted buffer counts are 120, 229, and 212. Prompt-hash strata cover 123 teacher and 147 predicted buckets; no image is copied into a buffer.

## Failures, retries, and limitations

- The initial Stage 07 experiment-directory upload failed because its parent did not yet exist; the exact parent was created and the same reviewed config/manifest/scripts were uploaded successfully. Formal execution itself completed on the first run.
- The head is conservative overall but highly task-dependent. A larger predicted buffer than teacher buffer indicates over-conservative insufficiency prediction; the nontrivial false-sufficient rate still leaves missed-residual risk.
- Task 0 has no historical expert by construction, so all ImageNet-R rows are residuals. No predicted Pair rows exist, so Stage 07 cannot analyze Pair-specific sufficiency empirically.
- Teacher and predicted buffers are intentionally separate. Stage 08, if later authorized, must use the teacher buffer as the primary training buffer; this stage does not train or instantiate a Candidate Expert.

Machine-readable metrics and preregistration live under `experiments/runs/v6_ucit_staged/stage07_residual_buffer/`. Full feature bundles, checkpoints, shards, and sampling manifests remain under `/data/ckpt/zhaozhuofan/v6_ucit_staged/stage07_residual_buffer/` and are not committed.
