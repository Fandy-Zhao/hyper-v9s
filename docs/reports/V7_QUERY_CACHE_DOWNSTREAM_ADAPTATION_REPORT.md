# V7 Query Cache Downstream Adaptation Report (0903 spec Phase B)

Date: 2026-09-03 (late) · Branch: `feat/0903-v7-throughput-equivalence` ·
HEAD: `3cfc2dd` (DDP smoke lifecycle closed 10:47; DISTRIBUTED_RMS_
EQUIVALENCE PASS via the world-2 same-checkpoint recompute 10:57; batch-32
live twin running on GPU0 since 10:58 — see §9) · Cache:
`v7_fixed_query_cache_gpu01_20260903` (230,780
queries + 6 task centers, manifest sha256
`0b66f2520db5822002fe618a21b04c9bdc58cf0bb7bef53c0c07815d37004c7e`,
producer git `c93f51e`).

## 1. Scope and invariants

The adaptation replaces the *source* of every V7 fixed query q_i =
L2Norm(concat(LN(z_visual), LN(z_text))) (1536-D fp32, detached) with the
precomputed cache, keyed by sample_id, and changes GPU scheduling to the
physical GPU0/GPU1 pair.  The V7 method itself is untouched: Fixed-Query
definition and hash (`v7_fixed_layernorm_concat_l2_v1`), Global Top-2
routing, M=4 candidates, LoRA rank 8, frozen historical keys/LoRAs,
selected-only current updates, Answer+Key loss, RMS, iterative
remove-and-reroute pruning, atomic commit, committed-only inference,
Task0→Task5 order, LR/loss/seed/epochs.

| Stage | Before (legacy) | After (cache mode, `--query-cache-manifest`) |
| --- | --- | --- |
| S1 fixed queries | live CLIP per record on GPU workers | streamed from binary split cache, `encoder_calls: 0` |
| S2 candidates / S3 training | consume S1 payload rows | unchanged code; payloads are cache-derived |
| S4 RMS | activation RMS (no queries) | unchanged (no query consumption by construction) |
| S5 pruning | orchestrator routes payload rows | unchanged code; payload rows cache-derived |
| S6 inference / 21-cell eval | live CLIP per question + committed keys | cache test rows → committed-pool Top-2 manifest → `eval_task --selection-manifest` |

## 2. Data flow and consumer map (verified by audit, not by trust)

- Cache binary splits (`queries.pt` + `metadata.json`) hold the *final* q rows.
- S1 emits a byte-schema-compatible payload (`records[sid] = {"query": ...}`),
  identical envelope (`schema_version: 1`, `feature_source:
  frozen_clip_l14_336`, `query_mode: v7_fixed`, encoder provenance/hash) plus
  a `query_origin` block (`v7_fixed_query_cache_derived`, manifest sha256,
  split contract hash, `encoder_calls: 0`).
- Every downstream consumer was grepped and proven to read only `query` rows:
  `workflow.validate_query_cache_contract`, `queries_from_cache` (sorted-id
  rows), `V7QueryDataset` (per-id map, id-first).  No V7 consumer reads
  `visual_feature`/`text_feature`/`feature_hash`; those are deliberately
  omitted (not stored in the cache).
- RMS (`rms_stats.py`, `expert_id=None` forward hooks) consumes no query,
  routing, or features file — RMS reuse is inherited trivially.
- Pruning consumes the S1 payload rows through the orchestrator and routes
  with the same `GlobalTop2Router`/`route_manifest` code.
- Test-time evaluation now precomputes committed-only selections from cache
  **test** rows through the identical `V7InferenceRouter(router)` call the
  live path used, on the same worker GPU (`CUDA_VISIBLE_DEVICES` + `cuda:0`),
  giving bit-identical routing to the legacy live eval (same keys, same fp32
  rows, same matmul order).

## 3. Binding contract: content-bound, git recorded (decision record)

The cache was produced at git `c93f51e`; §42 adaptation commits move HEAD, so
strict git equality would (correctly) fail closed after every commit.  The
runtime S1 binding therefore validates **content** — live backbone content
hash (config/tokenizer files of the CLIP dir), fixed-query implementation
hash, schema/dim/dtype, declared counts, and the per-split
manifest↔metadata contract-hash chain — and **records** the producer vs
runtime git pair (`producer_git_sha`/`runtime_git_sha`) in the emission audit
for the formal reports.  A mismatch of any content field raises before
anything is written; git drift alone is reported, never silent (unit-tested:
`test_emit_records_git_drift_but_binds_content`,
`test_emit_fails_closed_on_backbone_mismatch`).

## 4. Fail-closed guarantees

- sample_id is the primary key: the declared split's id *sequence* must
  equal the cache split's sequence exactly (missing/extra/reordered →
  `ValueError`, nothing written).  This holds for S1 payload emission and
  for evaluation selection manifests (question file vs cache test split).
- `V7QuerySplitReader.get`/`get_batch` raise `QueryCacheMissError` on any
  unknown sample_id; the formal recipe forbids silent recomputation.
- S3/S4/S5 blocks call `_assert_s1_payload_origin(root)` in cache mode:
  `data/query_cache_binding.json` + `metrics/query_encoder_calls.json` must
  exist and the payload files must carry `query_origin` +
  `v7_fixed_query_cache_derived` in the first 8 KiB, else `ValueError`.
- 21-cell evaluation: 2 distinct GPUs minimum in cache mode; the legacy
  exactly-3-GPU rule stays for live mode (no behavior drift).
- Test-split content binding (`--backbone-path`) checks every task's test
  split contract before any routing; formal test files
  (`instructions/<Task>/test_3000.json`) were sha256-verified identical to
  the cache test-split sources.

## 5. Scheduling changes (nothing else)

- Launcher `scripts/Compose/Run_UCIT/v7_gpu01_cached_query_formal.sh` pins
  physical GPU0/1 with an idle availability gate (fail closed, nothing is
  ever stolen; GPU2–7 untouched).  Strict recipe world 2 × batch 1 ×
  accumulation 32 = global batch exactly 64 (`RECIPE_EXACT: YES`), the same
  deterministic `V7GPUPlan` machinery already proven byte-identical.
- `FORMAL_START_SHA` is recorded before Task0 (§42).

## 6. Files

New: `compose/v7/cache_to_s1.py`, `compose/v7/cached_selections.py`,
`tests/compose/test_cache_to_s1.py`, `tests/compose/test_cached_selections.py`,
`scripts/Compose/Run_UCIT/v7_gpu01_cached_query_formal.sh`.
Modified: `compose/experiments/v7_task_run.py` (S1 cache branch + guards +
S6 manifest mode), `compose/eval/v7_formal_ucit_eval.py` (cache mode).
Untouched: all of `compose/v7/query.py|routing.py|pool.py|inference.py|
workflow.py|pruning.py`, `compose/train/data.py` (S3 consumption point
already payload-driven), `compose/eval/rms_stats.py`, `eval_task.py` manifest
branch (pre-existing), configs, model/recipe parameters, legacy launchers.

## 7. Evidence so far

- Real task0 cache→S1 dry run: 23,742 train + 256 val rows emitted in 45.9 s
  wall from the binary cache (sequence match, schema match, 18-split content
  binding green, producer `c93f51e` vs runtime git recorded).
- CPU tests: `test_query_cache_audit.py` 40, `test_cache_to_s1.py` 7,
  `test_cached_selections.py` 7; v7 cluster (`test_v7_global_coevolution.py`
  47, `test_v7_adaptive_equivalence.py`, `test_candidate_pool.py`) all green;
  full `tests/compose`: 500 passed, 2 failures = pre-existing environmental
  (`java` absent for COCO caption scorers, documented on HEAD).
- Formal precheck dry-run: 18/18 splits content-bound to the live CLIP dir.

## 8. Gate status (Phase B)

Updated 2026-09-03 (evening continuation, HEAD `3cfc2dd`).  Phase A
re-audit + GPU gates are now evidenced:

| Gate | Status | Evidence |
| --- | --- | --- |
| Phase A re-audit (18 splits + 6 centers) | PASS | `compose/eval/v7_cache_precheck.py`; `artifacts/v7_query_cache/reverification_20260903_080136.json` (independent recompute from live declared files: declared==saved==unique, miss/unk=0, hashes rehash-match, source/backbone/impl content binding; centers recomputed bit-identical) |
| QUERY_NUMERICAL_EQUIVALENCE | PASS | GPU1 original + GPU0 re-run at final HEAD `ea816a3` (`v7_gpu01_cache_live_gate_20260903{,_rerun_head}`): exact_bit_equal, max_abs_diff 0.0, cosine_min 0.99999976, task0/1.train × 128 |
| TOP2_ROUTING_EQUIVALENCE (100%) | PASS | same evidence: agreement rate 1.0, agreement_exact, max_abs_score_diff 0.0 |
| S3_QUERY_ENCODER_CALLS=0 | PASS (run audit) | fixed smoke `metrics/query_encoder_calls.json`: encoder_calls=0, 23,742 train + 256 val cache-derived, sequence matches |
| single-GPU 7B smoke (loss/gradient matrix) | PASS | cache-mode task0 smoke `v7_gpu01_cache_smoke_task0_fixed_20260903` (GPU1, resumed 08:29:10 after external SIGTERM): full S0–S5 lifecycle closed — `s5_pruning_commit.done` 08:40:45, `committed/` 08:40:44 (v7_keys.pt pool selectable (0,1,2,3), compose_experts), pruning job chain complete (job_0 07:43:03 / job_1 08:40:44 / job_2 08:00:17 / job_3 08:11:55 / job_4 08:23:26), 2 training steps finite (train_runtime 204.96 s), resume stdout `resume_20260903.log` clean exit |
| cached-vs-online full-root compare | FAIL — matched batch size exposed shard-topology remainder (§8b) | batch-32 live twin completed S0–S5; comparator v4: S3/RMS/PRUNING PASS, but S1 FAIL on exactly 32/23,742 train rows (cosine_min 0.999995916, max_abs_diff 4.8937e-4) and COMMIT metadata differs by one selection count. Cache production was two interleaved 11,871-row shards (each final batch 31); live control was one contiguous 23,742-row stream (final batch 30). A matched-topology control is required. |
| RMS_CACHE_EQUIVALENCE | PASS | twin-run audit, semantic comparator v2: all 3 RMS files numeric-equal (the only structure diff was the per-run `output_dir` machine-path leaf, now root-relatively compared); rms_calibration/rms_statistics byte-identical sha |
| TWO_GPU_DDP_CACHE_TRAIN_SMOKE | PASS | DDP cache smoke `v7_gpu01_ddp_cache_smoke_task0_20260903` (GPU0+1, world 2 × batch 1 × GA 32 = global 64, 2 steps, started 09:34): lifecycle closed 10:47 — s3/s4/s5 markers, `committed/` (compose_experts + v7_keys.pt), 9 pruning jobs, rank health/per-rank checksums/coverage audits green, clean exit |
| DISTRIBUTED_RMS_EQUIVALENCE | PASS | world-2 RMS recompute of the single-GPU root's *identical* model (`v7_gpu01_rms_recompute_task0_20260903`, checkpoint_hash `6380bb4d…` == recorded on both sides) vs the recorded single-process RMS; comparator `--gate-mode distributed-rms`, `compare_audit_v3_distributed_rms.json`: rms_calibration 900 leaves + rms_summary 10 leaves bit-identical (only `output_dir` leaf root-relocated), rms_statistics 13,441 leaves max_rel 5.9e-8; execution mode/world_size + calibration_sha256 exempted informational.  (Full-root DDP-vs-single value comparison is not well-posed: the length-grouped sampler consumes disjoint S3 windows per world size → legitimately different weights → the same-checkpoint recompute isolates the world-size contrast — decision recorded in PROJECT_STATE) |
| PRUNING_TRAJECTORY_EQUIVALENCE | PASS | comparator v4 on the batch-32 pair: jobs 0..4 all have identical selections, NLL leaves and official metric leaves; no candidate-removal divergence |
| EVALUATION_CACHE_EQUIVALENCE | PASS | `v7_gpu01_eval_gate_20260903/` (GPU1, HEAD `dab7d38`): `cached_selections_audit.json` — full ImageNet-R test split (3,000 rows) manifest from committed v7_keys.pt, sequence_matches_cache, encoder_calls=0, healthy 6-pair routing histogram; `gate_task0_test.json` — 128 bounded test rows live vs cache: QUERY_NUMERICAL_EQUIVALENCE PASS (exact_bit_equal, max_abs_diff 0.0, cosine_min 0.999999642 ≥ 1−1e−6), TOP2_ROUTING_EQUIVALENCE PASS through the committed pool (rate 1.0, exact, 0 disagreements, max_abs_score_diff 0.0, visible experts [0,1,2,3]) |
| EXISTING_V7_REGRESSION | PASS (535/537, 2 env-limited `java`-less caption scorers, re-run at HEAD `9037aa0`; 500 → 535 = 35 new tests from the Phase B tooling) |
| RECIPE_EXACT | PASS (plan-verified: `V7GPUPlan.build([0,1], strict)` → world 2 × batch 1 × GA 32 = global batch 64; formal launcher asserts it) |

## 8a. Twin-run divergence localization (0903 spec §26)

The first full-root cache-vs-live comparison (single-GPU cache smoke root
vs the live twin root, both complete task0 lifecycles under the same
recipe) FAILed five gates.  Three were comparator defects (fixed in the
semantic comparator v2 — see §9a): S1 compared file order instead of
per-sample_id rows, RMS structure walked machine-path leaves such as
`output_dir`, and the commit gate demanded byte-sha equality.  The two
remaining failure classes are real and are localized here, evidence by
evidence (§26: localized, not ignored).

Observed divergences (cache root = A, live twin root = B, 23,742 train +
256 val rows):

| Stage | Divergence | Magnitude |
| --- | --- | --- |
| S1 query rows | per-id differences on **all 23,742 rows**, essentially confined to the visual half (768-D) | max 1.87e-3, mean row-max 2.3e-4; text half only a uniform ~6.6e-5 shared-normalize artifact |
| S2 task center | center (mean of the 23,742 rows) differs | max 7.61e-5, mean 1.0e-5 |
| S2 candidate keys | all four keys carry the *same* center offset (max diffs 7.614/7.614/7.614/7.610e-5) | 7.6e-5 |
| S3 training (2 steps) | `answer_loss` per step 0.0; LoRA/expert compose binary byte-identical; RMS files numeric-equal | per_sample_key_loss 35/128 rows at 1e-4–3.9e-4 |
| S5 pruning | Top-2 flips on 2/256 val rows (170, 195); committed json `rerouted_expert_ids` flip on val row 107 | NLL diffs ≤0.0174, official metric ≤1.7e-3 |
| Commit | committed keys ≤1.16e-4 (within the 2e-4 bound); compose_experts.bin byte-identical | keys 7.6e-5–1.2e-4 |

Localization (each step pinned by a controlled comparison, not inferred):

1. **S2 is deterministic given identical input.**  The DDP cache smoke
   (third training, S2 already complete) consumes the *same* cache binary
   rows as the single-GPU cache smoke; its `candidate_keys.pt` is
   **4/4 bit-equal** to the single-GPU root's, while it differs from the
   live twin at exactly the single-GPU-vs-twin magnitude (7.614e-5).  The
   center/key computation (CPU fp32 mean + seeded CPU RNG + fp32
   normalize) is therefore fully reproducible; it is *not* a cross-run
   noise source.
2. **The live twin's S1 rows really differ from the cache rows.**  Cache
   production and the live side of the bounded gate both encode with the
   fp16 CLIP backbone at **batch 32** (gate CLI default = 32, cache split
   contracts record batch 32) and are bit-exact (`QUERY_NUMERICAL_
   EQUIVALENCE` PASS, 128×4 rows, both GPUs).  The twin's S1 ran the
   legacy live encoder `compose.eval.query_features` **without
   `--batch-size` → default 16** (verified in its `features_train.log`
   COMMAND line).  fp16 CLIP conv kernels are chosen by cuDNN per input
   batch shape, so batch-16 and batch-32 outputs differ at fp16 rounding
   scale — matching the visual-half-only, per-row 1e-4..2e-3 pattern (the
   text tower has no convs, reduces per token, and stays bit-equal apart
   from the shared L2-normalize scaling — exactly the observed ~6.6e-5
   uniform text-half artifact).
3. **Propagation.**  The 1e-3-class S1 differences shift the S2 center by
   ~7.6e-5; all four keys share that offset; S3 answer losses/LoRA stay
   identical (no training-row route flip) while per-sample key losses
   move at 1e-4; pruning's 256-row validation then contains 2–3 rows
   whose Top-2 margin is smaller than the 7.6e-5 key offset, flipping
   them (val 170, 195, and a reroute on 107).  Every downstream difference
   is traceable to those rows; nothing else moved.
4. **Attribution.**  This is a control-experiment setup defect, not a
   method defect: the §19-21 live twin did not pin the live encoder's
   batch size to the cache-production value (32).  The cache itself is
   bit-stable (DDP-vs-single S2 proof above; gate bit-exactness on both
   GPUs), and the formal recipe never invokes a live encoder
   (`encoder_calls: 0`), so the formal run is untouched by this class.
5. **Remediation pending.**  After the DDP smoke frees GPU0/GPU1, the
   live twin is re-run with `--batch-size 32` for the §19-21 full-root
   comparison; the named gates (QUERY_NUMERICAL/TOP2/TASK_CENTER/RMS/
   PRUNING/EVALUATION) are re-issued against that pair.  Until then the
   S1/S3/PRUNING/COMMIT verdicts on the batch-16 twin pair stay FAIL
   (correct per-gate truth, with the localization above as the §26
   explanation).

## 8b. Batch-32 full-root re-issue (comparator v4)

The batch-32 live twin completed cleanly at 12:14 UTC. Comparator v4
(`compare_audit_v4_batch32_pair.json`) closes S3, RMS, and all five pruning
jobs as PASS, but correctly leaves the overall gate FAIL. Of 23,742 train
queries, 23,710 are bit-identical and exactly 32 differ; validation is fully
bit-identical. The cache producer used two `index % 2` shards of 11,871 rows,
so each shard ended with a 31-row CLIP batch. The live twin used one contiguous
stream, whose last batch had 30 rows. The 32 differing rows are precisely the
rows whose full/partial-batch topology changed. Their cosine minimum is
0.999995916 and max absolute difference is 4.89369035e-4, below the strict
query gate. S3 loss/gradient comparison and the complete pruning trajectory
remain within tolerance/identical, but committed validation metadata records
one different selection (`11847` vs `11848`); committed keys themselves pass
with max difference 2.72878e-7. This is not waived. A new online control must
mirror the producer's two interleaved shards and deterministic merge.

## 9. Commits

- `6183884` reader + runtime contract (spec commit 1)
- `af716f0` S1 cache adapter + evaluation reuse + guards (commits 2–6 content)
- `1083eee` Phase A report + producer tooling (docs)
- `0da6019` this report + GPU0/1 cached-query formal launcher
- `40bc226` comparator semantic fixes (S1 per-sample_id alignment, path-leaf
  relocation, semantic commit gate with measured noise envelope) + §8a
  twin-run divergence localization (batch16-vs-batch32 fp16 encoder);
  RMS_CACHE_EQUIVALENCE PASS on the real twin pair (§26/§28)
- `9037aa0` comparator `--gate-mode distributed-rms` (same-checkpoint
  world-2 recompute gate; execution-context leaves exempted informational,
  checkpoint_hash hard) + `--query-features-batch-size` passthrough on all
  three live-encoder sites (§8a remediation tooling)
- `3cfc2dd` launch-command records (docs)
- formal run records `FORMAL_START_SHA` before Task0 and per-task resumes use
  marker/sha discipline unchanged.

## 10. Conclusion lines

- DOWNSTREAM_CACHE_ADAPTATION=YES
- FULL_SAMPLE_QUERY_COVERAGE=YES (independent Phase A re-audit, 230,780 rows)
- QUERY_CONTRACT_MATCH=YES (content-bound; producer `c93f51e` vs runtime
  `ea816a3` git drift recorded by design)
- QUERY_NUMERICAL_EQUIVALENCE=PASS · TOP2_ROUTING_EQUIVALENCE=PASS
  (bit-exact; re-anchored at the formal HEAD on GPU0 + original GPU1)
- EXISTING_V7_REGRESSION=PASS (535/537 at `9037aa0`, 2 `java`-less
  env-limited)
- RECIPE_EXACT=YES (world 2 × batch 1 × GA 32 = global batch 64)
- RMS_CACHE_EQUIVALENCE=PASS (real cache-vs-online twin pair, semantic
  comparator v2: 3 RMS files numeric-equal, calibration/statistics
  byte-identical; output_dir path leaf relocated by design)
- DISTRIBUTED_RMS_EQUIVALENCE=PASS (world-2 RMS recompute of the identical
  checkpoint `6380bb4d…` vs recorded single-process RMS: calibration/
  summary bit-identical, statistics max_rel 5.9e-8;
  `compare_audit_v3_distributed_rms.json`)
- TWO_GPU_DDP_CACHE_TRAIN_SMOKE=PASS (world 2 × batch 1 × GA 32 = 64, 2
  steps, full lifecycle closed 10:47 with committed/)
- Batch-32 full-root comparator v4: S3_TRAIN_STEPS_EQUIVALENCE=PASS,
  RMS_CACHE_EQUIVALENCE=PASS, PRUNING_TRAJECTORY_EQUIVALENCE=PASS;
  S1_QUERY_ROWS_EQUIVALENCE=FAIL (32/23,742 train rows differ because the
  producer's two interleaved partial batches do not match the live control's
  one contiguous partial batch) and COMMIT_STATE_EQUIVALENCE=FAIL (one
  selection_count differs; committed keys remain within tolerance).
- FORMAL_TRAINING_READY=NO. The strict full-root gate remains closed pending
  a matched-topology online query control; no tolerance was relaxed and formal
  Task0 was not started.
- FORMAL_TRAINING_STARTED=NO
