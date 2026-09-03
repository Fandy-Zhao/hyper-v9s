# V7 Query Cache Downstream Adaptation Report (0903 spec Phase B)

Date: 2026-09-03 (evening) · Branch: `feat/0903-v7-throughput-equivalence` ·
HEAD: `0da6019` · Cache: `v7_fixed_query_cache_gpu01_20260903` (230,780 queries +
6 task centers, manifest sha256 `0b66f2520db5822002fe618a21b04c9bdc58cf0bb7bef53c0c07815d37004c7e`,
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

## 8. Gate status (Phase B, GPU-gated — pending idle GPU0/1)

Blocked at report time by other users' jobs (GPU0: `openpi serve_lerobot`,
~8.9 GiB; GPU1: `continual_train`, ~435 MiB); nothing is preempted; a
20-minute poll resumes when both are idle.

| Gate | Status |
| --- | --- |
| QUERY_NUMERICAL_EQUIVALENCE | pending (GPU live vs cache rows) |
| TOP2_ROUTING_EQUIVALENCE (100%) | pending |
| S3_QUERY_ENCODER_CALLS=0 | design-guaranteed + pending run audit |
| single-GPU 7B smoke (loss/gradient matrix) | pending |
| cached-vs-online step compare | pending |
| RMS_CACHE_EQUIVALENCE / DISTRIBUTED_RMS_EQUIVALENCE | design-guaranteed (no queries) + pending |
| PRUNING_TRAJECTORY_EQUIVALENCE | pending |
| EVALUATION_CACHE_EQUIVALENCE | pending |
| EXISTING_V7_REGRESSION | PASS (500/502, 2 env-limited) |
| RECIPE_EXACT | plan-verified; formal launcher asserts it |

## 9. Commits

- `6183884` reader + runtime contract (spec commit 1)
- `af716f0` S1 cache adapter + evaluation reuse + guards (commits 2–6 content)
- `1083eee` Phase A report + producer tooling (docs)
- `0da6019` this report + GPU0/1 cached-query formal launcher
- formal run records `FORMAL_START_SHA` before Task0 and per-task resumes use
  marker/sha discipline unchanged.

## 10. Conclusion lines

- DOWNSTREAM_CACHE_ADAPTATION=YES
- FORMAL_TRAINING_READY=NO (GPU0/1 availability gate pending: foreign jobs
  present at report time; nothing is ever preempted)
- FORMAL_TRAINING_STARTED=NO
