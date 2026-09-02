# V7 Formal Pre-Training Audit — Final Report

Date: 2026-09-02 · Branch: `exp/v7-full-data-global-key-expert-coevolution` · Repo: `/home/zhaozhuofan/Hyper-LlaVA`

Companion reports: `docs/reports/V7_FORMAL_DATA_AUDIT.md` (Phase C, per-record original-data audit),
`docs/reports/V7_FORMAL_DATA_DRYRUN.json` (Phase E, CPU dry run).

## 1 · Git state

| | |
|---|---|
| Audit start HEAD | `16b0051` (fix #1 baseline; branch clean) |
| Final HEAD | `53f2469` |
| Fixes landed | `5819da8` (fix #1 query-text) · `1d9fa1b` (fix #2 builder) · `61640b4` (fix #2b preflight coverage) · `5e634c9` (fix #3 config) · `53f2469` (dry-run report) |
| Working tree | clean; all changes committed separately with regression tests |

## 2 · Code gates

All PASS on the final head — verified by focused pytest + subprocess smoke (details §8):

- **V7 core** `v7_global_coevolution`: 4 rank-8 candidate LoRA experts, Global Top-2 cosine routing from step 1 (no warmup/quota/oracle/cluster), key loss λ=0.1 on selected current only.
- **Full-data contract**: formal mode requires `max_steps` absent/-1 and bans `max_samples`; coverage audit asserts full train coverage; smoke coverage artifact shows `full_data_required` toggling correctly.
- **Fixed query**: 1536-D `L2Norm(concat(LN z_v 768, LN z_t 768))`, image+instruction only; center computed from formal train only. Cached-query text extraction canonicalized (fix #1) to placeholder-free `question_text()` — previously the cached train/val/key/pruning space contained literal `<image>` while live routing did not (measured cosine drift ~0.94).
- **Gradient isolation / freeze**: trainable-parameter audit = current keys (6,144 = 4×1536) + current LoRA (79,953,920) only; historical keys/LoRA/RMS frozen + checksum-verified; query parameters 0; optimizer whitelist matches.
- **RMS κ calibration**: validation-only; calibration artifact records `calibration_split: validation`, clip [0.25, 4.0]; rejects test.
- **Pruning**: iterative remove-and-reroute, validation-only, on committed experts; commit refuses < 2 experts; s6 inference uses committed experts with no oracle/task_id/cluster flags.
- **Resume**: stage markers bound to `run_contract_hash` (git SHA + sha256 of config/train/val/test/annotation + sha256-tree of previous checkpoint + recipe); stale/unbound markers raise; formal mode enforces `WORLD_SIZE == 1`.
- **Metric wiring**: 0 ImageNet-R Accuracy · 1 ArxivQA Accuracy · 2 VizWiz Average(eval_caption) · 3 IconQA Accuracy · 4 CLEVR Accuracy · 5 Flickr30k Average(eval_caption); annotation swap to rewritten val file for IC tasks when paths match.

## 3 · Dataset mapping (final formal layout)

All files under `/data/dataset/zhaozhuofan/UCIT/`. Original train files unchanged; formal train = original minus seed-42 256-sample carve; test files untouched (sha256 verified unchanged from Phase C).

| Task | Formal train | Validation | Val annotation | Formal-train count | Removed | Test |
|---|---|---|---|---|---|---|
| ImageNet-R | `v7_train/ImageNet-R/train.json` | `v7_validation/ImageNet-R/validation.json` | validation.json | 23,742 | 256 | `instructions/*/test_3000.json` |
| ArxivQA | `v7_train/ArxivQA/train.json` | `v7_validation/ArxivQA/validation.json` | validation.json | 39,720 | 280 | same |
| VizWiz | `v7_train/VizWiz/train.json` | `v7_validation/VizWiz/validation.json` | validation_coco.json | 39,520 | 480 | same |
| IconQA | `v7_train/IconQA/train.json` | `v7_validation/IconQA/validation.json` | validation.json | 29,603 | 256 | same |
| CLEVR | `v7_train/CLEVR/train.json` | `v7_validation/CLEVR/validation.json` | validation.json | 39,743 | 257 | same |
| Flickr30k | `v7_train/Flickr30k/train.json` | `v7_validation/Flickr30k/validation.json` | validation_coco.json | 38,916 | 1,084 | same |

Validation = 256 samples per task, carved disjointly from each task's **original train** split with deterministic seed 42 (per-task `random.Random(42)`); test never participates. Source sha256s + original record indexes + artifact sha256s in `v7_validation/<Task>/provenance_build.json`. Rebuild is byte-identical (unit-tested). Caption stats: VizWiz 256 images / 256 questions / 460 ref captions; Flickr30k 256 images / 256 questions / 1,084 ref captions (reference-rich-first selection). ImageNet-R: 200/200 classes in validation, every class ≥ 40 members post-carve; class-stratified two-stage largest-remainder (never positional slicing).

## 4 · Split isolation (per built file pair, `audit_split_isolation`)

| Axis | train↔val | train↔test | val↔test |
|---|---|---|---|
| image+question / normalized-record | 0 / 0 (all 6 tasks) | 0 / 0 | 0 / 0 |
| source-id | 0 | 0 | 0 |
| image-only | 0 | ArxivQA 1,258 (pre-existing, reported) | ArxivQA **6** (report-only; inherited from the pre-existing train∩test image sharing — no identity or record overlap) |

Fatal axes (image+question, normalized record) are zero everywhere and enforced with `raise`; the `test_data_used_for_*` usage flags are all false.

## 5 · Data quality

Full per-record audit in `V7_FORMAL_DATA_AUDIT.md`: 12/12 files parsed; zero missing images per-record; zero placeholder-in-answer; answer/role/conv schema uniform; ImageNet-R 200 classes (41–334 per class, 1 answer/class); question-tasks twins answer-consistent (ArxivQA 1,739 content-consistent groups; CLEVR 96); caption sibling answers consistent per image. Existence check is per-record file existence (no per-image byte-hashing — documented choice).

## 6 · Recipe (formal)

1 epoch · per-device batch 1 · grad-accum 64 → **effective global batch 64** · lr 2e-4 · key lr 3e-4 · wd 0 · warmup 0.03 · cosine · bf16 · grad-ckpt · seed 42 · workers 4 · drop_last False · `model_max_length` 2048 · `max_steps` = -1/absent · WORLD_SIZE == 1. Verified in dry-run recipe block.

## 7 · Full-data / supervision / mask contract

Coverage audit on formal train (`v7_full_data_coverage.json` in smoke shows the accounting shape); supervision counts are answer-token based with `--compose_v7_require_full_coverage` toggling `formal_run`; no `train∪val` training — RMS and pruning consume validation only (`calibration_split: validation` observed in smoke), key loss applies on selected current experts only.

## 8 · Tests (fresh, on final head 53f2469)

| Suite | Result |
|---|---|
| `tests/compose/test_v7_global_coevolution.py` | **36 passed** (6.04 s) |
| `tests/compose -k "v7 or rms or routing or pruning or provenance"` | **87 passed** (8.34 s) |
| `tests/compose` (full, launcher PATH with java) | **416 passed + 8 subtests** (70.26 s) |
| `tests/compose/test_build_v7_formal_splits.py` (new) | 17 passed (3.5 s) |

The two caption-scorer parity tests fail without java on PATH (`FileNotFoundError: java`) — reproduced identically at the earlier audit head `5819da8` in a detached worktree, so pre-existing and environmental: the six-task launcher always prepends the conda env bin (`v7_six_task_run.sh` line 17), under which the full suite passes.

## 9 · Query contract (verified)

`query_mode = v7_fixed`; cache provenance bound to CLIP-L/14-336 backbone path + resolved path; feature source `frozen_clip_l14_336`; cached queries [N, 1536]. Fix #1 regression test asserts cached-query text == placeholder-free `question_text(record)`.

## 10 · Expert / router contract (verified)

4 candidate keys + LoRA per task; Global Top-2 from the first step; keys frozen after commit with `v7_keys.pt` state + checksum binding (`previous_checkpoint` sha256-tree chaining); OldOld no-op; commit ≥ 2 experts enforced.

## 11 · RMS contract

Validation-only κ calibration (clip [0.25, 4.0]); calibration sha256 + checkpoint hash recorded; single-GPU fp64 aggregation. Smoke artifact confirms split + expert ids.

## 12 · Pruning contract

Iterative remove-and-reroute on validation only; NLL + (when configured) official metric; audit stores per-iteration selections/metrics; final commit after pruning.

## 13 · Launcher chain

`scripts/Compose/Run_UCIT/v7_six_task_run.sh`: sequential per-task runs with `previous_checkpoint = <task_root>/committed` chaining; auto `--resume` on non-empty task root; per-task round-robin GPU; PATH prepend provides `java` to caption scorers; declared split files validated before Task 0.

## 14 · CPU dry run

`V7_FORMAL_DATA_DRYRUN.json`: **6/6 PASS** — declared files present; on-disk sha256 of every built artifact matches provenance; validation count 256; formal-train counts match source-minus-removed; isolation on built files clean (report-only ArxivQA val↔test image overlap 6); per-record val image existence; query-text canonicality; metric resolution incl. java.

## 15 · GPU audit

`nvidia-smi` at smoke time: GPUs 0/1/3/5/6/7 busy with other users' processes (11–21 GB used); GPU 4 held 13.9 GB memory from another process; **GPU 2 was truly idle** (4 MiB, 0%, no compute process) — no process was killed, no VRAM preempted, nothing waited on.

Bounded real-7B smoke on GPU 2 (2026-09-02, scratch root `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_smoke_audit_20260902`, outside the formal RUN_ROOT; 12 train / 64 val / 2 test records carved from the built formal files; `--smoke-max-steps 3`; NLL-fallback pruning metric):

- **Task0 (ImageNet-R, task 0) full lifecycle s0→s6 PASS** (~30 min): s0 prep + isolation audit → s1 fixed queries (1536-D) → s2 candidates → s3 training 3 optimizer steps (12 samples; trainable-parameter audit = 6,144 key + 79,953,920 LoRA params, query/historical 0; freeze audit on) → s4 RMS κ calibration on validation (clip [0.25, 4.0] observed; dominance ratios 272–368) → s5 pruning: 10 NLL remove-and-reroute iterations on validation, trajectory recorded, final pool `[1, 3]` (2 experts, commit ≥ 2) → committed checkpoint (`compose_experts.bin`, `v7_keys.pt`, sha256-tree `b34a0ef5…`) → s6 inference over 2 test rows from the committed experts (`eval/summary.json`).
- **Task1 (ArxivQA, task 1) on Task0's commit PASS** (~30 min): `previous_checkpoint = <task0>/committed` honored — `previous_checkpoint_hash` equals Task0's `b34a0ef5…`; full lifecycle repeated with 4 historical frozen experts from Task0 (checksum-bound); pruning retained `[1, 3, 5, 6]` (Task0 experts 1,3 + Task1 experts 5,6); committed (`36bda168…`); s6 inference on 2 test rows.
- Logs/diagnostics inspected: coverage math (smoke `full_data_required=false`, 12/12 accounted), RMS `calibration_split: validation`, pruning trajectory decisions per removed candidate.

GPU 2 left free on completion.

## 16 · Blockers

- P0: none.
- P1: none.
- P2: report-only ArxivQA validation↔test image overlap (6 images, no identity/record leakage) — inherent to the pre-existing train∩test image sharing; recorded, never fatal.
- DATA: resolved (Phase D built + verified + config migrated).
- RESOURCE: none at smoke time (GPU 2 idle, smoke completed). Residual: any future run needs a free card; the launcher round-robins GPUs 0–3 which are currently other users' — the formal run must be scheduled when cards free.

## 17 · Verdicts

- **CODE_READY**: YES (PASS, §2/§8)
- **DATA_READY**: YES (PASS, §3–§5, dry-run 6/6)
- **GPU_SMOKE_READY**: YES (PASS, §15 — real-7B lifecycle s0→s6 twice, chained commit verified)
- **FORMAL_EXPERIMENT_READY**: **YES** — six-task training must NOT start without user confirmation.

## Known residual risks

1. Flickr30k / VizWiz validation references = available sibling captions (k ≤ 5; deduplicated), never a fabricated or external 5-reference set — the original Flickr30k annotation is not on this machine; deviation recorded in provenance and reports. Builder keeps an upgrade path if the original annotation becomes available.
2. ArxivQA val↔test image-only overlap (report-only).
3. GPU smoke succeeded on an idle card (GPU 2), but cards rotate among users; the formal six-task run must be scheduled when a card is free (launcher round-robins GPUs 0–3). No process was ever killed or preempted. Smoke was bounded (3 steps, small splits); it validates machinery, not method quality.
