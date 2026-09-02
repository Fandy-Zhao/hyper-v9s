# Hyper-LLaVA V7 implementation report

## 1. Git and scope

- Baseline commit: `a8f3a7860631aec8e2ea0d65ad9794838ffaffc7`
- Branch: `exp/v7-full-data-global-key-expert-coevolution`
- Method: `v7_global_coevolution`
- Implementation head before this report: `2ce11cd922f4d3b305507a767f002db10698d4e2`
- Historical V6/V6.1/V6.2 and One-shot code was retained. V7 is an explicit
  new mode and runner, not a replacement demo.

## 2. Changed files and responsibilities

| File | Responsibility |
| --- | --- |
| `CURRENT_CODE_AUDIT.md` | Baseline entry points, data structures, reusable and bypassed paths. |
| `configs/v7_global_coevolution.yaml` | Single V7 method plus query/candidate/routing/training/pruning config. |
| `compose/v7/query.py` | Parameter-free 768+768 fixed LayerNorm/concat/L2 query and full-center coverage assertion. |
| `compose/v7/pool.py` | 1536-D Key lifecycle, four reproducible center-perturbed candidates, normalization and checksums. |
| `compose/v7/routing.py` | Historical+current deterministic cosine Global Top-2 and route diagnostics. |
| `compose/v7/training.py` | Answer-token NLL helper, selected-current Key loss, no-op and gradient audits. |
| `compose/v7/hf_trainer.py` | Dynamic routing from step one, optimizer whitelist, adapter/key checkpointing and JSONL diagnostics. |
| `compose/v7/pruning.py` | Train/val usage, true remove-and-reroute validation, redundancy and configurable decisions. |
| `compose/v7/commit.py` | Filter pruned LoRA/Key, bind RMS and validation metadata, freeze retained candidates. |
| `compose/v7/checkpoint.py` | Atomic task/candidate/optimizer/scheduler/RNG/RMS/counter/config resume state. |
| `compose/v7/inference.py` | Committed-only, task-free fixed-query Global Top-2 inference. |
| `compose/v7/workflow.py` | Full split ID/cache alignment and candidate preparation helpers. |
| `compose/experiments/v7_task_run.py` | Resume-safe full-data stages: data, query, candidates, train, RMS, prune/commit, inference. |
| `compose/experiments/v7_smoke.py` | 30-step Task0 + 30-step Task1 CPU sparse-LoRA smoke. |
| `compose/train/{arguments,data,train_compose}.py` | V7 CLI/config, exact query-cache join and formal trainer dispatch. |
| `compose/eval/query_features.py` | Optional `v7_fixed` raw 1536-D query extraction. |
| `compose/eval/eval_task.py` | `--v7-key-state` inference and cross-task pair logging. |
| `compose/adapters/manager.py` | Clear stale gradients when an expert becomes frozen. |
| `tests/compose/test_v7_global_coevolution.py` | Tests 1--17 from the acceptance contract. |
| `tests/compose/test_grouped_execution.py` | Align one stale fixture with the existing four-slot Compose contract. |

## 3. V7 runtime contract

Fixed Query is implemented in `compose/v7/query.py:26`. Its actual shape is
`[B, 1536]`; it has zero parameters, returns a detached tensor, and is stable
across task stages. Real-data preparation confirmed train/val query dimensions
of 1536 and `num_train_samples == num_queries_used_for_center` (8 == 8 in the
bounded smoke). The formal runner never uses `records[:2000]`, teacher subsets,
residual subsets or NLL-filter subsets for Expert Training.

Candidate initialization is in `compose/v7/pool.py:22`: four rank-8 Candidate
IDs use `normalize(task_center + reproducible tangential perturbation)`. There
is no KMeans, warm-up, quota or forced expert. Pairwise cosine and coverage are
persisted.

Global Top-2 is in `compose/v7/routing.py:34`. All non-pruned historical and
current keys are in one cosine matrix from step one. Task0 is the same code with
an empty historical set. Ties use a deterministic ID-order epsilon, not an
old/new preference.

Per-sample LoRA routing reuses `ComposeLinear.forward`: it index-selects only
the rows that selected each expert and index-adds their deltas. It is therefore
stronger than batch-wide route-signature grouping and does not leak a selected
Candidate to other samples. Existing per-layer RMS kappa and fixed `1/sqrt(2)`
pair composition remain unchanged.

Answer Loss is the standard teacher-forcing target-token mean NLL. The V7
integration receives the LLaVA answer loss in `compose/v7/hf_trainer.py:163`;
the explicit audit helper is in `compose/v7/training.py`. It updates only
selected current LoRA modules because base/history are frozen and sparse row
execution omits unselected modules.

Key Loss is in `compose/v7/training.py:37`. It detaches fixed queries, computes
each sample's mean `1-cos(q,key)` only over selected current candidates, assigns
zero to Old+Old samples, then averages over the batch. Historical and
unselected current Keys receive no attraction.

The optimizer is a strict whitelist of current Candidate LoRA plus current
Keys. Query/base/vision/projector/embedding and historical parameters cannot
enter it. Old+Old-only micro-batches receive a zero-valued current-Key anchor so
backward and the optimizer step are safe no-ops. Historical Key/LoRA gradients
and start/end checksums are asserted; stale gradients are cleared on freeze.

## 4. Task-end lifecycle

`compose/v7/pruning.py:39` first records train/val usage, then evaluates the
full validation Global Top-2 route. For every Candidate it removes that
Candidate from the selectable pool and re-runs Global Top-2 for every sample;
the saved route matrix proves replacement experts fill the vacated slots. It
records metric gain and token-average NLL gain. The default uniform metric is
explicitly labeled `negative_token_average_nll_proxy` when an official
task-specific metric is not configured.

Usage is diagnostic only. A Candidate is never deleted solely for low usage.
Redundancy requires both configured high Key cosine and near-zero removal
contribution; the lower-contribution Candidate is pruned without merging.
Thresholds are config fields; `candidate_prune_enabled: false` retains all four.
Zero through four retained Candidates is legal.

`compose/v7/commit.py:19` writes an inference checkpoint containing historical
experts plus retained Candidates only. It persists normalized learned Keys,
LoRA tensors, per-layer RMS kappa, origin task, usage/removal/redundancy data,
sets lifecycle to historical/formal, freezes Key and LoRA, and advances pool
version. Pruned candidates remain only as unselectable audit metadata.

## 5. Task0, Task1+, resume and inference

- Task0: full declared train split -> fixed queries/task center -> four current
  Candidates -> Global Top-2 among four from step one -> selected Key/LoRA
  learning -> val pruning -> commit 0--4.
- Task1+: load frozen committed checkpoint -> initialize four current
  Candidates -> one historical+current pool -> Global Top-2 -> OldOld safe
  no-op, OldNew selected-new update, NewNew two-new update -> pruning/commit.
- Resume: `compose/v7/checkpoint.py:35` stores task/step, registry, committed and
  current Keys/LoRA, lifecycle, optimizer/scheduler, RNG, RMS, counters, config
  and pool version. `load_v7_checkpoint` explicitly refreezes history.
- Inference: `compose/v7/inference.py:12` rejects current candidates and accepts
  no task ID. `eval_task --v7-key-state` computes fixed CLIP visual/text Query,
  global cosine Top-2 over committed Keys, applies frozen RMS dual-LoRA, and
  records cross-task pairs. No answer/oracle/clustering/test statistics enter.

## 6. V6/V6.1/V6.2 bypass audit

The V7 runner never imports or calls the teacher/oracle search, residual NLL
threshold, residual selector, query clustering, cluster-supervised Key learner,
Set Router/Cardinality/Pair MLP, contribution/global Key calibration, shadow
update, hyperbolic/set/pair key, candidate warm-up, load-balancing router or
pair-weight trainer. Those modules remain intact for historical experiments.

## 7. Tests and smoke validation

### Unit tests

- V7 acceptance suite: **17 passed**.
- V7 plus data/grouped-execution related regression: **24 passed**, 2 dependency
  warnings.
- Earlier related batch including RMS: **27 passed**, 2 dependency warnings.
- Post-GPU-fix focused V7/data/grouped regression: **26 passed**, 3 dependency
  warnings.
- Full Compose audit run: **353 passed, 27 failed, 8 subtests passed**. Of the
  failures, 25 are pre-existing stale tests that construct/expect the former
  three-slot selection while the baseline production contract is four slots;
  2 caption evaluator tests lack the external `java` executable. V7's new
  tests and changed runtime paths are green. One grouped fixture used in the
  V7 regression set was corrected; unrelated stale fixtures were not mass
  rewritten.

### CPU Task0/Task1 smoke

- Task0: 30 optimization steps / 90 routed samples; NewNewRate 1.0; Key and
  LoRA gradients observed; finite losses; four Candidate selection counts were
  42/48/41/49 (not collapsed); checkpoint reload refroze history.
- Task1: 30 steps / 90 samples; OldOldRate 0.3556, NewNewRate 0.6444; eight
  OldOld no-op steps; current Key and LoRA gradients observed; historical Key
  and LoRA checksums unchanged. A quota-free independent geometry probe
  confirmed OldNew is reachable; the natural synthetic batch did not force it.
- All six smoke checks passed: finite Task0, Task0 gradients, Task0 resume,
  Task1 gradients, route logic and historical freeze.

### Real ImageNet-R preparation smoke

On physical GPU 7, the resume-safe runner processed disjoint declared 8-train
and 4-validation records using the real frozen CLIP model. It produced 1536-D
queries, exact 8/8 center coverage, four distinct 1536-D Candidate Keys and
reproducible pairwise cosines near (but below) one. Artifacts are under:

`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_smoke_seed42/task0_realprep`

### Real 7B GPU2/GPU3 lifecycle smoke

A bounded two-task run completed the complete S0--S5 lifecycle with the real
LLaVA-1.5-7B base model. Task0 ran on physical GPU2 over 8 ImageNet-R training
and 4 disjoint validation samples. Its two optimization steps had finite total
losses 3.8550 and 4.6711, nonzero selected-current Key/LoRA gradient norms,
224/224 RMS layers, and five validation NLL passes (full plus four true
remove-and-reroute evaluations). Candidate 2 was retained and committed.

Task1 resumed that checkpoint on physical GPU3 over 8 ArxivQA training and 4
disjoint validation samples. All 448 historical adapter tensors loaded. Its two
optimization steps had finite total losses 1.6593 and 1.1516 with nonzero
selected-current Key/LoRA gradient norms. Historical Key and LoRA checksums
were unchanged. RMS again covered 224/224 layers; pruning performed the full
plus four removal-reroute evaluations and retained Candidates 4, 5 and 7. The
final committed pool contains frozen experts 2, 4, 5 and 7.

The run uncovered and fixed two pre-optimizer boundary defects: V7 Top-2 routes
are now padded to the production four-slot execution representation, and BF16
adapter checksums hash dtype/shape/raw bytes without NumPy BF16 conversion.
Artifacts are under:

`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_gpu23_validation_seed42`

This two-step run validates execution, gradients, historical freezing, RMS,
pruning, commit and resume. It is not evidence of task convergence or final
benchmark quality. Natural Task1 samples selected NewNew routes only; the
quota-free CPU geometry probe remains the evidence that OldNew is reachable.

## 8. Known risks / TODO

1. Real 7B execution has completed only as a bounded two-step lifecycle smoke.
   Full declared-data training, task convergence and official benchmark metrics
   remain future experiment work.
2. The runner's default validation performance signal is a documented
   negative-NLL proxy. A task-specific official evaluator can replace it when
   a labeled validation evaluator/annotation is declared; test answers remain
   prohibited.
3. Task0 is allowed to retain fewer than two experts. In that legal but
   degenerate result, strict Global Top-2 inference cannot run until a later
   task brings the committed pool to at least two; the router fails clearly
   rather than inventing a duplicate/fallback expert.
