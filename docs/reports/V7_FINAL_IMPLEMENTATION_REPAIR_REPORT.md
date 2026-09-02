# V7 Final Implementation Repair Report

## 1. Git State

- Branch: `exp/v7-full-data-global-key-expert-coevolution`
- Audit baseline: `ee971f75a69c2945a85d380620476d57084b0e3d`
- Validated implementation SHA: `ac7fb8dbaf7631ed5bc017475efea8b112c7e864`
- Worktree at validation: clean
- GitHub push: blocked. Server HTTPS remote has no credential; the local host
  cannot reach `github.com:443`, and server SSH has no authorized GitHub key.
  The complete final branch is preserved in the server repository and in the
  local `v7-final-delivery.bundle`.
- Commits after baseline:
  `1ea87e8`, `78feddf`, `0fa30a0`, `6e7c16d`, `f082390`,
  `0a8f76a`, `10242fb`, `7ea3b26`, `d4c3270`, `ac7fb8d`.

## 2. Executive Verdict

| Gate | Verdict | Evidence |
|---|---|---|
| ALGORITHM_SKELETON | PASS | Fixed 1536-D query, Historical+4 Current global cosine Top-2, no forbidden teacher/router/quota path. |
| FORMAL_FULL_DATA | PASS (code) | Formal mode has no step cap; observed-ID coverage must be 100%. A 2001/2001 audit test passes. No six-task training was started. |
| ANSWER_NLL | PASS | Training dataset/collator supervision mask is reused; shifted answer-token NLL fails on zero supervised tokens. |
| RMS_PARITY | PASS | Persisted historical kappa is applied and frozen; current validation kappa is used for pruning, commit and inference. |
| IMAGE_PREPROCESSING | PASS | `pad/-2/patch/mlp2x_gelu` plus projector path/hash is validated across stages. |
| ITERATIVE_PRUNING | PASS | One candidate is removed per iteration, then the surviving pool is rerouted and rescored. |
| TOP2_POOL_CONTRACT | PASS | Task0 minimum two experts; Task1+ may retain zero current experts when history already has two; commit asserts pool size. |
| SPLIT_ISOLATION | PASS | Real path/SHA/source-ID/image+question/record checks; stage-use booleans are derived from bound source paths. |
| OFFICIAL_VAL_METRIC | PASS (code) | Existing UCIT evaluator is reused; NLL fallback is explicitly labeled. Formal runs require an explicit metric. |
| GRAD_ACCUMULATION | PASS | Autograd hooks record only the current microbatch contribution without clearing accumulated gradients. |
| RESUME | PASS | Key, candidate LoRA payload, optimizer, scheduler and counters are restored; continuous-vs-resume state test passes. |
| ATOMIC_COMMIT | PASS | Complete sibling temporary directory is validated, fsynced and atomically renamed; injected failure leaves no final directory. |
| FORMAL_UCIT_LAUNCHER | PASS (code) | Separate resumable `v7_six_task_run.sh`; old launchers remain unchanged. |
| GPU_SMOKE | RESOURCE_BLOCKED at final HEAD | Every GPU had an external process; GPU2/3 had only 3.5/5.1 GiB free. No process was interrupted. Earlier real-7B evidence is retained separately. |

`FORMAL_EXPERIMENT_READY = NO`

The code correctness gates pass, but the exact formal launcher currently
fails before Task0 because `/data/dataset/zhaozhuofan/UCIT/v7_validation/`
does not exist. Preparing audited, test-disjoint validation instruction files
and the VizWiz/Flickr COCO annotations is mandatory. This is an operational
data blocker, not permission to substitute the test set.

## 3. Issue-by-Issue Repair

| Issue | Previous Behavior / Root Cause | Fix | Main Files | Tests | Status |
|---|---|---|---|---|---|
| Fake full-data | Implicit 30-step cap; no observed dataloader coverage | Uncapped epoch formal mode; explicit smoke override; observed-ID distributed coverage audit | `v7_task_run.py`, `hf_trainer.py`, config | formal/smoke + 2001 IDs | PASS |
| Prompt-contaminated NLL | Labels could clone full prompt input | Reuse `LazySupervisedDataset` and training collator; shared shifted answer mask | `nll_eval.py`, `training.py` | prompt unchanged / answer changed / shared preprocessing | PASS |
| RMS drift | Caller-dependent application and possible history overwrite | Apply persisted calibration by default; merge only current validation kappa; checksum history | `load_compose.py`, `rms.py`, `rms_stats.py` | history/current parity | PASS |
| Preprocessing drift | Stage-specific image processing | Runtime contract with projector SHA; formal `image_aspect_ratio=pad` | eval/train/runner | mismatch fail test | PASS |
| Simultaneous pruning | All removals decided from iteration 0 | Iterative remove-one-and-reroute with deterministic tie-break and trajectory | `pruning.py`, runner | interchangeable candidates | PASS |
| Pool below two | Commit could expose invalid Top-2 pool | Minimum-pool protection and commit assertion | `pruning.py`, `commit.py` | Task0/Task1 cases | PASS |
| Declarative leakage flag | Booleans were not tied to actual stage inputs | File/record provenance plus `bind_pipeline_data_usage` | `provenance.py`, `workflow.py` | path/SHA/rewritten-ID overlap | PASS |
| Validation metric | `-AnswerNLL` served as metric and loss | Reuse accepted UCIT task evaluator; retain answer NLL as auxiliary | `v7_validation_metric.py`, runner | evaluator parity | PASS |
| Accumulation audit | Stale accumulated `.grad` looked like current contribution | Current-graph parameter hooks | `hf_trainer.py` | R1 then R2 microbatches | PASS |
| Query hard-code | Train/test CLIP path embedded in callers | Schema-fixed CLIP-L/14-336 path, explicit CLI, config-file hash provenance | config, `query_features.py`, `eval_task.py` | path/hash parity | PASS |
| Dead pair scale | YAML accepted values runtime ignored | Schema-lock to `1/sqrt(2)` | config, LoRA runtime | invalid 0.8 rejected | PASS |
| Partial resume | Custom state was saved but optimizer/scheduler reconnection was implicit | Restore immediately or after optimizer/scheduler construction; preserve counters | checkpoint, trainer | continuous-vs-resume | PASS |
| Partial commit | Final directory existed before all artifacts | Temporary transaction, validate, fsync, atomic rename | `commit.py` | injected second-save failure | PASS |
| Stale tests / NLL writer | 3-slot fixtures; missing `os` import | Align to established four-slot contract; unit-test writer | tests, `nll_eval.py` | full suite | PASS |

No repair changes V7 method semantics.

## 4. Final V7 Pipeline

```text
Full Train Split
        ↓
Fixed 1536-D Query (schema-fixed CLIP-L/14-336)
        ↓
Full-Train Task Center
        ↓
4 Current Candidate Keys
        ↓
Historical + Current Global Top-2
        ↓
Full-data LoRA + Key Co-Evolution
        ↓
Validation-only RMS
        ↓
Official Metric + Answer NLL
Iterative Remove-and-Reroute
        ↓
Top-2-safe Atomic Commit
        ↓
Committed-only Test Inference
```

## 5. Training Update Contract

| Component | Contract |
|---|---|
| Base / Vision Tower / Projector | Frozen |
| Embedding | Not an optimizer parameter |
| Fixed Query | Parameter-free and detached |
| Historical Key / LoRA / RMS | Frozen, checksum-audited |
| Selected Current Key | Updated by Query-Key attraction |
| Selected Current LoRA | Updated by answer loss |
| Unselected Current Key / LoRA | No new microbatch gradient contribution |
| OldOld | No new expert contribution; prior accumulated gradients are preserved |

## 6. Data Contract

The bounded real-data evidence predates the final Query/atomic/resume hardening
commit but remains valid for split isolation and the model lifecycle.

| Task / split | Path | SHA256 | Count |
|---|---|---|---:|
| ImageNet-R train | `.../source/imagenetr_train32.json` | `3d2f9fe6923e39f62384f69610eae132c70e0463c94d2fc4c623a6e14db99120` | 32 |
| ImageNet-R val | `.../source/imagenetr_val8.json` | `2afb87d6498e4f6e0e970f6b23a35bf33a1429231c70f2ed528b5272b8439fb6` | 8 |
| ImageNet-R test | `.../source/imagenetr_test4.json` | `cea0f48f11deac30254752ba304ebd63101beac2ea633f27a1e7d3d004490ce1` | 4 |
| ArxivQA train | `.../source/arxivqa_train32.json` | `d1ca22bcc9da03416257883199199c29580f6bf498fd543b6266a2f3fe519295` | 32 |
| ArxivQA val | `.../source/arxivqa_val8.json` | `8fc64b92981736730d004b452eb930cd11ab134b863bb2984c2bb1381de5095a` | 8 |
| ArxivQA test | `.../source/arxivqa_test4.json` | `a2bf533a0e0f741613ad10e9abdbac5822e3e654002cecd2773a4b11fac62fb5` | 4 |

All three pairwise source-record and image overlaps are zero for both tasks.
The final pipeline additionally stores source index/ID, question hash, image,
and full source-record hash before rewriting internal IDs. Derived flags are:
`test_used_for_training=false`, `test_used_for_key_learning=false`,
`test_used_for_rms=false`, `test_used_for_pruning=false`.

## 7. Full-Data Evidence

- Formal recipe: 1 epoch, per-device batch 1, accumulation 64, global batch 64,
  LR `2e-4`, warmup `0.03`, cosine, BF16, seed 42, no default `max_steps`.
- Machine test: 2001 declared samples, 2001 unique observed, coverage `1.0`.
- Bounded real 7B run: 32 declared samples, explicit smoke 25 optimizer/micro
  steps; it is intentionally not reported as formal full-data evidence.
- CPU lifecycle smoke: Task0 and Task1 each 30 steps, all finite; resume and
  historical freeze checks passed.

## 8. RMS Contract

- Historical: persisted commit kappa is applied at Task-t load and never
  re-estimated or changed during training/RMS/pruning/inference.
- Current: training uses pre-commit kappa `1.0`; validation creates per-layer
  candidate kappa; pruning uses that kappa; retained kappa is atomically
  committed and loaded by inference.
- Real 7B evidence: 224 injected decoder layers; Task1 retained the Task0
  historical per-layer values and added current retained experts.

## 9. Pruning Contract

The real Task0 trajectory removed candidate 3, rerouted, removed 0, rerouted,
then protected 1 and 2 for strict Top-2. Task1 removed 6, rerouted, then removed
4 and retained 5/7. Because each removal changes the next full/minus score,
interchangeable candidates cannot both be deleted from an iteration-0 result.
The final runner writes `candidate_pruning_trajectory.json` separately.

## 10. Tests

| Command | Result | Duration | Failure classification |
|---|---|---:|---|
| `pytest tests/compose/test_v7_global_coevolution.py -q` | 33 passed | 6.54 s | none |
| `pytest tests/compose -q` | 396 passed, 8 subtests passed | 67.69 s | none |
| `python -m py_compile ...` + launcher `bash -n` + runner `--help` | PASS | bounded | none |
| `python -m compose.experiments.v7_smoke --steps 30` | all six checks true | 5.77 s | none |

Warnings: three upstream deprecation warnings (Transformers pytree and
DeepSpeed `find_executable`); no `NEW_REGRESSION`, `PRE_EXISTING_STALE_TEST`,
`EXTERNAL_DEPENDENCY`, or unclassified failure remains.

## 11. GPU Validation

Earlier bounded real-7B evidence at commit `0a8f76a`:

- Task0 ImageNet-R: 32/8/4 split, 25 finite steps, all four candidates used,
  224-layer validation RMS, iterative pruning, pool `[1,2]`, immediate
  four-sample committed inference succeeded.
- Task1 ArxivQA: 32/8/4 split, 25 finite steps, historical Key/LoRA/RMS
  unchanged, candidates 4-7 used, final pool `[1,2,5,7]`, committed inference
  succeeded. Natural bounded routes were NewNew; deterministic geometry tests
  cover OldOld/OldNew reachability without a quota.

Final-HEAD resource audit:

| Physical GPU | Free MiB | Utilization | Existing compute | Decision |
|---:|---:|---:|---|---|
| 0 | 13313 | 99% | python, 10898 MiB | not used |
| 1 | 3979 | 100% | python, 20232 MiB | not used |
| 2 | 3461 | 90% | python, 20748 MiB | not used |
| 3 | 5087 | 21% | python, 19122 MiB | not used |
| 4 | 9856 | 0% | external python, 13906 MiB | not enough safe 7B capacity |
| 5 | 9924 | 24% | external python, 13906 MiB | not used |
| 6 | 7584 | 34% | external python, 13906 MiB | not used |
| 7 | 9646 | 16% | external python, 13906 MiB | not used |

Chosen GPU: none. Reason: no card satisfied the free-memory/no-external-load
criterion. No process was killed, stopped, or interfered with.

## 12. Remaining Risks

1. Six audited formal validation instruction files and two COCO-format
   validation annotations are absent; the launcher correctly fails closed.
2. Final-HEAD real-7B Query-backbone-hash and atomic-commit changes have unit,
   static and CPU lifecycle coverage, but their new GPU smoke was resource
   blocked.
3. No six-task convergence, benchmark, ablation or multi-seed claim is made.
4. GitHub synchronization remains pending until an authenticated/reachable
   GitHub transport is available; no credential or remote configuration was
   altered during this task.
