# Migration Map: Fixed Candidate Slots -> Query-Clustered Residual Expert Discovery

Baseline commit: `3c2a7d1ded478e0d7faaa4964097e8170bc9cb3c`

## Audit findings (old pipeline)

The old v6 UCIT pipeline was: fixed candidate slots (2 per task, ids `(t+1)*10 + {0,1}`),
K-means++/random-orthogonal key competition, per-slot assignment via argmax cosine,
candidate training (`train_v6_candidate.py`), contribution validation
(`support_count >= tau_support`, `mean_gain >= tau_gain`, `key_accuracy >= tau_key`),
two-phase commit with rejected-candidate fallback, global router calibration
(`v6_calibrate.py` BCE distillation), and a three-runner family
(`v6_task1_dry_run`, `v6_task2_dry_run`, `v6_task_run`).

Two concrete bugs found during audit:

1. **Batch-0 Top-M bug**: `V6Router.retrieve` (compose/router/v6_router.py:205)
   used `top_indices[0]` — the Top-M of batch sample 0 — as the candidate set for
   the whole batch.
2. **RMS reference bug**: `rms_report` (compose/lora/v6_rms.py:213) used
   `reference = raw` (each expert's own RMS as its reference), making kappa
   identically ~1.0, and kappa never entered `ComposeLinear.forward` (report-only).

## Migration map

| Old file / class / function | New Compose file / class / function | Action |
|---|---|---|
| `router/v6_router.py` `V6QueryEncoder` | `router/query_encoder.py` `ComposeQueryEncoder` | Rewrite (GELU MLP + deterministic init) |
| `router/v6_router.py` `V6Router` | `router/router.py` `ComposeRouter` | Rewrite (per-sample Top-M, cosine-only) |
| `router/v6_router.py` `V6RetrievalResult` | `router/router.py` `ComposeRetrievalResult` | Rewrite (batch x M ids) |
| `router/v6_router.py` `V6SelectionResult` | `router/router.py` `ComposeRouterSelection` | Rewrite |
| `router/v6_router.py` save/load_v6_router_checkpoint | `router/router.py` save/load_compose_router_checkpoint | Rewrite, kind="compose_router" |
| `router/v6_calibrate.py` (global calibration) | — | **Delete** (global calibration forbidden) |
| `expansion/v6_residual.py` `is_residual` (gain-floor) | `expansion/residual.py` `is_residual` (tau_res) | Rewrite |
| `expansion/v6_residual.py` `V6ResidualRecord` | `expansion/residual.py` `ComposeResidualRecord` | Rewrite |
| `expansion/v6_residual.py` `build_residual_records` | `expansion/residual.py` `build_residual_records` | Rewrite |
| `expansion/v6_candidate.py` `V6CandidateConfig` / slots | — | **Delete** |
| `expansion/candidate_pool.py` `CandidateExpertPool` | — (kmeans++ init absorbed into `query_clustering.py`) | **Delete** |
| `expansion/v6_commit.py` validation gates | `expansion/commit.py` `commit_cluster_experts` | Rewrite (direct commit) |
| `expansion/v6_rejected.py` | — | **Delete** |
| `expansion/v6_base_pool.py` | — (empty pool is legal cold start) | **Delete** |
| `teacher/v6_teacher.py` `V6TeacherSearcher` | `teacher/teacher.py` `ComposeTeacherSearcher` | Rewrite (per-sample Top-M based) |
| `teacher/v6_teacher.py` `V6TeacherRecord` | `teacher/teacher.py` `ComposeTeacherRecord` | Rewrite |
| — | `expansion/query_clustering.py` (spherical K-Means + silhouette) | New |
| — | `expansion/expert_formation.py` (cluster -> expert manifest) | New |
| — | `router/key_learning.py` (cluster-supervised key loss) | New |
| `lora/v6_rms.py` `compute_v6_expert_rms` | `lora/rms.py` `compute_expert_rms` | Rewrite (multi-expert reference mean) |
| `lora/v6_rms.py` `rms_report` (reference=raw bug) | `lora/rms.py` `rms_report` (reference = mean over active experts) | Rewrite |
| `train/train_v6_candidate.py` | merged into `train/train_compose.py` (--compose-mode) | **Delete**, merge |
| `train/trainer.py` `inputs["v6_selections"]` | `inputs["compose_selections"]` | Modify |
| `train/data.py` `SelectionTaggedDataset` (in v6 file) | `train/data.py` `ComposeSelectionDataset` | Rewrite |
| `train/data.py` `SelectionCollator` (in v6 file) | `train/data.py` `ComposeSelectionCollator` | Rewrite |
| `experiments/v6_task_run.py` / `v6_task1_dry_run.py` / `v6_task2_dry_run.py` | `experiments/task_run.py` | Rewrite (one runner, task 0..5) |
| `experiments/v6_snapshot.py` `V6Snapshot` | `experiments/snapshot.py` `ComposeSnapshot` | Rewrite (saves query encoder + keys + rms) |
| `experts/task_state.py` v6 stages | `experts/task_state.py` Compose stages | Rewrite |
| `experts/registry.py` lifecycle-active view | `experts/registry.py` flag-based active view + next_expert_id | Modify |
| `experts/transaction.py` provisional commit | `experts/transaction.py` direct active commit | Modify |
| `eval/v6_nll_eval.py` | `eval/nll_eval.py` | Rename |
| `eval/v6_query_features.py` (768-D image only) | `eval/query_features.py` (visual + text + 128-D query) | Rewrite |
| `eval/v6_assemble_expert.py` | `eval/assemble_expert.py` | Rename |
| `eval/v6_acceptance.py` | `eval/acceptance.py` | Rewrite (method-level stats added) |
| `eval/v6_multiseed.py` | — | Delete (aggregation trivial) |
| `cli/eval_v6_routed_ucit.py` | `eval/routed_ucit.py` | Rewrite |
| `configs/v6_ucit_*.yaml` | `configs/compose_ucit.yaml` | Rewrite |
| `scripts/v6_ucit/*` | `scripts/Compose/Run_UCIT/*` | Rewrite |
| `adapters/lora.py` (no calibration) | `adapters/lora.py` `set_expert_calibration` / pair scale | Modify |
| `experts/checkpoint.py` | `experts/checkpoint.py` (calibration persisted in manifest) | Modify |
| tests `test_v6_*.py` | `tests/compose/test_compose_*.py` | Rewrite per new semantics |

## Keep unchanged (engineering foundation)

`adapters/{inject,manager,runtime,types}.py`, `lora/statistics.py`,
`lora/rms_composition.py`, `experts/metadata.py`, `experts/pool.py`,
`teacher/{scorer,oracle_set,candidate_search,types}.py`, `eval/load_compose.py`,
`train/arguments.py`, `experiments/scheduler.py`, `model/`, `data/records.py`.
