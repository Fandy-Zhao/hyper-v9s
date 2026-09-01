# Hyper-LLaVA V7 current-code audit

- Audit baseline: `a8f3a7860631aec8e2ea0d65ad9794838ffaffc7`
- Source branch: `codex/task0-capacity-chain`
- V7 branch: `exp/v7-full-data-global-key-expert-coevolution`
- Worktree at audit: clean after the two pre-existing One-shot reports were saved in the baseline commit.

## Current pipeline

The formal V6.2 training entry is `python -m compose.experiments.task_run`. It
implements the S0--S12 state machine in `compose/experiments/task_run.py` and
launches `compose.train.train_compose` for LoRA optimization. The UCIT launch
scripts are under `scripts/Compose/Run_UCIT/`.

The current expert data is split between:

- `compose.experts.ExpertRegistry`: durable expert identity, lifecycle,
  checkpoint/key/RMS provenance and monotonic pool version;
- `compose.experts.ExpertPool` / `compose.adapters.ExpertManager`: runtime LoRA
  modules and train/freeze roles;
- `compose.router.ExpertKeyStore` and router checkpoints: expert keys and
  creation-task visibility;
- `compose.experiments.snapshot.ComposeSnapshot`: task-boundary registry,
  router, RMS and pool-checkpoint references.

## Query, routing and updates before V7

`compose/router/functional_query.py::ComposeQueryEncoder` currently applies
two affine `LayerNorm` modules and a `Linear -> GELU -> Linear` projection to
produce a 128-dimensional normalized query. It is initialized once and frozen,
but it is still a parameterized MLP. `compose/eval/query_features.py` extracts
CLIP visual/text features and caches this query.

V6.2 updates cluster-expert LoRA weights in
`compose/train/train_compose.py` (`cluster_expert` mode). Historical experts are
loaded then frozen through `ExpertPool.train_only`. Keys are initialized from
cluster centroids and optimized separately in
`compose/router/key_learning.py::learn_cluster_keys`; contribution refinement
and the set-router are then trained in S7 of `task_run.py`.

`compose/adapters/lora.py::ComposeLinear.forward` already performs true
per-sample routing: for each expert it index-selects only rows that selected the
expert, computes the expert delta on that sub-batch, and index-adds it back.
This provides selected-only LoRA gradients without forcing one pair for the
whole batch. It also applies per-layer RMS kappa and the fixed pair scale
`1/sqrt(2)`. `ComposeTrainer` keeps the selection context alive during gradient
checkpoint recomputation.

Task 0 is an empty-registry cold start in V6.2, but it still follows base-NLL,
residual, bootstrap-cluster, cluster-LoRA and key-learning stages rather than
global Top-2 over four candidates.

## V6/V6.1/V6.2 modules V7 must bypass

- teacher/oracle search in S2 (`compose.teacher`, answer-NLL candidate search);
- residual threshold/split in S3 (`compose.expansion.residual`);
- residual query clustering in S5 (`compose.expansion.query_clustering`);
- cluster-supervised key learning and contribution refinement in S7;
- `ExpertSetRouter`, query/router MLPs, cardinality and pair scorers;
- direct commit of every cluster expert in S8;
- any global/post-hoc key calibration, shadow update, hyperbolic/set/pair-key
  path, candidate warm-up or learned pair weights.

The current runner explicitly truncates expert-discovery data with
`teacher_train = records_all[:teacher_search_train_samples]` (default 2000).
V7 must not call that path: its task center and training dataset must use the
entire declared train split. Validation must be a distinct declared split and
test answers must never enter routing, training or pruning.

## Reusable components

- `ComposeLinear` per-sample sparse execution and RMS-calibrated dual-LoRA
  composition;
- adapter injection, `ExpertManager`, adapter-only checkpoint serialization;
- expert registry metadata and atomic transaction/checkpoint helpers;
- CLIP visual/text feature extraction (raw 768+768 features only);
- standard teacher-forcing answer-token NLL in `ComposeLlavaForCausalLM`;
- UCIT official evaluators and sharded execution utilities;
- task snapshots, RNG helpers and JSON/JSONL reporting conventions.

## Required V7 additions

1. Parameter-free fixed 1536-D multimodal query and full-split feature cache.
2. Four rank-8 current candidates initialized around the full-task center.
3. One selectable historical+current key pool with global per-sample Top-2
   from step one; only current keys/LoRA are optimizer parameters.
4. A V7 trainer that adds selected-current key attraction, handles Old+Old
   no-gradient micro-batches, audits gradients/checksums and persists metrics.
5. Usage plus remove-and-reroute contribution and redundancy pruning, followed
   by atomic commit of only retained candidates.
6. V7 checkpoint/resume state (keys, candidates, optimizer/scheduler/RNG/RMS,
   counters, step/config/pool version) with explicit refreezing after load.
7. Inference restricted to committed frozen experts and the same fixed query,
   cosine global Top-2 and RMS-calibrated fixed dual-LoRA composition.
8. A single explicit `method: v7_global_coevolution` configuration and task
   runner; V6.x files remain available and are not deleted.

