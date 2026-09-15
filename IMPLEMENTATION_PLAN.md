# V9-S — Answer-Guided Responsibility Distillation

Branch: `exp/v9-answer-guided-key-expert-coevolution` (base `798268d`, `main` untouched).

This is the *simplified* V9: the same closed loop, with the layers that made the
first draft ambiguous removed rather than tuned. What was cut, and why, is in
§6 — it is the most useful part of this document, because the removed mechanisms
are the ones a later reader is most likely to re-introduce by accident.

## 0. Audit summary (what already exists and is reused verbatim)

| V9-S need | Existing V7/V8 implementation | Reuse |
| --- | --- | --- |
| One backbone + N LoRA branches | `compose/adapters/lora.py::ComposeLinear.forward` — `base_layer(x)` once, then per-expert `index_select → LoRAExpert → gate·kappa·scale → index_add_` | **verbatim** |
| Multi-key max aggregation | `compose/v8/routing.py::MultiKeyRouter.score_matrix` (`scatter_reduce_ amax` then distinct-expert Top-K) | **verbatim** (inference) |
| Fixed query | `q = L2Norm(concat(LayerNorm(z_vis), LayerNorm(z_txt)))`, 768+768=1536, `V7QueryConfig` | **verbatim** |
| Query cache I/O | `V7QueryDataset` / `V7QueryCollator` | **verbatim** |
| Frozen-parameter enforcement | `ExpertManager.train_only`, `freeze_historical`, `assert_historical_lora_frozen`, `adapter_checksums`, `historical_checksums`, `assert_task_freeze_integrity` | **verbatim** |
| DDP | `mean_sync_accumulated_gradients`, `attach_v7_ddp_key_anchor`, `_enable_non_reentrant_checkpointing`, rank0-only save | **verbatim** |
| per-sample answer NLL | `llava_llama.per_sample_token_mean_nll`, surfaced on `self.v7_per_sample_answer_nll` | **verbatim** |
| RMS kappa | `compose/lora/rms.py` (`kappa_k_l = clip(R̄_l / R_k_l, 0.25, 4.0)`) + `merge_commit_frozen_calibration` | **verbatim** |
| Task-end prune/commit | `compose/v7/commit.py`, `CandidatePruner` | **adapted** |
| Checkpoint contract | `compose_experts.bin` / `compose_experts.json` / `v7_keys.pt` | **kept identical** |

Two additive deltas are required from the V8 core, both defaulting to the V8
behaviour so that `main` is bit-identical:

1. `compose/adapters/types.py` — `MAX_ACTIVE_EXPERTS = 4` is a module constant
   asserted in `ComposeSelection.__post_init__`. V9-S needs `C + M` slots
   (4 + 2 = 6, or 8 + 2 = 10 on a wide step). Add an instance field
   `max_slots: int = MAX_ACTIVE_EXPERTS`; every V8 call site keeps 4.
2. `compose/adapters/lora.py` — cardinality scaling (`1/sqrt(N)`, pair `1/√2`) is
   V8's variance rule and contradicts `h ← base + Σ_k a_k·κ_k·u_k`. Add
   `cardinality_scale: "v8" | "none"`, default `"v8"`. V9-S selects `"none"`.

## 1. Hardware reality (reported, not worked around)

`nvidia-smi -L` → **one** `RTX 4090` (48 GB). Physical multi-GPU verification is
impossible on this machine. The real training entry runs under
`compose.experiments.local_ranks --nproc-per-node 2` with both ranks pinned to
GPU 0 over gloo, which exercises every distributed code path that matters (DDP
gradient all-reduce, `find_unused_parameters`, `all_gather_object` audits,
rank0-only checkpointing, non-elastic resume, pool-consistency assertions). It is
a genuine two-rank DDP run and **not** a multi-*device* one; every artifact says
which, and the final report repeats it.

## 2. Closed loop

```
Key decides where to learn ──► Expert learns what to do ──► Answer judges whether it helps
        ▲                                                              │
        └──────── L_key (BCE) ◄── responsibility ◄── gate gradient ◄───┘
```

Per optimizer step: **one** backbone forward carries every offered expert, one
answer NLL comes out, then

```
p      = sigmoid(cos(q, eff_key)/τ − b)        # differentiable w.r.t. keys
a      = _gate(key.detach(), b.detach())       # same value, keys outside it
a.requires_grad_(True)                         # a leaf: an input, not a constant
G_ik   = −a_ik · ∂L_ans/∂a_ik                  # stop-gradient, create_graph=False
r_ik   = G⁺_ik / (Σ_j G⁺_ij + ε)               # answer-derived responsibility
L_key  = BCE(p, r)                             # the only Key supervision
L_total= L_ans + λ_key·L_key + λ_sparse·L_sparse
```

then **one** backward on `L_total`. No second-order graph, no per-expert backbone
enumeration, no pair enumeration.

### The detach contract

The composition consumes `a`, which is built from **detached inputs** and then
marked a leaf with `requires_grad_(True)`. That single move does both jobs at
once: `∂L_ans/∂a ≠ 0` — the answer is differentiable w.r.t. what was deployed —
while `∂a/∂key = 0`, so `L_ans` reaches **no** key parameter and a key's only
supervision is `L_key`, a detached teacher.

The alternative reading — `a = p.detach()` — is wrong, and was the first draft's
bug. Autograd reports no gradient *for a constant*, so detaching the gate makes
`∂L_ans/∂a ≡ 0` and the responsibility teacher identically zero: the keys then
never leave their k-means initialisation, while every logged quantity reads a
plausible `0.0`. "Detach the gate" and "make the gate an input" are opposite
operations and only the second preserves a measurable contribution.
`contribution.gate_gradient` now raises on both disconnections rather than
returning that zero.

`p` (the live copy) is what `L_key` and `L_sparse` train; `a` is what the
composition and the contribution gradient use. The straight-through form
`hard − p.detach() + p` collapses to `hard` under this contract, which is why the
final stage *is* the deployed Top-2 rule rather than a surrogate that resembles
it.

### Multi-key functional expert

`1 Expert : {base origin key (frozen)} ∪ {task_alias keys (absolute, per-task,
independent)}`. There is no `normalize(base + γ·delta)` anywhere in the main
path: a task key is an absolute vector, and moving it cannot drag the identity
the expert was committed with.

### Periodic wide retrieval

The door the fixed exploration expert used to hold open. On a fraction of steps
(`wide_retrieval.ratio`) the recall is widened from `top_c` to `wide_retrieval.top_c`,
under a **seeded per-step Bernoulli draw** so every rank derives the row width
from `(seed, step)` with no collective. The cache is built once at
`max(top_c, wide_top_c)`; the narrow step masks the tail columns to a zero gate.

## 3. Module tree (`compose/v9/`)

```
config.py       V9Config + 15 sections — no magic numbers elsewhere.  The
                responsibility / inference / exact_oracle sections are
                declarative: their __post_init__ refuses any value that would
                make the run a different method.
keys.py         V9KeyPool(MultiKeyExpertPool): base origin keys (frozen) +
                absolute task_alias keys (per task, independent) + candidate
                keys + per-expert routing bias.  Holds the legacy_v9_only
                migration for checkpoints written by the V9 v1 draft.
retrieval.py    per-task cached historical Top-C at max(top_c, wide_top_c),
                cached once per task, deterministic across ranks, saved beside
                the query cache; is_wide_step(step, ratio, seed)
contribution.py G = -a·dL/da (stop-grad), r = G_pos/(ΣG_pos+ε),
                contribution_statistics, calibration_report (G_grad vs G_exact),
                pair_rerank_report (the §35 ablation only)
losses.py       L_key (masked BCE), L_sparse (mean Σa), L_budget (ablation only)
schedule.py     Bootstrap / Soft / ST-Hard Top-2 stage schedule + temperature ramp
router.py       independent per-expert sigmoid gate, temperature annealing,
                deployed_gates() = the Top-K rule inference serves, V9Router
                -> dense [B, S] ComposeSelection
multi_key.py    global multi-key inference aggregation: expert_score(i,k) =
                max over the memory keys of K_k of cos(q_i, e)
inference.py    query-only deployment routing (no answer, no task id),
                purity-checked by assert_v9_inference_purity
audit.py        task-end historical task-key audit + candidate commit statistics
checkpoint.py   atomic task-level V9-S save/resume + RNG capture/restore
data.py         V9QueryDataset / V9QueryCollator: the fixed-query cache path
trainer.py      V9ComposeTrainer(ComposeTrainer): one-forward closed loop, single
                backward, global stats, §34 diagnostics, V9-S checkpoint payload
```

`compose/experiments/v9_task_run.py` orchestrates one task end to end
(query cache → pool → retrieval → training → audit → commit);
`configs/v9s_main.yaml` is the recipe and `configs/v9s_preflight.yaml` the
reduced-length preflight.

## 4. Parameter audit

Trainable — current-candidate LoRA (`← L_ans`), current-candidate Key
(`← L_key` only), current-candidate routing bias, and the current task's own
temporary new key on each historical expert (`← L_key` only).
Frozen — vision tower, projector, LLM backbone, embeddings, fixed query,
historical LoRA, historical base keys, earlier tasks' task keys, RMS kappas.
Supervision is one-directional: **`L_ans` never reaches a key**, not directly and
not through a gate.

## 5. Execution order

audit ✅ → clean branch ✅ → plan ✅ → config ✅ → V8 Multi-Key semantics ✅ →
residual decomposition removed ✅ → answer gate detached from the key graph ✅ →
responsibility as the only key supervision ✅ → wide retrieval replaces the fixed
exploration expert ✅ → M = 2 default ✅ → regularization simplified ✅ →
`legacy_v9_only` schema migration ✅ → 64 static checks ✅ → **one** 2-rank
end-to-end preflight (10 bootstrap / 25 soft / 15 ST steps + the §30 calibration
in the same run) — which found the detach bug and the masked `L_sparse` below,
both fixed and pinned by the two regression tests it names → re-run of the *same*
preflight → full Task0 → Task0 eval/audit/commit → Task1…Task5 →
lower-triangular matrix → final report.

### What the first preflight found

It is worth recording, because it is the argument for the run existing at all: a
50-step training loop completed cleanly, wrote its checkpoints, passed its
isolation audit, and would have gone on to a full Task 0 producing a table of
numbers in which **no key was ever supervised**. Every logged quantity was
`0.0`, and `0.0` is what a healthy run looks like too.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `gate_grad_abs_mean: 0.0`, `contribution_mean: 0.0`, `responsibility_*: 0.0`, `loss_key: 0.0` | composition consumed `p.detach()`; the answer loss was a function of no gate tensor, and `gate_gradient` returned a clean zero for it | forward gate built from detached inputs and marked a leaf; `gate_gradient` raises instead of returning that zero |
| `loss_sparse: 0.0` and `loss_total == loss_answer` | the sparse/budget terms were masked by `contribution.valid`, which is all-`False` exactly while the routing is undecided | row-level regularisers are defined on the routing row; the mask no longer reaches them |

## 6. Removed from the V9 v1 draft (do not re-introduce)

Each of these was a second, disagreeing definition of something the method
already defined once. Kept only as declared ablations where noted.

| Removed | Why |
| --- | --- |
| base key + task **residual** key decomposition, `γ` mixing | two parameters for one routing direction; the base key must stay frozen and the task key must be free, and a sum ties them |
| direct `∂L_ans/∂key` through the gate | a second answer-side gradient on the same parameter, unsupervised by the responsibility. The config refuses `direct_answer_gradient_to_key: true` |
| fixed Exploration Expert (a reserved slot on every row) | paid on every step to answer a question that arises on few; replaced by periodic wide retrieval at one cache column of persistent cost |
| default M = 4 | four experts competing for one answer signal, no distinct decision added; **M = 2** is the recipe, M = 4 the ablation |
| `L_sparse` **and** `L_budget` on by default | two objectives on one quantity; `L_sparse` keeps, budget loss is off (`use_budget_loss: false`) and restores as the ablation |
| task-conditioned inference | a run that needs the task id is answering "which task is this?" rather than "which expert does this sample need"; `inference.task_id` refuses `true` |
| pair-aware reranker as default | the deployed pair is the two highest gates; reranking is measured by the bounded calibration when `inference.pair_rerank` is on (the preflight turns it on to exercise the code path once, which is all the ablation claims) |
| Exact Oracle in the training loop | `G_exact` needs one backbone forward per expert per sample. `exact_oracle.training` refuses `true`; it runs only on ≤32–128 held-out samples as a check on the proxy |
