# V9 — Answer-Guided Key–Expert Co-Evolution

Branch: `exp/v9-answer-guided-key-expert-coevolution` (base `798268d`, V8 branch `main` untouched).

## 0. Audit summary (what already exists and is reused verbatim)

| V9 need | Existing V8/V7 implementation | Reuse |
| --- | --- | --- |
| One backbone + N LoRA branches | `compose/adapters/lora.py::ComposeLinear.forward` — `base_layer(x)` once, then per-expert `index_select → LoRAExpert → gate·kappa·scale → index_add_` | **verbatim** |
| Multi-key max aggregation | `compose/v8/routing.py::MultiKeyRouter.score_matrix` (`scatter_reduce_ amax` then distinct-expert Top-K) | **verbatim** (inference) |
| Fixed query | `q = L2Norm(concat(LayerNorm(z_vis), LayerNorm(z_txt)))`, 768+768=1536, `V7QueryConfig` | **verbatim** |
| Query cache I/O | `V7QueryDataset` / `V7QueryCollator` (`queries.pt` tensor + `sample_ids`) | **verbatim** |
| Frozen-parameter enforcement | `ExpertManager.train_only`, `freeze_historical`, `assert_historical_lora_frozen`, `adapter_checksums`, `historical_checksums`, `assert_task_freeze_integrity` | **verbatim** |
| DDP | `torchrun` + `mean_sync_accumulated_gradients`, `attach_v7_ddp_key_anchor`, `_enable_non_reentrant_checkpointing`, rank0-only save | **verbatim** |
| per-sample answer NLL | `llava_llama.per_sample_token_mean_nll`, surfaced on `self.v7_per_sample_answer_nll` | **verbatim** |
| RMS kappa | `compose/lora/rms.py` (`kappa_k_l = clip(R̄_l / R_k_l, 0.25, 4.0)`) + `merge_commit_frozen_calibration` | **verbatim** |
| Task-end prune/commit | `compose/v7/commit.py`, `CandidatePruner` | **adapted** |
| Checkpoint contract | `compose_experts.bin` / `compose_experts.json` / `v7_keys.pt` | **kept identical** |

Two deltas required from V8 core (both additive, both default to the V8 behaviour):

1. `compose/adapters/types.py` — `MAX_ACTIVE_EXPERTS = 4` is a module constant asserted in
   `ComposeSelection.__post_init__`. V9 needs `C + M + 1 = 9` slots. Add an instance field
   `max_slots: int = MAX_ACTIVE_EXPERTS`; every V8 call site keeps 4 and stays bit-identical.
2. `compose/adapters/lora.py` — cardinality scaling (`1/sqrt(N)`, pair `1/√2`) is V8's variance
   rule and contradicts the §8 formula `h ← base + Σ_k a_k·κ_k·u_k`. Add
   `cardinality_scale: "v8" | "none"`, default `"v8"`. V9 config selects `"none"`.

## 1. Hardware reality (reported, not worked around)

`nvidia-smi -L` → **one** `RTX 4090` (48 GB). Physical multi-GPU verification is impossible on
this machine. Plan: run the real training entry under
`torchrun --standalone --nproc_per_node=2` with both ranks on GPU 0, which exercises every
distributed code path that matters (DDP gradient all-reduce, `find_unused_parameters`,
`all_gather_object` audits, rank0-only checkpointing, non-elastic resume, pool-consistency
assertions). This is disclosed as a known limitation, not presented as multi-GPU validation.

The fixed-query cache is **absent** locally, so S1 must encode queries online.

## 2. Closed loop being implemented

```
Key decides where to learn ──► Expert learns what to do ──► Answer judges whether it helps
        ▲                                                              │
        └──────── Key responsibility loss (BCE) ◄── responsibility ◄── gate gradient
```

Per optimizer step: one backbone forward → `C+M+1` differenced-gated LoRA branches →
one answer NLL → `autograd.grad(L_ans, gates, retain_graph=True)` → `G = −a·∂L/∂a` →
`r = G₊/ΣG₊` → `L_key = BCE(a, r)` → **one** backward on `L_total`. No second-order graph,
no per-expert backbone enumeration, no pair enumeration.

## 3. New module tree (`compose/v9/`)

```
config.py       V9Config + sections (expert/retrieval/query/key/routing/bootstrap/
                schedule/loss/composition/validation) — no magic numbers elsewhere
keys.py         V9KeyPool(MultiKeyExpertPool): base key (frozen) + task_residual delta
                (trainable, zero-init) + candidate keys + per-expert routing bias.
                effective_key = normalize(base + gamma*delta)
retrieval.py    per-task cached historical Top-C + 1 exploration expert (rebuilt once
                per task, deterministic across ranks, saved next to the query cache)
contribution.py local conditional contribution G = -a·dL/da (stop-grad) and
                answer-derived responsibility r = G_pos / (sum G_pos + eps);
                calibration_report (G_grad vs G_exact) and pair_rerank_report (§35)
losses.py       L_key (masked BCE), L_sparse (mean Σa), L_budget (mean relu(Σa-2)²)
schedule.py     Bootstrap / Soft / ST stage schedule + temperature ramp
router.py       IndependentSigmoidGate (s = cos(q, eff_key)/tau; p = sigmoid(s - b);
                temperature annealing; Straight-Through HardTop2 for stage 3) and
                V9Router, the module registered on the model (holds candidate keys,
                residual deltas, biases) -> dense [B, S] ComposeSelection
multi_key.py    global multi-key inference aggregation: expert_score(i,k) =
                max over the memory keys of K_k of cos(q_i, e)
inference.py    query-only deployment routing (no answer, no task id), purity-checked
audit.py        task-end: historical task-key audit + candidate commit statistics
                (+ apply_candidate_commit / apply_historical_residual_audit),
                global all_reduce aggregation
checkpoint.py   atomic task-level V9 save/resume + RNG capture/restore
data.py         V9QueryDataset / V9QueryCollator: the fixed-query cache path
trainer.py      V9ComposeTrainer(ComposeTrainer): one-forward closed loop, single
                backward, global stats, V9 checkpoint payload
```

(Commit lives in `audit.py` as `apply_candidate_commit` -- there is no separate
`commit.py`; the module map above is checked against `compose/v9/` on disk.)

New entry point `compose/experiments/v9_task_run.py` (per-task orchestrator) plus
`configs/v9_*.yaml`. V8 entry points are untouched.

## 4. Parameter audit (what trains, what does not)

Trainable: current-candidate LoRA, current-candidate keys, current-candidate routing biases,
historical `task_residual` deltas for the current task only.
Frozen: vision tower, projector, backbone, token embeddings, fixed query, historical LoRA,
historical base keys, previous tasks' residual/alias keys, RMS kappas.
Fail-fast if anything in the frozen set is `requires_grad` (`trainable_parameter_audit`).

## 5. Execution order (§43)

audit ✅ → clean branch ✅ → `IMPLEMENTATION_PLAN.md` → data/model structs → routing →
contribution/responsibility → DDP-safe statistics → checkpoint/inference closure →
static checks → **one** multi-GPU end-to-end preflight (10 bootstrap / 25 soft / 15 ST steps
+ G_grad vs G_exact calibration in the same run) → fix only real failures → full Task0 →
Task0 eval/audit/commit → Task1…Task5 → lower-triangular matrix → final report.
