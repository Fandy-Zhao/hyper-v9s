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
`legacy_v9_only` schema migration ✅ → 74 static checks ✅ → **one** 2-rank
end-to-end preflight (10 bootstrap / 25 soft / 15 ST steps + the §30 calibration
in the same run) — which found the detach bug, the masked `L_sparse` and the
gradient-window bug below, all three fixed and pinned by the regression tests
it names → re-run of the *same* preflight, which reached the calibration stage
and crashed there, exposing that the whole post-training path had never executed
— audited statically and fixed below → re-run of the *same* preflight, which
trained cleanly through all 50 steps and then died inside the calibration at the
first exact-removal forward, on an accumulator taken out on the wrong device →
**completed** 2-rank preflight (50 steps, calibration report produced, task-end
audit applied, pool committed and handed off), whose §35 report was wrong and is
fixed below → full Task0 → Task0 eval/audit/commit → Task1…Task5 →
lower-triangular matrix → final report.

### The formal run scripts (`experiments/runs/0915_v9_main/`)

`run_chain.sh` → `run_task.sh <N>` for N = 0…5, plus `probe_task1_handoff.sh`,
which is *not* part of the chain — it is the one pre-chain GPU run, and the
reason for it is in the next section.

One root per task, `task0`…`task5`, not one shared root. That is forced by the
query cache: the file is `features/train.json`, its name contains no task, and
an existing file is never re-encoded (spec §4) — a shared root would give task 1
ImageNet-R's queries, and the id check would then abort the task rather than
silently train on them. `_previous_state_path` supports both layouts; with
per-task roots the committed pool is found one level above the checkpoint
directory (`<task{N-1}>/state/key_pool_task{N-1}.pt`), which is where the audit
writes it. Per-task roots also keep each task's metrics, audit and handoff
artefacts beside the task that produced them.

Cost, counted rather than estimated: 211 244 training samples over the six
formal splits = 6601 optimizer steps × 28.36 s = 52.0 h of pure training, plus
per-task startup, query-cache encoding (93.4 KB/sample, 19.7 GB total; ~2.5 min
per 1600 samples, measured on the task-1 probe) and the §30 calibration. The
runner refuses to start a task with under 6 GB free, and that is the only disk
guard: 45 GB is free and the whole chain needs ~25 GB, so nothing is deleted
automatically — a finished task's `checkpoint-*` directories are left in place,
because they are both its resume point and its evidence.

### The task-1 handoff probe (the path a task-0 preflight cannot reach)

Task 0 owns no historical expert: no committed pool to load, no task keys, no
frozen key, a routing row that is just the M candidates. Everything the continual
setting adds first executes at task 1, which in the formal chain is 5.8 h in.
`probe_task1_handoff.sh` runs that task against the *preflight's* committed task
0, on the ArxivQA splits truncated to 1600 train samples, otherwise the main
config.

It executed the whole path, and it found one bug on the way:

| | |
| --- | --- |
| Handoff | `task 1 pool: 2 historical, candidates [2, 3], task keys 2`; the preflight's task-0 pool loaded as frozen history, task 1 minted candidates 2 and 3 on top of it |
| Wide retrieval is not decoration | `Top-2 reaches 2 experts, wide Top-2 reaches 2 (0 only in the wide recall)` — expert 0 is outside Top-C and present in the wide recall, which is the door the fixed exploration expert used to hold open |
| **Crash on step 0** | `RuntimeError: One of the differentiated Tensors does not require grad` at `_assert_answer_loss_isolated_from_keys`. The §41 (A) audit hands autograd the *whole* key set; a frozen historical key is not a differentiable leaf, so autograd refuses it. Task 0's key set is entirely trainable, so no task-0 run — preflight or formal — could ever have hit this, and it fires after the handoff, at the first step of a 10 h task |
| Fix | Frozen keys are reported rather than differentiated (`frozen_keys_not_differentiated`, `measured_keys`); the freeze audit independently proves they are frozen, so together the two cover the whole key set. Both regression tests reproduce the production error when the fix is reverted, and 76/76 static checks pass with it |
| Reporting gap found at the same time | The isolation report was computed every task and **written nowhere**: the evidence for §41 (A) was a silence, indistinguishable from a check that never ran. Now written to `v9_answer_key_isolation.json`, with a test that the report survives `json.dump` (a tensor in it would have failed *after* the training loop) |
| Calibration with a historical row (64 samples) | `pearson 0.9733`, `spearman 0.7802`, `sign_agreement 0.8125` — the gate-gradient proxy survives trained experts in the row |
| Task-end audit, on a non-empty pool | Candidates 2, 3: `selected_rate 1.0`, `positive_contribution_rate 0.376`, but `validation_gain −0.0074 <= 0` → deleted. Historical 0, 1: `retain_task_key`, retention 1.000, `selected_rate 0.76`. Freeze audit clean (`historical_key/lora/rms_unchanged`), coverage 1600/1600 with 0 padding |
| What that last row means | On a 50-step probe the candidates cannot yet beat a trained historical expert, so the audit deletes them and the pool does not grow — correct behaviour under §19 and a property of the *probe's* length, not a result. Whether 742 steps clear a 0.0 validation-gain bar is the first thing the formal run will answer |

### The corrected cost projection

The 52.0 h figure this file carried earlier was wrong, and the probe is what
showed it: it used the preflight's 28.36 s per optimizer step, and that step
offered **two** experts. Cost scales with the experts a row offers, not with the
pool: two measured points (3.66 s/micro-step at 2 offered, 6.07 s at 4) fit
`1.25 s + 1.21 s per offered expert`, and a formal row offers
`min(top_c=4, history) + M=2` on base steps and `min(wide top_c=8, history) + 2`
on the 5% wide steps.

| task | steps | history | base experts | base s/step | hours |
| --- | --- | --- | --- | --- | --- |
| 0 ImageNet-R | 742 | 0 | 2 | 29.3 | 6.0 |
| 1 ArxivQA | 1241 | 2 | 4 | 48.6 | 16.7 |
| 2 VizWiz | 1235 | 4 | 6 | 67.8 | 23.3 |
| 3 IconQA | 925 | 6 | 6 | 67.8 | 17.7 |
| 4 CLEVR | 1241 | 8 | 6 (10 wide) | 67.8 (106.4) | 24.1 |
| 5 Flickr30k | 1216 | 10 | 6 (10 wide) | 67.8 (106.4) | 23.6 |

**111.3 h of training, ~4.6 days**, plus query-cache encoding of 211 244 images
(0.6–5.5 h depending on the encode rate, which the probe measured only including
model load) and ~3 h of calibration, audits and per-task startup: **~5 days
total**. No lever exists inside the frozen recipe — the cost *is* "every offered
expert participates in one backbone forward" — and `top_c`, `wide top_c`, M and
the stage ratios are all spec values.

It is worth recording, because it is the argument for the run existing at all: a
50-step training loop completed cleanly, wrote its checkpoints, passed its
isolation audit, and would have gone on to a full Task 0 producing a table of
numbers in which **no key was ever supervised**. Every logged quantity was
`0.0`, and `0.0` is what a healthy run looks like too.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `gate_grad_abs_mean: 0.0`, `contribution_mean: 0.0`, `responsibility_*: 0.0`, `loss_key: 0.0` | composition consumed `p.detach()`; the answer loss was a function of no gate tensor, and `gate_gradient` returned a clean zero for it | forward gate built from detached inputs and marked a leaf; `gate_gradient` raises instead of returning that zero |
| `loss_sparse: 0.0` and `loss_total == loss_answer` | the sparse/budget terms were masked by `contribution.valid`, which is all-`False` exactly while the routing is undecided | row-level regularisers are defined on the routing row; the mask no longer reaches them |
| task-end audit: `candidate_lora_checksums` differ across ranks — the two ranks were never training the same model | `_at_sync_boundary` read `(global_step + 1) % accumulation`, and `global_step` only advances *at* the boundary it detects, so the flag was constant within a window: it fired on all 8 micro-steps of one window in 8 and on none of the other 7 | boundary is `accelerator.sync_gradients` again, as in V8 — and a boundary the trainer fails to notice now raises rather than diverging |

That re-run trained cleanly end to end (1420 s, checkpoints written, isolation
audit `max_abs_gradient == 0.0`), then died the moment it left the training loop.
The lesson is the same one, one stage later: **the preflight is the first
execution of every stage after the loop**, and a stage that has never run cannot
be assumed to work because the stage before it did. The crash was a
`FileNotFoundError` on the retrieval cache; rather than spend a third GPU run
discovering the next one, the whole post-training path — calibration → task
statistics → audit → commit → handoff — was read line by line. It had nine
defects. Three would each have aborted the run where it stood; two would have
silently produced a wrong number in a report that is supposed to justify the
method.

### What the second preflight found

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: features/historical_topc_task0_val.pt` on rank 0, gloo `Connection closed by peer` on rank 1 | `build_task_retrieval(force_build=True)` read its *cache* argument as "do not write one" and built into `os.devnull`: the flag means "do not read one, but *do* write it" | `cache_path` is always the real path — the rebuild is already forced by `replace(config, cache=False)`, which is what suppresses the read, and the explicit `save` then writes where the manifest points; a test builds into a `tmp_path` and loads it back |
| the calibration loader fed `historical_topc`-less batches | it used `V7QueryCollator`, which does not collate the V9 top-C column | `V9QueryCollator` |
| `RuntimeError: shape mismatch`, `[B]` against `[B, S]` | `delta` is per sample, the removal matrix is per sample **and** per slot | `delta.reshape(-1, 1)`; the regression test uses a `[3, 5]` fixture so `B ≠ S` and the broadcast cannot come back |
| `ValueError: padded slots must have zero gates` at `types.py:86` | the exact-removal and pair-probe paths left the gate of a padded slot untouched | every path zeroes the gate wherever it pads the id, not only where it masks the loss |
| the whole run hung at the calibration barrier, then died at teardown | `global_task_statistics()` all-gathers, and it was called inside the rank-0-only `should_save` block: rank 1 left `train()` and exited while rank 0 sat in the collective | the collective is hoisted out of the `should_save` block; every rank enters it |
| `candidate_commit_applied: false` in every audit artefact | `apply_candidate_commit` was imported by `v9_task_run.audit_task` and never called — candidates were audited, decided, and then nothing happened | `audit_task` applies the decision and reports it |
| `TypeError: add_expert() got an unexpected keyword argument 'committed_task'`, on the *next* task's load | the commit wrote `committed_task` as a top-level record field; `from_state` splats every unrecognised field into `add_expert(**fields)` | bookkeeping goes in the record's `extra`, which is the declared extension point — found by the new round-trip test, and it would have broken Task 1, not Task 0 |
| `responsibility_mean` was `1/B` of the quantity §30 defines | `gate_gradient` was fed `per_sample.mean()`, so the batch width leaked into the gradient and into every `grad_mean` in the calibration report | `.sum()`; the test pins the convention with a `B = 4` fixture and a comment saying why |
| `pair_rerank_report.deployed_rate` was computed from `torch.zeros_like(deployed)` | the flag was declared and then filled with a constant, so the ablation reported a rate it had not measured — a tautology, not a bug in the arithmetic | the probe compares the deployed pair it actually served against the pair column 0 selected, per row, and reports the real rate |

### What the third and fourth preflights found

The same lesson, one stage further each time. The third run trained its 50
steps cleanly and died on the calibration's *dataset*; the fourth trained
cleanly and died on the calibration's *backward*. Both were in code that had
never executed, and neither was reachable from a CPU test that only exercised
the stages in isolation.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValueError: V7 query cache misses 64 train samples` — every sample of the validation split, against a cache that exists and is correct | the encoder names cache entries `id` → `question_id`, and the validation split carries its identifier in the *second* field while the training split carries it in the first; `V7QueryDataset` read only `id` → index, so no validation sample could ever be addressed | one `_query_cache_sample_id` rule for the dataset, `V9QueryDataset` uses the parent's ids instead of re-deriving them, and a test compares the rule against the encoder's own `shard_expected_ids`. No validation path had ever gone through this dataset — V8 evaluates through `v7_dynamic_eval`, which reads `question_id` first |
| `CheckpointError: A different number of tensors was saved during the original forward and recomputation. 197 vs 29.` on the calibration's first batch | the calibration backward ran *after* the selection context had closed; activation checkpointing recomputes each block during `backward()`, and a recomputation that cannot see the routing rebuilds a different graph. The training loop has always held the context across forward *and* backward (`_ddp_training_step` says so in as many words) — the calibration was a second, wrong copy of that contract | the backward moved inside the context. The stub model in the static suite now runs its body inside `torch.utils.checkpoint` with the gates passed as inputs, so this test fails on the old code — verified by reverting the fix and watching it fail |

### What the fifth preflight found

By now the pattern is explicit enough to state as a rule: **each run reaches
exactly one stage further than the last, and the stage it dies on has never
executed before.** The fifth run trained its 50 steps without a single traceback
(1418.68 s, `train_loss 0.4132`, three stages entered in order), wrote its
checkpoints, and then died in the calibration's first exact-removal forward.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` at the `torch.where` that fills the removal matrix | `torch.zeros(expert_ids.shape, dtype=torch.float32)` takes the *shape* from the routing row but not the device: a shape-only allocation defaults to CPU, so the first comparison against a cuda `delta` is a hard error | the accumulator is allocated on `expert_ids.device`. The static test that drove this method was CPU-only, and on CPU every tensor agrees by construction — which is the whole reason this survived 74 green checks and five GPU runs. The new test executes the *same* method with the routing tensors on a second device, and reproduces the production traceback exactly when the fix is reverted |

The run produced no calibration report at all: it aborted before
`calibration_report` was reached, so `v9_contribution_calibration.json` was never
written and `G_grad` vs `G_exact` is still unmeasured. That is the honest state
of the §30 check before the next run, and it is why the §40 checklist cannot be
signed yet.

### What the sixth preflight found

The first run that finished: 50 steps (10 bootstrap / 25 soft / 15 hard, 1418 s
of training), checkpoints written, `v9_contribution_calibration.json` finally
produced, task-end audit applied, pool committed and handed off. Two of its
findings are worth recording.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `pair_rerank.best_pair_is_deployed_rate: 0.0` in a run where the deployed pair *is* the probed pair on every sample | the probe emits an **indicator** (`1.0` = column 0 is what this row's deployed rule serves) and the report compared it against an arg-min **position** (`0` = column 0 is the best measured pair). Each half was pinned by its own test — the probe against `[1.0, 0.0]`, the report against a hand-written `torch.zeros` — so both passed while the composition of the two reported `0/N` for an all-success case and a spurious success for a sample whose deployed pair was never measured at all | the report reads the indicator as an indicator, computes the rate on the samples the question applies to, and reports `comparable_samples` beside it. Found by re-executing the real probe against the *saved* pool/keys/bias: 32/32 rows flag 1.0, so the corrected rate is 1.0, not 0.0. The recorded artefact is left as the run wrote it; the fix applies to the next run |
| the per-step timing looked like 24.7 s of unexplained overhead | `training_step_sec` is a **micro-step** duration and the logged cadence was a **whole optimizer step** (8 micro-steps). 8 × 3.66 s = 29.3 s ≈ the 28.36 s cadence | no change. Measured rather than assumed: the dataloader delivers 179 samples/s on 2 workers (0.1 s per step's worth of samples), so the pipeline is not a factor and no worker-count change is warranted |

### The calibration result (spec §30, §34, §35), first time produced

32 held-out validation samples, task 0, after the 50-step preflight:

| Quantity | Value | Reading |
| --- | --- | --- |
| `pearson` | **0.8854** | the gate-gradient proxy tracks the exact remove-and-reroute effect |
| `spearman` | **0.9180** | and tracks it in rank, which is what the responsibility normalisation uses |
| `sign_agreement` | **0.8594** | it gets the *direction* right on 86% of the samples |
| `grad_mean` / `exact_mean` | −0.2953 / −0.1693 | the proxy overestimates magnitude ~1.7×; the direction is what `L_key` consumes, which is why responsibility is normalised |
| `top1_agreement` / `topk_recall` | 0.0 / 0.0 | the extreme tail disagrees. n = 32, and a single arg-max over 32 near-tied values is the weakest possible form of this statistic — reported, not hidden |
| `validation_scores.soft` / `hard_top2` / `gap` | 0.8685 / 1.1428 / **+0.2743** | the price of serving the deployed Top-2 rule instead of the mixture the objective minimises, measured on held-out data |
| `candidate_validation_gain` | {0: −0.1717, 1: −0.1668} | both candidates currently *hurt* on held-out data |

The last row is the honest state of a 50-step compressed run: the audit recorded
the negative gain and committed anyway, with the reason written into
`data/audit_task0.json` (`"committed to keep 2 selectable experts for the Top-2
deployment rule"`). The full-schedule run is what has to make that number
positive, and it is the first thing the Task 0 report must show.

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
