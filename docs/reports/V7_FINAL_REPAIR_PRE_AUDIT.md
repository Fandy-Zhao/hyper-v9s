# V7 Final Repair Pre-Audit

Audit baseline: `ee971f75a69c2945a85d380620476d57084b0e3d` on
`exp/v7-full-data-global-key-expert-coevolution`.

This pre-audit records the implementation state found before the formal-repair
commit series. It supersedes readiness conclusions in earlier V7 reports; it
does not supersede their historical experiment evidence.

| Issue | Current Status at baseline | Evidence | Code Location | Planned Fix | Method Semantics Changed? |
|---|---|---|---|---|---|
| Formal full-data training | FAIL | Runner implicitly capped training at 30 optimizer steps and did not prove dataloader coverage. | `compose/experiments/v7_task_run.py`; `compose/v7/hf_trainer.py` | Separate uncapped formal mode from explicit smoke mode and audit observed source IDs. | NO |
| Answer-only validation NLL | FAIL | Evaluation cloned prompt input IDs into labels instead of reusing the training supervision mask. | `compose/eval/nll_eval.py`; `compose/train/data.py` | Build NLL batches through the training dataset/collator and fail on zero supervised answer tokens. | NO |
| RMS calibration parity | FAIL | Persisted historical/current kappa application depended on the caller and was not verified across stages. | `compose/eval/load_compose.py`; `compose/eval/rms_stats.py`; `compose/lora/rms.py` | Apply one persisted runtime calibration contract by default and freeze historical values. | NO |
| Image preprocessing parity | FAIL | Train, NLL, RMS and inference did not share a machine-checked vision/projector contract. | `compose/eval/nll_eval.py`; `compose/eval/rms_stats.py`; `compose/eval/eval_task.py` | Enforce `pad` and validate vision/projector provenance before each stage. | NO |
| Candidate pruning | FAIL | Leave-one-out decisions were evaluated from one initial pool and could over-prune substitutes simultaneously. | `compose/v7/pruning.py` | Remove at most one candidate per iteration and reroute/rescore the surviving pool. | NO |
| Strict Global Top-2 pool | FAIL | Commit did not hard-fail when fewer than two distinct selectable experts survived. | `compose/v7/pruning.py`; `compose/v7/commit.py` | Protect the best two on Task0 and assert the committed selectable pool is at least two. | NO |
| Train/validation/test isolation | FAIL | `test_data_used=false` was declarative and rewritten IDs obscured source provenance. | `compose/v7/workflow.py`; `compose/experiments/v7_task_run.py` | Hash paths/files/source records before rewriting IDs and derive stage-use booleans from the pipeline inputs. | NO |
| Task-specific validation metric | FAIL | Candidate survival used negative answer NLL as both metric and loss. | `compose/experiments/v7_task_run.py` | Reuse the accepted UCIT evaluator and label explicit NLL fallback honestly. | NO |
| Accumulation-aware gradient audit | FAIL | Reading accumulated `.grad` could attribute an earlier microbatch's gradient to the current route. | `compose/v7/hf_trainer.py` | Record current-autograd contributions with hooks without clearing legal accumulated gradients. | NO |
| Formal six-task launcher | FAIL | Existing historical launcher did not express the V7 lifecycle. | `scripts/Compose/Run_UCIT/` | Add a separate resumable V7 launcher; preserve old launchers. | NO |
| Query backbone provenance | FAIL | CLIP-L/14-336 path was hard-coded in query extraction and V7 inference. | `compose/eval/query_features.py`; `compose/eval/eval_task.py` | Put the fixed backbone/path in V7 config, pass it explicitly, and record a stable backbone hash. | NO |
| Pair-scale schema | FAIL | YAML accepted any positive value while runtime silently used its own default. | `compose/v7/config.py`; `compose/adapters/lora.py` | Schema-lock the value to `1/sqrt(2)` and validate the runtime constant. | NO |
| Resume equivalence | PARTIAL | V7 state saved optimizer/scheduler/RNG, but custom trainer load did not explicitly reconnect saved optimizer/scheduler state. | `compose/v7/checkpoint.py`; `compose/v7/hf_trainer.py` | Restore pending states after optimizer/scheduler construction and compare continuous vs resumed state. | NO |
| Atomic commit | FAIL | Final directory was created before all weights, manifest and key state were written. | `compose/v7/commit.py` | Write and validate a sibling temporary directory, then atomically rename it. | NO |
| Regression coverage | PARTIAL | Core V7 tests existed, but formal acceptance, resume, atomicity and several stale 3-slot fixtures were missing. | `tests/compose/` | Add targeted acceptance tests and align stale fixtures to the established four-slot contract. | NO |

The planned fixes preserve the declared V7 algorithm: fixed 1536-D query,
Historical+4-Current global cosine Top-2 from step one, selected-current
Answer/Key learning, permanently frozen historical Key/LoRA/RMS,
validation-only pruning, commit, and task-free committed inference.
