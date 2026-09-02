# Project State

## 2026-09-02 — V7 real 7B GPU2/GPU3 validation

- GPU2 completed V7 stages S0--S2 on a bounded real ImageNet-R split, then the
  first training step exposed an unpadded Top-2 versus four-slot execution
  boundary mismatch before any optimizer update.
- The fix pads only the V7 execution representation with `-1`/zero slots; the
  Global Top-2 route, active gates, Key loss and sparse LoRA semantics remain
  unchanged.
- Validation resumes from S3 on GPU2, followed by dependent Task1 on GPU3.

## 2026-09-01 — V7 full-data global key–expert co-evolution

- Baseline: `a8f3a7860631aec8e2ea0d65ad9794838ffaffc7`.
- Active branch: `exp/v7-full-data-global-key-expert-coevolution`.
- Audit: `CURRENT_CODE_AUDIT.md`.
- Status: implementation and CPU/real-query smoke complete; V6/V6.1/V6.2 paths remain intact.
- Plan: fixed 1536-D query, four full-data candidates, global Top-2 from step
  one, selected-current Key/LoRA updates, validation removal-reroute pruning,
  frozen commit, resume-safe inference, then unit and two-task smoke tests.
- Validation: V7 acceptance 17/17; related regression 24/24; CPU Task0/Task1
  30+30 steps passed; real ImageNet-R full-declared-split query/candidate
  preparation passed. Real 7B optimization smoke awaits safe GPU capacity.

## Snapshot
- Date: 2026-07-30
- Branch: `exp/0730-compose-task1-oracle`
- Project type: Python deep learning / MLLM research (ACL 2025)
- Current focus: Staged Compose Task1 parity, multi-expert, Oracle, and capacity-control validation

## Active Work
- **Compose foundation**: Independent LLaVA model, LoRA expert composition, ExpertPool, adapter checkpoints, and UCIT Task1 entry are implemented on `feat/0729-compose-foundation`.
- **Compose validation**: Stages A--E are complete. The suite has 42 passing unit tests, grouped bf16 CUDA execution matches its reference, and full Task1 Compose/PEFT parity, expert isolation, deterministic Oracle, and rank-matched controls are recorded.
- **Compose experiment decision**: The 6,000-sample Oracle is reproducible, but mean/median synergy are negative and the L2 pair underperforms the exactly parameter-matched rank-16 adapter in NLL and accuracy. Stop condition C is active; no Set Router is implemented.
- **Project governance initialization**: Creating AGENTS.md, directory structure, archiving deprecated files, moving analysis tools to `tools/`
- **Branch `zzf`**: Active development branch for Hyper-LLaVA experiments

## Known Risks
- **Large binary files in root**: Three `flash_attn-*.whl` files (~1GB total) and `nohup.out` (23MB) are in the repository root — these should be moved to external storage or `.gitignore`'d
- **Symlinked directories**: `instructions/`, `runs/`, `ucit_instructions/` are symlinks to external paths (`/data/ckpt/`, `/data/dataset/`) — repository portability depends on these paths existing
- **Partial test coverage**: Compose has focused unit tests, while the retained LLaVA/Hyper paths still rely primarily on full training/eval runs
- **Checkpoint compatibility**: The earlier 296-layer Compose smoke checkpoint is intentionally rejected by the strict decoder-only loader and must not be used as a valid expert checkpoint.
- **Composition generality**: The negative stop decision is based on two task-trained functional proxies; it does not establish a universal result for other decompositions or larger pools.
- **`gaussian.py`**: Contains non-runnable code snippets extracted from model implementation — kept for reference only
- **`compute_routing_weights.py`**: Must remain in root — imported by `llava/train/train_MOE.py` at line 44

## Validation Status
- [x] Compose Python import isolation and syntax smoke test
- [x] Compose unit tests (42 tests)
- [x] Compose UCIT Task1 two-step DeepSpeed smoke run
- [x] Explicit LLaVA-to-Compose config conversion and core-dimension validation
- [x] Compose shell script `bash -n` check
- [x] Training dry-run (single GPU, ZeRO-2, two steps)
- [x] Strict 448-tensor checkpoint reload and output-equality validation
- [x] Explicit gate-normalization contract and active-row grouped execution
- [x] Mixed top-1/top-2 CPU and bf16 CUDA forward/backward equivalence
- [x] Full UCIT Task1 Compose and matched PEFT training/evaluation
- [x] Two-expert isolation and strict pool reload
- [x] Deterministic 6,000-sample empty/single/pair Oracle audit
- [x] Exactly parameter-matched rank-16 NLL and generation control
- [ ] Full six-task UCIT benchmark (outside this task)

## Next Steps
1. Do not implement the Compose Set Router for the current expert pool; stop condition C is recorded in ADR-0730.
2. If composition is revisited, first produce a new deterministic fixed-set audit that beats an exactly parameter-matched single adapter on NLL and accuracy.
3. Extend beyond two functional proxies only as a separately governed experiment with the same isolation and capacity controls.
4. Move or `.gitignore` the existing large `.whl` and `nohup.out` files in a separate cleanup task.
