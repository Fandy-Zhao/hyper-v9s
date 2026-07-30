# Project State

## Snapshot
- Date: 2026-07-30
- Branch: `exp/0730-compose-task1-oracle`
- Project type: Python deep learning / MLLM research (ACL 2025)
- Current focus: Staged Compose Task1 parity, multi-expert, Oracle, and capacity-control validation

## Active Work
- **Compose foundation**: Independent LLaVA model, LoRA expert composition, ExpertPool, adapter checkpoints, and UCIT Task1 entry are implemented on `feat/0729-compose-foundation`.
- **Compose validation**: 16 unit tests pass and the post-review two-step single-GPU ZeRO-2 Task1 smoke completed with 224 decoder-only adapters and finite loss (`2.3188`, `0.8274`).
- **Compose Stage A**: Gate normalization is explicit (`none`, `l1`, or `l2`), and heterogeneous selections execute each expert only on active batch rows. All 30 unit tests and a bf16 CUDA forward/backward reference check pass.
- **Project governance initialization**: Creating AGENTS.md, directory structure, archiving deprecated files, moving analysis tools to `tools/`
- **Branch `zzf`**: Active development branch for Hyper-LLaVA experiments

## Known Risks
- **Large binary files in root**: Three `flash_attn-*.whl` files (~1GB total) and `nohup.out` (23MB) are in the repository root — these should be moved to external storage or `.gitignore`'d
- **Symlinked directories**: `instructions/`, `runs/`, `ucit_instructions/` are symlinks to external paths (`/data/ckpt/`, `/data/dataset/`) — repository portability depends on these paths existing
- **Partial test coverage**: Compose has focused unit tests, while the retained LLaVA/Hyper paths still rely primarily on full training/eval runs
- **Checkpoint compatibility**: The earlier 296-layer Compose smoke checkpoint is intentionally rejected by the strict decoder-only loader and must not be used as a valid expert checkpoint.
- **`gaussian.py`**: Contains non-runnable code snippets extracted from model implementation — kept for reference only
- **`compute_routing_weights.py`**: Must remain in root — imported by `llava/train/train_MOE.py` at line 44

## Validation Status
- [x] Compose Python import isolation and syntax smoke test
- [x] Compose unit tests (16 tests)
- [x] Compose UCIT Task1 two-step DeepSpeed smoke run
- [x] Explicit LLaVA-to-Compose config conversion and core-dimension validation
- [x] Compose shell script `bash -n` check
- [x] Training dry-run (single GPU, ZeRO-2, two steps)
- [x] Strict 448-tensor checkpoint reload and output-equality validation
- [x] Explicit gate-normalization contract and active-row grouped execution
- [x] Mixed top-1/top-2 CPU and bf16 CUDA forward/backward equivalence
- [ ] Full UCIT benchmark evaluation (requires GPU cluster + dataset)

## Next Steps
1. Run the Stage B Task1 smoke, full Compose and rank-matched PEFT training, strict reload, and common evaluation.
2. Continue to independent multi-expert and Oracle experiments only after Stage B parity is explained.
3. Run the rank-16 capacity control before deciding whether a Set Router is justified.
4. Move or `.gitignore` the existing large `.whl` and `nohup.out` files in a separate cleanup task.
