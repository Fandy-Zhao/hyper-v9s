# Project State

## Snapshot
- Date: 2026-07-29
- Branch: `feat/0729-compose-foundation`
- Project type: Python deep learning / MLLM research (ACL 2025)
- Current focus: Compose foundation validation alongside the retained Hyper-LLaVA baseline

## Active Work
- **Compose foundation**: Independent LLaVA model, LoRA expert composition, ExpertPool, adapter checkpoints, and UCIT Task1 entry are implemented on `feat/0729-compose-foundation`.
- **Compose validation**: 16 unit tests pass and the post-review two-step single-GPU ZeRO-2 Task1 smoke completed with 224 decoder-only adapters and finite loss (`2.3188`, `0.8274`).
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
- [ ] Full UCIT benchmark evaluation (requires GPU cluster + dataset)

## Next Steps
1. Review Compose Foundation and decide whether to integrate it into the long-running development branch.
2. Add a non-fixed router only in a separate follow-up after defining expert keys and evaluation criteria.
3. Run a full UCIT Task1 epoch and evaluate adapter checkpoints before starting Task2 composition experiments.
4. Move or `.gitignore` the existing large `.whl` and `nohup.out` files in a separate cleanup task.
