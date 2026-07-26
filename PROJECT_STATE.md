# Project State

## Snapshot
- Date: 2026-07-26
- Branch: `zzf`
- Project type: Python deep learning / MLLM research (ACL 2025)
- Current focus: Project governance initialization — cleanup and documentation

## Active Work
- **Project governance initialization**: Creating AGENTS.md, directory structure, archiving deprecated files, moving analysis tools to `tools/`
- **Branch `zzf`**: Active development branch for Hyper-LLaVA experiments

## Known Risks
- **Large binary files in root**: Three `flash_attn-*.whl` files (~1GB total) and `nohup.out` (23MB) are in the repository root — these should be moved to external storage or `.gitignore`'d
- **Symlinked directories**: `instructions/`, `runs/`, `ucit_instructions/` are symlinks to external paths (`/data/ckpt/`, `/data/dataset/`) — repository portability depends on these paths existing
- **No test suite**: The project has no automated tests; validation relies on full training/eval runs
- **`gaussian.py`**: Contains non-runnable code snippets extracted from model implementation — kept for reference only
- **`compute_routing_weights.py`**: Must remain in root — imported by `llava/train/train_MOE.py` at line 44

## Validation Status
- [ ] Python import smoke test
- [ ] Config validation (`config.json`)
- [ ] Shell script `bash -n` check
- [ ] Training dry-run (requires GPU)
- [ ] Full UCIT benchmark evaluation (requires GPU cluster + dataset)

## Next Steps
1. Complete governance initialization (AGENTS.md, directory structure, file cleanup)
2. Move or `.gitignore` the large `.whl` and `nohup.out` files
3. Decide on `gaussian.py` disposition (keep as reference or integrate into docs)
4. Create a `dev` branch for integration workflow
5. Run import smoke test and shell syntax validation
