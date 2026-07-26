# Roadmap

## Milestones
- [x] ACL 2025 paper accepted (HiDe-LLaVA)
- [x] UCIT benchmark released (6 tasks: ArxivQA, CLEVR-Math, Flickr30k, IconQA, ImageNet-R, VizWiz)
- [x] CoIN training pipeline (8-task sequential)
- [x] HyperMOELora implementation (CLIP-guided multi-expert LoRA)
- [x] Gaussian modality routing (image/text fusion weights)
- [x] CKA similarity analysis tools
- [x] Multiple training order evaluations (UCIT, UCIT_AIRFCV, UCIT_IFRCAV)
- [ ] Project governance initialization (current)

## Backlog
- [ ] Move large `.whl` files out of repository root
- [ ] Add `.gitignore` for `nohup.out`, `__pycache__/`, `*.whl`
- [ ] Clean up `gaussian.py` — either integrate into docs or remove
- [ ] Standardize training script paths (remove hardcoded `/mnt/haiyangguo/` paths)
- [ ] Add CI-compatible smoke tests (import check, config validation)
- [ ] Add type hints to core modules
- [ ] Document the CKA-based layer selection methodology

## Deferred
- Multi-node training support beyond DeepSpeed single-node
- LLaVA-NeXT model variant integration (scripts exist but untested)
- Gradio demo deployment

## Open Questions
- Should `gaussian.py` be archived as a deprecated reference file or integrated into `docs/`?
- Are the `flash_attn-*.whl` files still needed in-repo, or can they be fetched from PyPI/external storage?
- What is the target merge strategy: `zzf` -> `main` directly, or `zzf` -> `dev` -> `main`?
