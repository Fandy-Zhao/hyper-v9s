# Analysis Tools

Standalone scripts for CKA similarity analysis, Gaussian statistics, data cleaning, and model debugging. Moved from repository root during governance initialization (2026-07-26).

## Scripts

| Script | Purpose | Usage |
| --- | --- | --- |
| `test_CKA_sim.py` | Compute Centered Kernel Alignment similarity between model layers across tasks | `python test_CKA_sim.py` |
| `calbc.py` | Analyze expert distribution overlap via Log Bhattacharyya coefficient; compute image/text fusion weights | `python calbc.py` |
| `calpd.py` | Map expert Gaussian statistics to Poincaré ball, compute hyperbolic distance matrices | `python calpd.py` |
| `calwd.py` | Compute pairwise 2-Wasserstein distance between expert image/text Gaussian statistics | `python calwd.py` |
| `clcalbc.py` | Simulate sequential continual learning stages and compute adaptive image/text fusion weights per stage | `python clcalbc.py` |
| `verify_gaussian.py` | Validate diagonal Gaussian assumption via PCA, marginal distribution, and correlation plots | `python verify_gaussian.py` |
| `change_vizwiz_json.py` | Clean VizWiz annotation JSON: fix duplicate image extensions in file paths | `python change_vizwiz_json.py` |
| `debug.py` | Load and inspect adapter checkpoint keys (LoRA/adapter weights) | `python debug.py` |
| `eval.py` | COCO caption evaluation entry point (BLEU, METEOR, ROUGE_L, CIDEr) — custom replacement for pycocoevalcap | Used via `pycocoevalcap` import |

## Dependencies

Most scripts require: `numpy`, `torch`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`. Install via `pip install -r ../requirements.txt`.

## Note

- `compute_routing_weights.py` remains in the repository root because it is imported at runtime by `llava/train/train_MOE.py` (line 44).
- `gaussian.py` remains in the repository root as a reference file (non-runnable code snippets extracted from the model implementation).
