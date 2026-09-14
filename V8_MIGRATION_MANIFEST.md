# V8 Migration Manifest

SOURCE_HOST=ubuntu
SOURCE_PROJECT_PATH=/home/zhaozhuofan/Hyper-LlaVA
SOURCE_BRANCH=exp/v8-answer-supervised-multikey
SOURCE_COMMIT=81ca61daf89f14a9b378a4ab035078bc9202a6dd
WORKTREE_DIRTY=yes
TIMESTAMP_UTC=2026-09-12T06:03:27Z
CURRENT_TRAIN_CONFIG=configs/v8_exact_accelerated.yaml
CURRENT_MICROBATCH=4
CURRENT_EFFECTIVE_BATCH=64
CURRENT_MEGABATCH=64
CURRENT_CHECKPOINTING=True
QUERY_CACHE_PATH=/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs/v7_fixed_query_cache_gpu01_20260903
MODEL_EXPECTED_PATH=/data/ckpt/zhaozhuofan/models/llava-v1.5-7b
DATASET_EXPECTED_PATH=/data/dataset/zhaozhuofan/UCIT/datasets

## Modified tracked files
compose/adapters/lora.py
compose/model/compose_llava.py
compose/train/arguments.py
compose/train/data.py
compose/train/train_compose.py
compose/train/trainer.py
compose/v7/hf_trainer.py
compose/v7/query_cache.py
llava/model/language_model/llava_llama.py

## Untracked V8-relevant files
?? compose/experiments/autotune_microbatch.py
?? compose/experiments/baseline_summary.py
?? compose/experiments/compare_profiles.py
?? compose/experiments/compare_recipe.py
?? compose/experiments/sampler_window_invariance.py
?? compose/experiments/stage_budget.py
?? compose/experiments/verify_query_tensor.py
?? compose/experiments/window_residual.py
?? compose/train/profiler.py
?? compose/train/v8_flags.py
?? configs/v8_exact_accelerated.yaml
?? docs/reports/V8_ACCELERATION_IMPLEMENTATION.md
?? docs/reports/V8_CURRENT_RUNTIME_AUDIT.md
?? docs/reports/V8_RECIPE_EQUIVALENCE_REPORT.md
?? docs/reports/V8_SPEEDUP_REPORT.md
?? docs/reports/V8_SPEED_BASELINE.md
?? docs/reports/data/phase1_query_tensor_equivalence.json
?? docs/reports/data/sampler_window_invariance.json
?? docs/reports/data/stage_budget_formal_20260903.json
?? tests/compose/test_compare_recipe.py
?? tests/compose/test_compose_forward_kwargs.py
?? tests/compose/test_metrics_path_convention.py
?? tests/compose/test_per_sample_loss.py
?? tests/compose/test_v8_analysis_tools.py
?? tests/compose/test_v8_exact_accelerated.py
?? tests/compose/test_v8_flags.py
?? tests/compose/test_window_residual.py
