# Cached Sample Oracle Audit and Offline Task-Level Decomposition

## Final status

- **CACHE STATUS: COMPLETE (LOSS-LEVEL)**
- **METRIC CACHE STATUS: PARTIAL**
- `STOPPED — 未启动任何重新推理或补充实验。`
- GPU 使用：`False`（本轮仅 CPU 文件审计与聚合）。

The six formal Sample Oracle NLL caches are complete at the loss level: every task has 3,000 samples and every sample has all 56 candidate-set losses. This is sufficient for exact mean teacher-forcing-loss matrices and loss-based Task Oracle decomposition. It is not sufficient to reconstruct benchmark accuracy for tasks whose per-combination prediction cache is absent.

## Cache audit

- Source experiment: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_v62_seed42_no_router_oracle`
- Oracle code commit: `1b2de8cc6cc07714ea24b0fe795d4a0dfb2224d3`
- Formal run commit: `fb5e9083d52cc996cdc2ab5a58634a8c489c5b6b`
- Expert pool: `E0..E9` (10 experts)
- Candidate sets: `[] + 10 singles + 45 pairs = 56`
- Primary raw cache: `sample_oracle/task{0..5}/nll.jsonl`
- Each NLL file: `3000 rows × 56 set_nll values`, valid JSON, router_called=false
- Sample generation cache: 3,000 selected-combination predictions per task, not 56 predictions per sample
- Complete 56-combination test prediction/metric cache: Task0, Task1
- Partial/interrupted test prediction cache: Task2
- Missing test prediction cache: Task3, Task4, Task5
- Direct Uniform/UR combination cache identified: none in the current formal oracle outputs; no values were inferred.

## Task-level decomposition

`Mean Loss` columns are exact offline aggregates from the 56-way NLL cache. `Metric` columns are benchmark metrics only where all 56 fixed candidate prediction files already exist. A lower loss is better.

| Task | Best Single (loss) | Task Oracle (loss) | Sample Oracle (loss) | Adaptivity gain (loss) | Best Single (metric) | Task Oracle (metric) | Sample Oracle metric |
|---|---|---|---:|---:|---:|---:|---:|
| ImageNet-R | [E4] (1.742492) | [E0,E6] (1.189956) | 0.867288 | 0.322668 | 23.13 [E0] | 35.97 [E0,E6] | 44.90 |
| ArxivQA | [E1] (0.160710) | [E1,E5] (0.078232) | 0.033108 | 0.045124 | 89.40 [E1] | 93.70 [E1,E5] | 95.87 |
| VizWiz | [E2] (1.954976) | [E2,E3] (1.945946) | 1.832567 | 0.113379 | N/A | N/A | 53.04 |
| IconQA | [E4] (0.534236) | [E3,E4] (0.432739) | 0.231885 | 0.200855 | N/A | N/A | 84.43 |
| CLEVR | [E7] (0.488905) | [E3,E7] (0.451345) | 0.199239 | 0.252106 | N/A | N/A | 86.50 |
| Flickr30k | [E9] (1.839405) | [E6,E9] (1.770458) | 1.713901 | 0.056556 | N/A | N/A | 55.35 |

## Sample Oracle composition distribution

| Task | Empty | Single | Pair | Top experts | Top pairs |
|---|---:|---:|---:|---|---|
| ImageNet-R | 0.37% | 29.30% | 70.33% | E0:2056, E9:658, E5:455, E6:449, E7:446 | [E0,E9]:405, [E0,E6]:331 |
| ArxivQA | 0.03% | 42.67% | 57.30% | E1:2940, E4:573, E5:519, E7:315, E2:215 | [E1,E4]:569, [E1,E5]:499 |
| VizWiz | 0.00% | 33.57% | 66.43% | E2:2709, E6:396, E9:380, E0:379, E3:316 | [E0,E2]:336, [E2,E6]:334 |
| IconQA | 0.37% | 36.17% | 63.47% | E3:922, E4:853, E7:810, E1:607, E5:524 | [E3,E7]:400, [E1,E4]:328 |
| CLEVR | 0.00% | 48.17% | 51.83% | E7:1696, E8:1553, E6:341, E9:303, E4:188 | [E7,E8]:483, [E7,E9]:136 |
| Flickr30k | 0.00% | 15.43% | 84.57% | E9:2079, E2:1167, E6:500, E0:470, E5:362 | [E6,E9]:321, [E0,E9]:299 |

## Interpretation and limits

1. The complete six-task 56-way loss cache makes the Task Oracle in the loss domain exactly reproducible without model inference.
2. Sample Oracle benchmark scores remain available from the existing selected-combination generation outputs, but they must not be compared to a missing Task Oracle accuracy as if it were known.
3. Task0 and Task1 have complete fixed-pool candidate predictions, so their benchmark Best Single/Task Oracle metrics can be recovered exactly from existing files. Task2 test was interrupted at the previously preserved partial checkpoint, and Task3–5 have no complete fixed-pool test prediction matrices.
4. The old auxiliary cache `compose/oracle/task1_task4_experts01_rank8_seed42/oracle_cache.jsonl` contains only 6,000 rows over ImageNet-R/IconQA with four candidate sets and an older commit; it is not merged into this V6.2 ten-expert audit.

## Generated artifacts

- Full loss/metric matrix: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_v62_seed42_no_router_oracle/cache_audit/combination_matrix.csv`
- Machine-readable matrix: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_v62_seed42_no_router_oracle/cache_audit/combination_matrix.json`
- Cache summary: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_v62_seed42_no_router_oracle/cache_audit/cache_summary.json`
- Cache audit: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_v62_seed42_no_router_oracle/cache_audit/cache_audit.json`
- Aggregation script: `/home/zhaozhuofan/Hyper-LlaVA/scripts/analysis/audit_cached_sample_oracle.py`
- This report: `/home/zhaozhuofan/Hyper-LlaVA/experiments/reports/task_level_oracle_from_cached_sample_oracle.md`

No model forward, generation, GPU computation, cache completion, or rerun was performed.
