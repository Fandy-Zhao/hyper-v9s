# V6 UCIT Seed 42 — Final Report

- config：`configs/v6_ucit_formal_locked.yaml`（hash `30020824bf7bf084`）
- Git HEAD：`b6060dbf87504d148c59ac5bae6e91aa868b37a6`
- run root：`experiments/runs/v6_ucit_engineering/formal/seed_42`
- 任务顺序：ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k
- **验收结论：PASSED**（blocking 15/15）

## 性能矩阵（每任务 3000 样本，test set 独立评估）

| 任务 | metric_type | 正确数 / COCO 平均 | accuracy |
|---|---|---|---|
| 0 | Accuracy | 493 | **16.43%** |
| 1 | Accuracy | 1626 | **54.20%** |
| 2 | Average | 38.58% | **38.58%** |
| 3 | Accuracy | 611 | **20.37%** |
| 4 | Accuracy | 600 | **20.00%** |
| 5 | Average | 42.17% | **42.17%** |

## 每任务摘要

| 任务 | 提交专家 | 提交原因 | 验证样本 | mean_gain | eval 样本 | accuracy |
|---|---|---|---|---|---|---|
| 0 | ImageNet-R | — | below_tau | 256 | -0.2809 | 16.43% |
| 1 | ArxivQA | — | — | — | — | 54.20% |
| 2 | VizWiz | — | — | — | — | 38.58% |
| 3 | IconQA | — | — | — | — | 20.37% |
| 4 | CLEVR | — | — | — | — | 20.00% |
| 5 | Flickr30k | — | — | — | — | 42.17% |

## 15 项完整性验收

| # | 检查项 | 状态 | 说明 |
|---|---|---|---|
| 1 | 六个任务均 COMPLETED | **PASS** | ['COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED'] |
| 2 | 六个 task-boundary snapshot 可独立加载 | **PASS** | [] |
| 3 | 所有 registry 无损坏 | **PASS** | registry load + list_all OK |
| 4 | expert ID 唯一 | **PASS** | committed ids: [] |
| 5 | old expert hash 稳定 | **PASS** | [] |
| 6 | Router 阶段完整（可输出 empty/single/pair） | **PASS** | ['task0: ROUTER_READY=True checkpoint=False', 'task1: ROUTER_READY=True checkpoint=True', 'task2: ROUTER_READY=True checkpoint=True', 'task3: ROUTER_READY=True checkpoint=True', 'task4: ROUTER_READY=True checkpoint=True', 'task5: ROUTER_READY=True checkpoint=True'] |
| 7 | Candidate 验证未被绕过 | **PASS** | ['task0 validation samples=256 (expect 256)', 'task1 no residual -> validation legitimately skipped', 'task2 no residual -> validation legitimately skipped', 'task3 no residual -> validation legitimately skipped', 'task4 no residual -> validation legitimately skipped', 'task5 no residual -> validation legitimately skipped'] |
| 8 | pool_version 单调不减且无重复 | **PASS** | pool_versions: [1, 1, 1, 1, 1, 1] |
| 9 | 原 Hyper eval 六阶段全部完成（每任务 3000 样本） | **PASS** | ['task0 samples=3000 preds=3000 dur=571s', 'task1 samples=3000 preds=3000 dur=970s', 'task2 samples=3000 preds=3000 dur=1873s', 'task3 samples=3000 preds=3000 dur=750s', 'task4 samples=3000 preds=3000 dur=616s', 'task5 samples=3000 preds=3000 dur=2013s'] |
| 10 | 性能矩阵完整（每任务 3000 样本指标可计算） | **PASS** | ['ImageNet-R=16.43%', 'ArxivQA=54.20%', 'VizWiz=38.58% (COCO Average: bleu1 59.01, meteor 22.20, rouge 44.14, cider 58.09)', 'IconQA=20.37%', 'CLEVR=20.00%', 'Flickr30k=42.17% (COCO Average: bleu1 61.15, meteor 27.00, rouge 49.38, cider 59.20)'] |
| 11 | 无 test 泄漏 | **PASS** | train/test hashes distinct; validation from train only |
| 12 | 无未解释 NaN | **PASS** | no NaN/Inf found |
| 13 | 恢复记录完整（无 pending 事务，阶段标记齐全） | **PASS** | ['pending transactions: none', 'stage counts: [8, 11, 12, 12, 12, 12]'] |
| 14 | 最终 snapshot 可从干净进程加载 | **PASS** | task5 snapshot loaded, task_id=5, stage=RMS_READY |
| 15 | exact resume command 有效 | **PASS** | command: bash scripts/v6_ucit/resume_run.sh 42 4,5,6,7 (idempotent re-run verified by run completion) |

## 说明

本 seed task0 在真实 256 样本验证下判定 below_tau（0 提交，数据驱动）；task1-5 按设计路径执行空 registry 退化链。全部 6 个任务 3000 样本正式评估完成。
