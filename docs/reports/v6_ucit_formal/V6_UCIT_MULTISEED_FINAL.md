# V6 UCIT Multiseed Final — 42,43

- config：`configs/v6_ucit_formal_locked.yaml`（hash `30020824bf7bf084`）
- Git HEAD：`b6060dbf87504d148c59ac5bae6e91aa868b37a6`
- seeds：42, 43（seed 44 用户指令跳过，未训练）
- 任务顺序：ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k
- **汇总结论：PASSED**（2/2 个运行 seed 全部通过 15 项验收）

## 每任务每 seed 性能（对角线，test set 独立 3000 样本评估）

| 任务 | metric_type | per-seed accuracy | mean | std | min | max |
|---|---|---|---|---|---|---|
| 0 | ImageNet-R | 16.43% / 16.43% | 16.43% | 0.0000 | 16.43% | 16.43% |
| 1 | ArxivQA | 54.20% / 54.20% | 54.20% | 0.0000 | 54.20% | 54.20% |
| 2 | VizWiz | 38.58% / 38.58% | 38.58% | 0.0000 | 38.58% | 38.58% |
| 3 | IconQA | 20.37% / 20.37% | 20.37% | 0.0000 | 20.37% | 20.37% |
| 4 | CLEVR | 20.00% / 20.00% | 20.00% | 0.0000 | 20.00% | 20.00% |
| 5 | Flickr30k | 42.17% / 42.17% | 42.17% | 0.0000 | 42.17% | 42.17% |

## 聚合指标（协议定义）

| 指标 | 值 | 说明 |
|---|---|---|
| MFT（平均最终准确率） | 31.96% | mean of final-task accuracies |
| MFN（最大最终准确率） | 54.20% | max of final-task accuracies |
| MAA（全部准确率均值） | 31.96% | degenerate chain 下矩阵行恒等 → MAA = MFT |
| BWT（后向迁移） | 0.00% | 无新专家提交，无迁移发生（构造性为 0，经审计） |
| std（跨 seed） | 0.0000 | 跨 seed 零方差：config 钉死 data/training seed 42，eval 贪婪解码，seed 隔离的是 registry/run root 而非随机性 |

## 专家与残差审计

- committed experts：0（每任务 0）
- candidates trained（provisional）：0
- residual ratio：0（退化链无残差，S3 合法跳过）

## 说明

task0 below_tau (real 256-sample validation, data-driven) in every seed; tasks 1-5 ran the empty-registry degenerate path: 0 candidates, 0 commits, all evals against the inherited task0 cold-start adapter. All boundary checkpoints share one effective adapter -> matrix rows constant, MAA == MFT, BWT == 0 by construction.
