# Hyper-LLaVA V6 可复用功能专家组合可行性总结

## 最终结论

`STOP_COMPOSITION`

本轮严格按预注册门槛完成 F0 上界审计、F1 真实任务对审计与 F2 受控功能组合。F0 证明旧 ImageNet-R/IconQA pair 仍有局部互补，但正确率独占价值很低；F1 找不到可辩护的真实功能依赖任务对，因此按规则转入 F2；F2 的五类通过条件在三个 seed 上全部为 `0/3`。工程隔离是成功的，功能收益不是。

## 当前负结果的准确复述

此前两个独立完整任务 expert 的 direct-sum accuracy 为 82.3833%，等参数 rank16 为 86.5333%；pair mean synergy 为 -0.03851718、median 为 -0.00094279。其问题不是 expert 无法独立训练、保存或加载，而是组合时缺乏稳定整体优势。因此本轮没有恢复 Set Router，也没有提前实现任何新 Router。

## Stage F0：现有缓存上界审计

F0 使用已有 6,000 样本逐样本缓存，没有训练或完整重跑模型。

- Oracle A：NLL 0.10853885、accuracy 88.3333%，比 best single 高 2.9000 pp，NLL 改善 0.01783272。
- Oracle B：NLL 0.09401556、accuracy 89.8167%，比 rank16 高 3.2833 pp，NLL 改善 0.01983935。
- PairExclusiveNLLRate 40.45%，但 PairExclusiveCorrectRate 仅 1.40%。Rank16CorrectPairWrong 为 5.3833%，Rank16WrongPairCorrect 为 3.45%，错误集 Jaccard 为 0.53139。
- synergy mean/median 为 -0.008620/+0.000217；最差 10% 负样本贡献 55.69% 的负幅度，负均值主要由严重尾部驱动。

F0 的三个停止子条件没有同时满足，因此记为“存在局部互补”，但这只支持继续做 formation 实验，不支持实现 Router。

## Stage F1：真实任务条件残差审计

检查 CLEVR 与 IconQA 后，没有找到能由现有标注直接构成的 A-only → A+B 功能依赖。CLEVR 缺少同分布 A-only 形状识别；IconQA 的显式形状题训练/测试占比仅约 0.55%/0.53%，且与计数题来自不同图像与题型。把 ImageNet-R/IconQA 继续解释成原子功能会违反任务约束。

F1 状态为 `SKIPPED_NO_DEFENSIBLE_REAL_TASK_PAIR`。没有启动 F1 GPU 训练，也没有把“不适用”伪装成方法通过或失败；按规则进入 F2。

## Stage F2：受控功能组合

构造 A=形状识别、B=计数、C=左右关系的离线受控数据。每个数据集有独立 1,600/200/400 train/val/test；生成 seed 730；恢复后的 36 个 JSON 与初始 manifest 哈希完全一致。

训练 7 类模型 × 3 seed，共 21 个正式 run。两个 rank-8 与 rank16 均为 39,976,960 个参数，LoRA scale、数据、epoch、optimizer、seed 全部匹配。评测矩阵共有 81 个配置，每项 400 样本；42 项来自初始成功结果，39 项来自新目录中的 batch=4 非破坏性恢复。

### 三种子结果

| Seed | A+B pair / best single / rank16 acc | A+B mean synergy | B-only Residual acc gain | B+C pair / best single / independent / rank16 acc | B+C mean synergy |
| ---: | --- | ---: | ---: | --- | ---: |
| 42 | 67.75 / 68.00 / 72.50% | -0.00729 | +21.25 pp | 34.00 / 32.25 / 32.25 / 33.00% | -0.27521 |
| 43 | 68.75 / 67.00 / 71.50% | -0.00602 | +21.75 pp | 34.00 / 32.75 / 31.75 / 33.50% | -0.27492 |
| 44 | 67.50 / 67.50 / 72.75% | -0.00597 | +19.75 pp | 34.50 / 32.75 / 32.50 / 33.00% | -0.26840 |

B-only accuracy 表明 Residual B 学到了一部分计数行为，但三个 seed 的 NLL 改善 95% CI 都跨 0。A+B 的 pair accuracy 均低于 rank16，mean synergy 三 seed 全负。B+C 的准确率三 seed 都略高于几个基线，却以约 -0.27 的 mean synergy 和相对 Independent B+C 约 -0.11 的 NLL 劣化为代价。因此生成准确率与 NLL 收益不一致，不能进入 Router 阶段。

### 隔离与身份

- 三个 seed 的旧 Expert A 均有 448 个 tensor 逐位不变；固定输入 logits 逐位不变，最大绝对差 0。
- 旧 A 与 Residual B 的 224 层 LoRA 增量平均余弦接近 0（约 -3.4e-4 至 -3.9e-4）。
- Residual B 在 B-only/A+B/B+C 上的层级激活方向平均余弦约 0.985–0.994，表示身份高度一致。

这证明训练隔离和可识别的 residual 表示成立，但没有转化为稳定的组合行为收益。

### 延迟、显存与失败模式

Base 平均生成延迟约 0.147 s/样本；Residual B 约 0.573 s；A+Residual B 约 0.733 s；Residual B+C 约 0.738 s；rank16 约 0.630 s。各配置平均峰值显存约 16.2–17.0 GiB，组合无显存优势且延迟高于 rank16。

初始 batch=8 有一次 OOM，失败原样保留；batch=4 的 39 项恢复全部成功。严重失败跨 seed 重复集中在 B+C 的高计数样本：例如 `controlled/B_plus_C/test/158` 三个 seed 都把目标 5 预测为 4，synergy 约 -2.04 至 -2.08。这说明负组合不是单一 seed 偶然事件。

## 预注册门槛与决策

F2 的 `seen_composition`、`b_only_transfer`、`unseen_composition`、`two_composition_contribution`、`nll_accuracy_direction_consistent` 五项条件均为 `0/3`。Stage F2 失败，最终决策只能是：

`STOP_COMPOSITION`

停止条件残差/自动专家组合方向。不得基于本轮结果实现 Query-Key Router、Set Router、Pair interaction network、双曲路由、Candidate Slot Pool、自动专家合并、Shadow Update、token-level routing、学习式组合 gate 或多于两个 expert 的组合。

## 主要证据

- F0：`experiments/runs/0730_residual_expert_feasibility/stage_f0/posthoc_oracle.json`
- F1：`docs/reports/0730_stage_f1_conditional_residual.md`
- F2：`experiments/runs/0730_residual_expert_feasibility/stage_f2/final_summary.json`
- F2 详细报告：`docs/reports/0730_stage_f2_controlled_composition.md`
- checkpoint 根：`/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1`
- F0 JSON SHA-256：`65b9ff8c8b717771428a1582f2f624e194ca494cd99247e4d57d4892da6f4ec5`
- F2 JSON SHA-256：`7b443133b5f75c4404f2f4aec52f961b85bcad788f9015ba9214bb239fd93efb`

本轮未 push、未创建 PR、未 merge，也未修改 14 个用户原有未跟踪 `sample_instructions/*.json`。
