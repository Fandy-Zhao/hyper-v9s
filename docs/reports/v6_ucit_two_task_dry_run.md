# V6 UCIT Engineering Closure — Stage E12：真实两任务 Dry Run

- 日期：2026-08-05
- 分支：`exp/compose-root-cause-diagnosis`
- 任务：ImageNet-R → ArxivQA（正式序列前两个，seed 42，两任务均只跑一次）
- 验收清单：`artifacts/v6_ucit_engineering/dry_run/acceptance.json`（18 项，18 PASS / 0 FAIL）

## 运行规模（工程 Dry Run，非正式论文实验）

| 项 | 值 |
|---|---|
| 模型 | LLaVA-v1.5-7B（bf16）+ 冻结 CLIP-L/14@336，rank-8/alpha-16 LoRA（7 个 target modules） |
| Task 1 冷启动 | 2000 训练样本（train 前 2000 条），1 epoch，global batch 24（batch 1 × accum 24，单卡 GPU 4） |
| Task 1 验证 | 256 样本（train 子集）：backbone vs candidate 答案 NLL |
| Task 2 教师搜索 | 256 训练 + 128 验证样本：empty / single(旧专家 10) 答案 NLL |
| Task 2 候选训练 | 未触发（residual=0 → 按设计 commit 0，不降阈值） |
| 测试集评估 | 每任务 500 样本（原 Hyper eval 入口；正式运行用 3000） |

## 关键结果

| 项 | Task 1 (ImageNet-R) | Task 2 (ArxivQA) |
|---|---|---|
| 验证 mean_gain（候选 vs backbone） | **+0.136**（support 255/256） | —（无候选） |
| 提交专家 | **1**（expert 10, provisional, pool_version→2） | **0**（旧专家无增益 → 全空教师 → 无残差） |
| 测试 accuracy（500 样本） | **27.4%**（137/500） | **50.6%**（253/500, backbone-only） |
| 教师 empty 比例 | — | 252/256 |
| residual 样本 | — | 0（判定正确：专家增益不足不为残差） |

Task 2 的 commit-0 是**符合设计**的结果：旧专家 10 在 ArxivQA 上无答案增益（empty=2.3631 vs single=2.3667），答案教师全部为空 → residual=0 → 不创建候选（`should_create_candidates` 门控，阈值未降）→ Router 直接全局校准。这正面验证了"Residual 判定依据答案教师"与"不强制提交"两条约束。

## 阶段执行记录

Task 1：S1 数据清单 → S2 冷启动（2000 样本）→ S3 验证（mean_gain 0.136）→ S4 事务提交（expert 10）→ S5 初始 Router → S6 RMS → S7 快照 → S8 原 Hyper eval（27.4%）。

Task 2：S0 独立加载 task-1 快照（experts=[10], pool_version=2）→ S1 答案教师搜索（empty/single，256+128 样本）→ S2 残差划分（residual=0, reuse=4）→ S3 CLIP 特征 → S4 候选（跳过）→ S5 验证（跳过）→ S6 提交 0 → S7 Router 校准 → S8 RMS → S9 快照 → S10 原 Hyper eval（50.6%）。

## 18 项验收（详见 acceptance.json）

全部 **PASS**。关键证据：
- 两任务真实数据读取（train 23998 / 40000 记录）
- 两快照独立加载（`V6Snapshot.load` + component hash 校验）
- 原 Hyper eval 读取 snapshot：compose 路径（eval_task，两任务）与 PEFT 导出路径（eval_task --adapter-kind peft，100 样本）均通过
- test split 未参与训练/阈值/Router 校准（6 项集中拒绝测试 + 运行记录）
- 无 NaN、无 adapter 泄漏、旧专家 hash 不变

## 已知问题（详见交接包 known_issues.json）

1. `llava.eval.model_answer` 直接读取 Hyper peft 路径受 Hyper/peft `_get_submodules` 4/3 值解包缺陷 + merge 后 vision tower 丢失阻塞（Hyper peft 自身缺陷；正式运行使用仓库 compose eval 体系 eval_task，不受影响）。
2. Task-1 快照的 state machine 停在 RMS_READY（create 时快照先于 advance(SNAPSHOT_READY)）；resume 分析仍正确映射恢复节点。
3. dry run 评估为 500 样本子集（正式运行 3000）。

## 结论

V6 工程闭环两任务 Dry Run **验收通过**（18/18 blocking PASS）。工程状态 READY 判定见交接包 `README.json`。
