# V7 正式三卡六任务实验报告

## 实验状态

`RUNNING`（2026-09-03）。正式代码门禁已通过，六任务训练与后续 21 单元下三角评测由同一启动脚本连续执行。

## 代码与配方

- 分支：`exp/v7-full-data-global-key-expert-coevolution`
- 基线：`8139904a8423b38c005b99fcff2fd42da0ae69c5`
- GPU：0、1、2
- 每卡 batch：1
- 梯度累积：21
- world size：3
- 有效全局 batch：63；目标 64；相对偏差：-1.5625%
- 学习率：LoRA `2e-4`，Key `3e-4`；不做 DDP 学习率缩放
- 输出根目录：`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_ucit_formal_3gpu_seed42`

## 实现边界

- 任务编排、固定 Query、候选初始化、RMS、剪枝和 commit 均保持单进程。
- 只有 S3 训练由 `torch.distributed.run --nproc_per_node 3` 启动。
- 当前 Key 注册在模型中并通过零值图依赖参与 DDP 图发现；真实梯度仍只来自被选中的当前 Key，LoRA 仍为稀疏 Top-2 执行。
- checkpoint 由 rank0 单写，barrier 对称；日志与使用计数按 rank 保存并聚合。

## 验证

- Python 编译与 shell 语法检查：通过。
- 聚焦 V7 回归：`39 passed`。
- 真实 7B 门禁：Task0，12 条训练样本，3 ranks，2 optimizer steps，通过。
- 门禁 optimizer：当前 LoRA `79,953,920` 参数 + 当前 Key `6,144` 参数，合计 `79,960,064`；无其他参数。
- 训练前后所有 rank 的 Key、当前 LoRA 与 optimizer 清单一致；损失及 Key/LoRA 梯度范数均有限。

## 后续自动阶段

Task0 到 Task5 严格串行训练并 commit。完成后在 GPU0--2 上以每卡最多一个生成进程评测恰好 21 个下三角单元，输出 `continual_matrix.json`、`continual_matrix.md` 及原始 Hyper-LLaVA 汇总得到的 MAA、MFN、MFT、BWT。本报告将在全部流程结束后补充最终提交、各任务门禁和指标。
