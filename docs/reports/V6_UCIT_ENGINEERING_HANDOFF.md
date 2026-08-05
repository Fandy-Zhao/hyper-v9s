# V6 UCIT Engineering Closure — 交接包（HANDOFF）

- 日期：2026-08-05
- 分支：`exp/compose-root-cause-diagnosis`，HEAD `41e70bd`
- 交接包目录：`artifacts/v6_ucit_handoff/`（14 个文件）

## 1. 结论

**READY：`ready_for_six_task_run = true`**（`artifacts/v6_ucit_handoff/READY.json`）。
两任务 Dry Run 18/18 blocking 验收 PASS，无 blocking issue。本批次按任务书在生成交接包后停止，未运行第三任务与六任务正式运行。

## 2. 批次成果（13 个独立提交，未 push）

| Stage | 内容 | Commit |
|---|---|---|
| E0 | 仓库与 UCIT 审计（任务序列 4 源锁定） | `8a11de1` |
| E1 | 统一 ComposeSelection（空/单/双） | `458d8a4` |
| E2 | Registry 生命周期 + 事务 + 任务状态机 | `8dde5fd` |
| E3 | 双模态 Query / Key / Router 双模式 | `07170ac` |
| E4 | 答案监督教师 + Top-M 召回 + 缓存绑定 | `15c2dbf` |
| E5 | Residual Buffer（答案教师驱动） | `64b90f4` |
| E6 | Candidate 池（1/2 槽）+ 工程配置 | `5b22af5` |
| E7 | 候选验证 + 事务提交（0/1/2） | `a482615` |
| E8 | 全局教师 + Router 校准 | `108894d` |
| E9 | RMS 收集 + kappa 报告 | `ffddac5` |
| E10 | 快照 + 恢复分析 | `acc35d3` |
| E11 | 验收测试 + 静态检查 + mock DDP | `c4d9569` |
| E12 | 真实两任务 Dry Run（ACCEPTED） | `41e70bd` |

测试基线：**339 passed + 14 subtests**（含 E11 新增 12 项与 E12 集成）。

## 3. UCIT 任务顺序（正式，勿改）

**ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k**
证据：`scripts/Hyper/Train_UCIT/Task{1..6}.sh`（权威）、`summarize_continual_metrics.py`、LlaVANext 脚本、stage00 审计。见 `task_sequence.json`。

## 4. 两任务 Dry Run 结果

- Task 1 (ImageNet-R)：冷启动候选验证 mean_gain **+0.136**（255/256）→ 提交 expert 10（provisional，pool_version 2）→ 500 样本 accuracy **27.4%**
- Task 2 (ArxivQA)：旧专家无增益 → 全空教师 → residual 0 → **commit 0（按设计）** → 500 样本 accuracy **50.6%**（backbone-only）
- 18 项验收全部 PASS（`two_task_acceptance.json`）

## 5. 下一批正式运行入口

- 配置：`configs/v6_ucit_formal_locked.yaml`（唯一默认来源；本批提供 `locked_config.yaml` 副本与追加说明）
- 启动：`exact_commands.sh`（环境/启动/状态/停止/恢复/快照校验/eval/持续指标）
- 恢复：`resume_commands.sh`（幂等，pending 事务清理）
- 评估：`compose.eval.eval_task`（compose/peft 双模式，均可读 snapshot）
- 输出根：`experiments/runs/v6_ucit_engineering/`

## 6. 已知问题（`known_issues.json`）

1. **KI-001**：`llava.eval.model_answer` 直接读 Hyper peft 路径受 Hyper peft 缺陷阻塞（`_get_submodules` 解包 + merge 丢 vision tower）。正式评估使用仓库 compose eval 体系（eval_task），不受影响；如需字面 model_answer，需单独修 Hyper peft。
2. **KI-002**：Task-1 快照状态机停在 RMS_READY（写快照先于 advance）；resume 映射仍正确。
3. **KI-003**：单卡 batch-1 训练慢；正式运行建议 4 卡 batch 6 × accum 1。

## 7. 资源

- 磁盘：`/data` 空闲 3.9T；六任务估计 < 20GB（dry run 全产物 119MB）—— 充足
- GPU：0,1,4,5,6,7 空闲（4-7 优先）；2,3 与 libolin 共享
- 内存：464GB available

## 8. 停止条件执行情况

本批次按任务书全部停止条件执行：未运行六任务、未运行 seeds 43/44、未开始正式论文实验、未启用 Shadow Update/双曲空间/token Router/expert-pair MLP/自动 merge、未 push。
