# V6 UCIT Formal Launch Supplement — BLOCKED 修复记录（DEV 补齐）

- 日期：2026-08-06
- 前置：启动审计 `docs/reports/v6_ucit_formal_launch_audit.md` → **BLOCKED**
- 决策：用户批准「补齐缺失组件」（2026-08-06 定时任务会话）

## 修复的 BLOCKING 差异

| DEV | 差异 | 修复 | 依据 |
|---|---|---|---|
| DEV-6 | `configs/v6_ucit_formal_locked.yaml` 不存在（README.json 与 exact_commands.sh 均引用） | 新建 `configs/v6_ucit_formal_locked.yaml`，内容 = locked_config.yaml 工程默认值 + 正式运行追加（tasks/training/eval 段、config_hash） | 任务书「复制为新配置版本 + 记录修改原因 + 新 hash + 新 run root」；exact_commands.sh 注释（正式规模 full、eval 3000、4 卡 batch 6×accum 1） |
| DEV-9 | `scripts/v6_ucit/six_task_run.sh` 不存在（README.json run_entrypoint） | 新建六任务串行编排脚本（seed 参数化、每任务独立输出根、幂等 done marker） | READY.json run_entrypoint；任务书「任务之间严格串行」 |
| DEV-10 | `scripts/v6_ucit/resume_run.sh` 不存在（README.json resume_entrypoint） | 新建幂等恢复脚本（pending 事务清理 + 六任务重跑 + hash 校验） | READY.json resume_entrypoint；resume_commands.sh 模式 |
| DEV-13 | VizWiz/IconQA/CLEVR/Flickr30k test 路径声明 `test.json`，实际 `test_3000.json` | 正式配置中修正为 `test_3000.json`（数据本身完整，train 全部存在） | 实际文件系统核验 |

## 新增组件（commit `c165fca`）

1. `configs/v6_ucit_formal_locked.yaml`
   - config_hash：`d52111fc25b4b735`（sha256 前 16 位，排除 config_hash 行）
   - 与 handoff locked_config.yaml 工程默认值一致（除注释）；追加 tasks/training/eval 段
2. `compose/experiments/v6_task_run.py` — 通用任务 runner（task index ≥ 1）
   - S0..S11 幂等阶段；slot ID = (task_id+1)*10, +1（task0→10、task1→20/21 已验收，唯一性验证见测试）
   - 全部调用 E1..E12 已验收模块；未修改任何已验收模块
3. `scripts/v6_ucit/six_task_run.sh` / `resume_run.sh`
4. `tests/compose/test_v6_task_run_formal.py` — 15 项回归测试

## 测试基线

- 新增：15 passed（formal config 完整性、config_hash 自洽、路径存在、六任务顺序、feature 开关全关、runner 幂等、snapshot task_id 校验、slot 唯一、编排脚本存在）
- 全套：**354 passed + 14 subtests**（基线 339 + 15 新增，无回归）

## 说明

- 工作区其余未跟踪/修改文件（dual_lora 分析产物等）不在本批次范围，未纳入 commit。
- 未 push。
- Git HEAD：`c165fca`（补丁 commit 前为 `61ed787`；交接包记录的 `41e70bd` 为代码零差异的 docs-only 前置状态，见审计 DEV-4）。

## 运行中修复（seed 42 task0 冷启动阻塞 bug）

- commit：`6d6e2a2`（fix(compose): encode_images dtype match for multi-GPU DataParallel）
- 复现：seed 42 task0 冷启动，4 卡（4,5,6,7）batch 6 × accum 1 → `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`（mm_projector）
- 根因：dry-run 只验证过单卡 batch-1（README 警告明确）；4 卡 DataParallel replica 线程丢失 autocast 上下文，vision tower fp32 输出 × mm_projector bf16 权重
- 修复：`encode_images` 中将 features cast 到 projector 权重 dtype（autocast 下为 no-op）
- 回归：`tests/compose/test_multimodal_dtype_multigpu.py`（3 项）；全套 357 passed + 14 subtests
- 受影响 seed：42（task0 阶段失败，未提交任何专家，快照未生成——安全重跑；无已完成任务受影响）

## 运行中修复 2（seed 42 task0 冷启动阻塞 bug：DataParallel vs ComposeSelection）

- commit：`ca1318a`（fix(v6-ucit): launch candidate training via torchrun DDP）
- 复现：seed 42 task0 冷启动，4 卡 batch 6 → `ValueError: selection batch size 24 does not match input batch size 6`
- 根因：多卡 HF Trainer 未初始化分布式时回退 DataParallel；DataParallel 只切分 tensor 输入，自定义 ComposeSelection（batch 24）被逐字复制到每个 replica，与切分后的输入（batch 6）不匹配。dry-run 只跑过单卡 batch-1（README 明确 4 卡是建议、从未实测）
- 修复：train_v6_candidate 经 `torch.distributed.run`（DDP）启动——每个 rank 独立 forward，selection 与 input 天然一致；rank-0 保存逻辑已存在（--local_rank）；DDP 同时让 encode_images 的 dtype guard 成为 no-op
- 回归：`test_candidate_training_uses_torchrun_ddp`；全套 359 passed + 14 subtests
