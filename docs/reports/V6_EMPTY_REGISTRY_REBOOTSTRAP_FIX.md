# V6 空 Registry Re-bootstrap 修复 — 最终验证报告

- 日期：2026-08-07
- 分支：`exp/compose-root-cause-diagnosis`
- 修复对象：V6 UCIT 正式持续训练的两个结构性问题
  - **问题 1**：Empty Expert Registry 是吸收态（task0 拒绝候选 → task1-5 永远不再训练 candidate）
  - **问题 2**：Rejected Candidates 通过 `_old_expert_checkpoint_chain`（cold_start /
    `last_known_checkpoint.json` fallbacks）被加载为 task1-5 的默认 adapter
- 前序审计：`docs/reports/v6_fix_empty_registry_audit.md`（Stage R0，5 个 findings P1-P5）

## 0. 结论摘要

| 问题 | 状态 | 一句话结论 |
|------|------|-----------|
| 问题 1（吸收态） | 已修复 | 空 registry → base-only teacher（真实 base NLL）→ 256 样本全部进 residual → re-bootstrap 训练 2 个 candidate → 验证 → 拒绝（below_tau）→ 隔离；状态机 COMPLETED |
| 问题 2（rejected fallback） | 已修复 | `_old_expert_checkpoint_chain` / `_old_expert_checkpoint_or_cold_start` 已删除；task1 以 `mode=rebootstrap`（无 old-expert flag）运行；rejected adapter 出现在任何 selected_experts / teacher / eval 集合中的路径为零 |

**A. registry 为空时后续任务还能训练 Candidate？ = YES**（smoke 实测：task1 residual=256 → candidate training 触发并完成）
**B. rejected Candidate 还有任何正式执行路径？ = NO**（smoke 实测：`active_lifecycle_ids=[]`、router pool=0、eval `expert_ids=[]`）

---

## 1. 16 项验证问题

### Q1. 空 registry 是否仍是吸收态？
**否，已修复。** 空 registry 不再短路 residual/candidate 流程：
- task0 提交 0 个专家 → task1 读取 snapshot 后 `active_lifecycle_ids=[]`。
- task1 不再走"empty teacher → 全部跳过"路径，而是 S1 对 base-only pool 做真实 NLL 评分
  （256 个样本，selections 仅 `empty` 键），S3 全部样本进入 residual（`residual_count=256`），
  S4 candidate training 重新启动。
- 实测证据：`residual/summary.json` → `{"base_only_mode": true, "residual_count": 256, "reuse_count": 0, "teacher_empty_count": 256}`。

### Q2. rejected candidate 是否还有任何正式执行路径？
**否。** 全链路核验（代码 + 实测）：
- `compose/experiments/v6_task_run.py` 中 `_old_expert_checkpoint_chain` 与 `_last_known_checkpoint_dir`
  已删除；`v6_task2_dry_run.py` 的 `_old_expert_checkpoint_or_cold_start` 已删除。
- teacher search / Router / selected_experts / composition / RMS / evaluation / residual baseline
  全部只从 `registry.active_lifecycle_ids()`（lifecycle ∈ {provisional, formal}）解析。
- smoke 实测：task1 `mode.json` 记录 `old_expert_checkpoint: null`、`mode: "rebootstrap"`；
  eval `load_summary.evaluation_selection = {"expert_ids": [], "gates": []}`，加载 adapter 张量数 = 0。

### Q3. R1 统一 active expert view 是否唯一且正确？
**是。** `ExpertRegistry.get_active_experts()`（lifecycle ∈ {provisional, formal}）是正式 pool 的
唯一来源；`get_rejected_candidates()` / `get_all_artifacts()` 仅用于诊断。`state_dict` 新增
`active_lifecycle_ids` / `rejected_candidate_ids`，`load_state_dict` 校验两值一致（旧 checkpoint
无该字段时按 lifecycle 推导，向后兼容）。regression 修复：同名 `active_expert_ids` property
（flag-based）与新方法冲突 → 方法改名 `active_lifecycle_ids()`，9 处 compose 调用 + 11 处测试调用同步。

### Q4. R2 rejected artifact 是否结构化、是否入 registry？
**是。** 拒绝时写入 `rejected_candidates/task_XX/candidate_YY/`：`adapter/compose_experts.bin` +
`rejection.json`（含 `excluded_from_active_pool: true`、`commit_thresholds: {tau_gain: 0.0, tau_support: 8}`、
`checkpoint_sha256`、reason、support、mean_gain）+ `validation.json` + `manifest.json`。
注册进 registry（lifecycle=REJECTED，terminal），**不**进 active pool、**不** bump pool_version。
smoke 实测：`state/expert_registry.json` → `rejected_candidate_ids: [20, 21]`、`pool_version: 1`（未变）。

### Q5. R3 fallbacks 是否全部删除？
**是。** 删除点：
- `v6_task_run.py`：`_old_expert_checkpoint_chain`（原 L148）、`_last_known_checkpoint_dir`（原 L133）、
  `last_known_checkpoint.json` 写入逻辑（原 S11）。
- `v6_task2_dry_run.py`：`_old_expert_checkpoint_or_cold_start`（原 L81）。
- eval/teacher 的 checkpoint 只从 active experts 解析；空 registry 时从 snapshot manifest 的
  `pool_checkpoint_dir`（base-only pool）解析。
- 实测：task1 运行全程无 `--old-expert-checkpoint` flag（`mode.json` 证明），task1 目录无
  `last_known_checkpoint.json` 文件。

### Q6. R4 空 registry 的 Base teacher 是否使用真实 base NLL？
**是。** 空 registry 时写入 base-only pool checkpoint（`compose_experts.json` `experts: []` + 空 state dict），
S1 用 `v6_nll_eval` 实测 base NLL。smoke 实测：`residual/residual.json` 每条记录
`empty_loss: 2.582...`、`old_teacher_loss: 2.582...`（真实测得的 base loss，不再是旧代码的 `None`）。

### Q7. R5 residual 判定是否复用既有阈值、无新阈值？
**是。** `build_residual_records(base_only_mode=True)`：空 teacher 样本不再 `continue` 跳过，
按既有 gain-floor 判据分类——`old_gain = empty_loss − teacher_loss = 0 < min_old_gain = 0.02`
→ 不充分 → residual（`reason = "base_only_insufficient"`）。`min_residual_samples=8`、
`min_old_gain=0.02` 等正式配置阈值**全部未改**。smoke 实测：256/256 样本
`residual_reason = "base_only_insufficient"`，`reuse_count = 0`。

### Q8. R6/R7 re-bootstrap 是否复用同一 trainer、无一次性 flag？
**是。** re-bootstrap 与 residual expansion 使用同一入口 `compose/train/train_v6_candidate.py`，
唯一区别是 old-expert flag 的有无（`mode.json` 记录 `rebootstrap | residual_expansion`）。
没有引入"只允许一次"的 flag——registry 为空时每个后续任务都可再次 re-bootstrap（R7 可重复）。
smoke 实测：task1 `mode.json = {"mode": "rebootstrap", "old_expert_checkpoint": null, "active_expert_ids": []}`。

### Q9. R8 Router pool=0 是否正常落盘、不训练无意义分类器？
**是。** S7/S8 用 `get_active_experts()` 枚举；pool=0 时保存空 router checkpoint
（`kind: "v6_router"`, `expert_keys: {}`, `key_metadata.experts: []`），不视为异常。
smoke 实测：`router/router_checkpoint.pt` → `expert_keys: OrderedDict()`,
`key_metadata: {"query_dim": 128, "experts": []}`。

### Q10. R9 snapshot/resume 语义是否完整？
**是。** `v6_snapshot.py` manifest 新增 `active_expert_ids` / `rejected_candidate_ids` /
`rebootstrap_allowed` / `pool_checkpoint_dir`；resume 只从 registry lifecycle 恢复 active。
smoke 实测（task1 snapshot manifest）：`active_expert_ids: []`、`rejected_candidate_ids: [20, 21]`、
`rebootstrap_allowed: true`、`pool_checkpoint_dir: ".../candidate/base_only"`、
`pool_version: 1`。

### Q11. R10 状态机是否显式表达空 registry / 无扩张路径？
**是。** 新增 `NO_EXPANSION_REQUIRED` 阶段（RESIDUAL_READY → NO_EXPANSION_REQUIRED →
GLOBAL_TEACHER_READY），skip 原因统一为 `insufficient_residual`；`RESIDUAL_READY → GLOBAL_TEACHER_READY`
直跳被判定非法（`test_undersized_residual_direct_jump_is_illegal`）。smoke 实测 task1 走了
CANDIDATE_TRAINING → CANDIDATE_TRAINED → CANDIDATE_VALIDATED →（commit 0）→ GLOBAL_TEACHER_READY →
ROUTER_TRAINING → ROUTER_READY → RMS_READY → SNAPSHOT_READY → EVALUATION_COMPLETE → **COMPLETED**。

### Q12. 测试覆盖是否达标（15 个单元测试 + 2 条集成链 + 回归）？
**是。** `tests/compose/test_v6_empty_registry_fix.py` 22 个空 registry 专项测试 + 2 条 3-task
集成链（Task0 FAIL→Task1 PASS→Task2 正常；FAIL→FAIL→PASS）+ runner mock 测试；
`test_task_state.py` 更新（NO_EXPANSION_REQUIRED 合法路径 + 直跳非法）；`test_v6_task_run_formal.py`
315 行新增。全套件 **396 passed**（本次回归在 DBG 清理后复跑确认，见下）。

### Q13. 真实 UCIT 冒烟测试（seed 42 task0→task1）结果？
**全部通过。** 详见第 2 节。三要素：
- `residual_count = 256`
- `candidate_trigger = True`（re-bootstrap，2 slots，训练完成 loss 0.627）
- **rejected adapter 出现在 selected_experts？ = 否**（eval `expert_ids: []`、router pool=0、
  `active_lifecycle_ids: []`；候选 20/21 在 `rejected_candidates/` 下，lifecycle=REJECTED）

### Q14. 旧 degraded-chain runs 是否标记 INVALID 且安全停止？
**是。** seed 42/43/44 均已写入 `SUPERSEDED_EMPTY_REGISTRY_FIX.json`
（`status: INVALID_FOR_FINAL_V6_RESULT`, reason: "empty-registry absorbing state and
rejected-candidate fallback"）。旧链（seed_42 task3 正在 teacher search）已按纪律在原子
checkpoint（task3 state=DATA_READY，仅 s0 完成标记）处 SIGTERM 安全停止，35 个进程终止、
GPU 释放、checkpoint 完好。**未删除任何旧 run 产物。** 修复后正式重跑必须从 task0 开始
（本轮未自动启动完整 6 任务重跑）。

### Q15. 19 项禁止约束是否全部遵守？
**是。** 未修改 fix-9 seeded representative validation split（task0 snapshot 原样复用）；
未修改 commit 阈值（tau_support=8, tau_gain=0.0，`rejection.json` 中如实记录）；
未降低 mean_gain>0 / support>=8 条件（本次 0 提交即真实结果，未强制 commit）；
未修改 LoRA rank/alpha/lr/epochs（rank=8, alpha=16.0, lr=2e-4, epochs=1，见 config）；
未修改 Router 方法、RMS 组合规则、UCIT task sequence；未启用 Shadow Update / 双曲空间；
**未 push**。
说明：为使 re-bootstrap 训练在 24GB 卡上可运行，添加了训练基础设施修复——
reentrant activation checkpointing + 仅对 `embed_tokens` 重新开启 requires_grad（保持
LoRA-only 更新，optimizer 不含任何 base 参数）——这是显存适配，不属于上述任何被禁改动，
不改变训练超参与模型结构。

### Q16. 代码质量与提交纪律？
**是。** 12 项代码质量标准遵守（无 glob 目录扫描、无一次性 flag、原子写、hash 校验、
audit 记录、monkeypatch 已移除、DBG 调试代码全部清理——`grep DBG/functools` 复核为空）。
提交拆分为 3 个可审计 commits（见第 4 节）。

---

## 2. 真实冒烟测试证据（seed 42，task0（fix-9 结果）→ task1）

运行目录：`experiments/runs/v6_ucit_engineering/smoke_empty_registry_fix/seed42/`

### 2.1 阶段执行（全部 .done）

```
s0_snapshot_load → s1_teacher → s2_residual → s3_features → s4_candidates →
s5_validation → s6_commit → s7_router → s8_rms → s9_snapshot → s10_eval
```

### 2.2 关键实测数据

| 项目 | 值 | 含义 |
|------|----|------|
| 空 registry 进入 task1 | `active_lifecycle_ids=[]`，`pool_checkpoint_dir=.../candidate/base_only` | 空 registry 不再是吸收态 |
| S1 base teacher | 256 样本真实 NLL，`empty_loss=2.582`（每样本实测） | 不再是 `None` |
| residual | `residual_count=256`，`reuse_count=0`，reason=`base_only_insufficient` | 复用 gain-floor 判据，无新阈值 |
| candidate trigger | `mode=rebootstrap`，无 old-expert flag，2 slots（20/21） | re-bootstrap 启动 |
| S4 训练 | 256 样本 / 144s / train_loss 0.627 / 峰值显存 ~17.6GB | 24GB 卡上可运行（reentrant ckpt 修复后） |
| S5 验证 | 两候选 mean_gain=0.0, support=0 → `below_tau` | 阈值未动，真实拒绝 |
| S6 commit | `committed_expert_ids=[]`，reason=`all_candidates_below_tau`，pool_version=1 | 未强制 commit |
| rejected 隔离 | `rejected_candidates/task_01/candidate_{20,21}/`（adapter + rejection.json + validation.json），lifecycle=REJECTED | 诊断 artifact，不入 active |
| S7 Router | `expert_keys={}`，`experts: []` | pool=0 正常落盘 |
| S9 snapshot | `active_expert_ids=[]`、`rejected_candidate_ids=[20,21]`、`rebootstrap_allowed=true` | resume 语义完整 |
| S10 eval | 3000 样本，`expert_ids=[]`（backbone-only），3.12 samples/s，峰值 15.5GB | **rejected adapter 未参与评估** |
| 状态机 | `COMPLETED`（合法路径） | R10 语义成立 |

### 2.3 问题 2 的反例核验

- task1 全程无 `last_known_checkpoint.json`（旧链路传播指针已删除）。
- `selected_experts` 相关三处全部为空：teacher selections（`old_teacher_set=[]`）、
  router `expert_keys={}`、eval `evaluation_selection.expert_ids=[]`。
- rejected 20/21 的 checkpoint 只存在于 `rejected_candidates/` 下，任何正式模块的
  checkpoint 解析路径均不覆盖该目录。

## 3. 代码改动清单

| 文件 | 内容 |
|------|------|
| `compose/experts/metadata.py` | `ACTIVE_LIFECYCLE_STATUSES = {provisional, formal}` |
| `compose/experts/registry.py` | `get_active_experts` / `get_rejected_candidates` / `get_all_artifacts` / `active_lifecycle_ids`；state_dict 扩展 + load 校验 |
| `compose/expansion/v6_rejected.py`（新） | R2 rejected artifact 结构 + `register_rejected_candidate` |
| `compose/expansion/v6_base_pool.py`（新） | R4 base-only pool checkpoint 写入（`experts: []`） |
| `compose/expansion/v6_residual.py` | R5 `base_only_mode` + `RESIDUAL_REASON_BASE_ONLY` |
| `compose/experts/task_state.py` | R10 `NO_EXPANSION_REQUIRED` 阶段 |
| `compose/experiments/v6_snapshot.py` | R9 manifest 新字段 |
| `compose/experiments/v6_task1_dry_run.py` | R2 接入（task0 拒绝时写 rejected artifact + 注册） |
| `compose/experiments/v6_task2_dry_run.py` | R2/R3/R4/R6/R8（删除 cold_start fallback，base-only teacher） |
| `compose/experiments/v6_task_run.py` | R2/R3/R4/R6/R8（删除 chain/last_known，S7 save_atomic 修复） |
| `compose/train/train_v6_candidate.py` | S4 显存修复：reentrant ckpt + embed_tokens grad anchor（LoRA-only 更新不变） |
| `compose/train/trainer.py` | training_step 内保持 use_selection 上下文（backward recompute 确定性） |
| `tests/compose/test_v6_empty_registry_fix.py`（新） | 22 个专项测试 + 2 条集成链 + runner mock |
| `tests/compose/test_task_state.py` / `test_v6_task_run_formal.py` | 状态机 + runner 行为测试更新 |
| `docs/reports/v6_fix_empty_registry_audit.md`（新） | Stage R0 审计报告 |
| `docs/reports/V6_EMPTY_REGISTRY_REBOOTSTRAP_FIX.md`（新） | 本报告 |

## 4. 提交纪律（3 个可审计 commits，未 push）

1. **core**：registry lifecycle 统一视图 + rejected 隔离 + base pool + residual base_only +
   状态机 NO_EXPANSION_REQUIRED + snapshot 字段（R1/R2/R4/R5/R9/R10）
2. **runners**：三个 runner 接入 + fallback 删除 + re-bootstrap orchestration +
   S4 显存基础设施修复（R3/R6/R7/R8 + trainer）
3. **tests-docs**：全部测试 + 两份报告

> 未 push 到远端；未自动启动 seed 42/43/44 完整 6 任务重跑（从 task0 开始的重跑需另行发起）。
