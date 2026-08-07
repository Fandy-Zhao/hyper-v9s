# V6 空 Registry / Rejected Candidate 结构问题审计报告（Stage R0）

- 日期：2026-08-07
- 分支：exp/compose-root-cause-diagnosis
- 审计目标：修复 Empty Registry 吸收态（re-bootstrap 缺失）与 Rejected Candidate
  继续作为后续任务默认 adapter 的生命周期冲突。本轮只修这两个结构问题。

## 1. 结论摘要

| # | 问题 | 严重性 | 根因 |
|---|------|--------|------|
| P1 | registry == empty 成为不可恢复吸收态 | 致命 | S1 空 registry 分支写出 `teacher_set=()` 记录后，S3 residual 构建把所有空 teacher 样本直接跳过 → residual=0 → S5 跳过 Candidate → registry 永远为空 |
| P2 | rejected task0 candidate 继续被 task1–5 加载 | 严重 | `_old_expert_checkpoint_chain` / `_old_expert_checkpoint_or_cold_start` 把 `prev_root/candidate/cold_start` 与 `last_known_checkpoint.json` 指针作为 fallback adapter 解析来源，S11 eval 与 S1 teacher search 均使用 |
| P3 | 无统一 active expert view | 严重 | 各 runner 用 `registry.list_all()`（含 candidate/rejected）构造 teacher/Router 候选集；没有按 lifecycle status 过滤的接口 |
| P4 | rejected 无结构化 artifact / 不入 registry | 中 | 拒绝时只写 `committed/rejected_<id>.json` 一行；snapshot 无法区分 active/rejected；`rejected_candidates/` 目录不存在 |
| P5 | eval 空 registry 时无合法 checkpoint 解析 | 中 | S11 在无 candidate pool 时要么 RuntimeError、要么加载 rejected adapter |

## 2. 涉及的文件与函数

### 2.1 Expert Registry 实现

- `compose/experts/registry.py`：`ExpertRegistry`（OrderedDict[int, ExpertMetadata]），
  `_active_ids` / `_trainable_ids` / `_pool_version`；`list_all()` / `list_active()`；
  lifecycle 迁移 `mark_provisional` / `mark_formal` / `mark_rejected`。
- `compose/experts/metadata.py`：`ExpertLifecycleStatus`（candidate / provisional /
  formal / archived / rejected）；`ExpertMetadata.__post_init__` 在无显式
  lifecycle 时按 `checkpoint_path` 推断（有 checkpoint → PROVISIONAL）。
- `compose/experts/transaction.py`：`CommitTransaction`（pending → artifacts →
  registry(provisional) → pool_version bump → marker 清除），拒绝时无任何调用。

### 2.2 Candidate commit / reject 逻辑

- `compose/experiments/v6_task1_dry_run.py` S4（task0）：below_tau 时只写
  `committed/commit_record.json`，**不写 rejected artifact、不注册 rejected 状态**。
- `compose/experiments/v6_task2_dry_run.py` S6（task1）、`compose/experiments/v6_task_run.py`
  S7（task2–5）：below_tau 时写 `committed/rejected_<slot>.json`（仅 slot_id/
  reason/stats），同样不入 registry。

### 2.3 task0 cold-start adapter 保存逻辑

- `v6_task1_dry_run.py` S2：`candidate/cold_start/`（train_v6_candidate 输出，
  含 `compose_experts.bin` / `compose_experts.json` / `candidate_10.pt`）。
  S3 验证、S4 按 `mean_gain>0 and support>=tau_support` 提交 0 或 1 个。
  该目录**无论提交与否都保留在磁盘**，成为下游 fallback 的来源。

### 2.4 task1–5 degraded-chain 入口

- task1：`v6_task2_dry_run.py::_old_expert_checkpoint_or_cold_start`（L81）：
  `old_ids` 非空 → `task1_root/committed/expert_XXXX`，否则 →
  `task1_root/candidate/cold_start`（**rejected adapter**）。
- task2–5：`v6_task_run.py::_old_expert_checkpoint_chain`（L148）：candidate/train →
  prev committed → **`prev_root/candidate/cold_start`** → **`_last_known_checkpoint_dir`**。
- `v6_task_run.py::_last_known_checkpoint_dir`（L133）：读
  `candidate/last_known_checkpoint.json` 指针文件（由 S11 eval fallback 写入，
  L952–958），把 rejected adapter 继续向更下游传播。

### 2.5 registry empty 时的分支

- `v6_task_run.py` S1 L300–309：`old_checkpoint is None` →
  `_write_empty_teacher_records`（L164）：`teacher_set=()`、`empty_loss=None`、
  `teacher_loss=None`；summary 注明 "teacher search skipped"。
- `v6_task_run.py` S5 L650–659：`if not residual:` → 写 `no_candidate.json`，
  状态机 CANDIDATE_TRAINING → CANDIDATE_TRAINED → CANDIDATE_VALIDATED 全部
  "skip"，`commit_count=0` → registry 保持空 → 下一个任务重复。

### 2.6 teacher search 的 empty-registry 处理

- `v6_task2_dry_run.py` S1：空 registry → selections 只有 `empty` →
  所有样本 `teacher_set=()` → S2 residual 0。
- `v6_task_run.py` S1：`_write_empty_teacher_records` → residual 0。
- 即：**empty registry 分支跳过了 teacher loss 的测量**（`empty_loss=None`），
  residual 判定无从谈起，等于直接短路。

### 2.7 residual buffer 构建逻辑

- `compose/expansion/v6_residual.py::build_residual_records`（L203）与
  `is_residual`（L181）：`if not record.teacher_set: continue`（L237）——
  **空 teacher 样本一律跳过，既不进 reuse 也不进 residual**。
  这是吸收态的第二个关键环节：即使 teacher 记录存在，只要 teacher_set 为空
  就进不了 residual 材料。

### 2.8 Candidate trigger 条件

- `v6_residual.py::should_create_candidates`（L253）：`residual_count >= min_residual_samples`
  （正式配置 8）。阈值本身正确，问题在上游 residual 恒为 0。

### 2.9 inference / evaluation adapter resolution

- `compose/eval/eval_task.py` L94–124：`--expert-ids ""` →
  `clear_default_selection()` → backbone-only；`--expert-ids x,y` → 固定选择。
  空集合推理路径（`ComposeSelection` PAD 行）已验证存在
  （`compose/adapters/types.py`、`compose/adapters/manager.py::make_selection`）。
- `compose/eval/load_compose.py::load_compose_model`：`expert_id=None` 时
  `manager.clear_default_selection()`；checkpoint 目录仍需
  `compose_experts.json`（adapter rank/alpha/layers）。
- `compose/experts/checkpoint.py::load_expert_checkpoint`：manifest 校验
  `format_version==1` 与 layers 一致性；`experts: []` + 空 state dict 可正常加载
  （`_expected_keys` 为空集合），**空专家 checkpoint 目录格式可行**。

### 2.10 Router expert enumeration

- `v6_task_run.py` S8 L879–885：`for expert in registry.list_all()` —— 包含
  candidate/rejected 状态；应改为 active view。
- `v6_task2_dry_run.py` S7 L498–504：同样 `registry.list_all()`。
- `v6_task1_dry_run.py` S5：按 `commit_record["committed_expert_ids"]` 添加，
  未提交时 router 为空（正确，但无 empty-router checkpoint 说明）。

### 2.11 RMS expert enumeration

- 当前 S9 为占位（`RMS_READY` 直接标记），无实际枚举；fix 后应保证 RMS
  只对 active experts 计算（与 R1 接口一致）。

### 2.12 snapshot 中 adapter 列表

- `compose/experiments/v6_snapshot.py`：manifest 只含 pool_version / task_id /
  hashes；registry 完整 JSON（含每 expert 的 lifecycle_status）写入
  `expert_registry.json`。**没有 rejected_candidate_ids 等显式区分字段**。

### 2.13 resume 时 Candidate 状态恢复

- `ExpertRegistry.load_state_dict`：恢复 experts + active/trainable ids；rejected
  专家（若有）会进入 `_experts` 但不在 `_active_ids`。问题在于当前没有任何
  机制把 rejected 记进 registry，因此恢复时无从区分。
- `v6_snapshot.analyze_resume`：只做 pending transaction 清理与 stage 映射，
  不涉及 lifecycle。

### 2.14 哪段代码导致 rejected task0 adapter 被继续加载

真实路径（seed 42，已核实磁盘产物）：

```
task1 S1 teacher search:
  v6_task2_dry_run._old_expert_checkpoint_or_cold_start(task0, old_ids=[])
    -> task0/candidate/cold_start            （rejected adapter）
task1 S10 eval:
  checkpoint_dir = task1/candidate/train 不存在
    -> task1_root/candidate/cold_start         （rejected adapter）
    -> 写 task1/candidate/last_known_checkpoint.json
task2 S1 teacher search:
  v6_task_run._old_expert_checkpoint_chain(task1, registry=[])
    -> task1/candidate/train 不存在
    -> task1/committed/expert_* 不存在
    -> task1/candidate/cold_start 不存在
    -> task1/candidate/last_known_checkpoint.json
        -> task0/candidate/cold_start          （rejected adapter，跨任务传播）
task2..5 S11 eval: 同 chain fallback + last_known_checkpoint.json 继续写
```

证据（seed_42/task1/candidate/last_known_checkpoint.json）：
`{"checkpoint_dir": ".../seed_42/task0/candidate/cold_start", "note": "degenerate chain: eval against previous task cold start"}`

## 3. 空 registry 为什么导致 candidate path 被跳过

因果链：

1. `registry.list_all()` == []（task0 below_tau，0 提交）。
2. S1 `old_checkpoint = _old_expert_checkpoint_chain(prev, registry)` 命中
   cold_start / last_known 指针 → 但 `old_ids == []`：
   - `v6_task_run` 分支：`_write_empty_teacher_records` → `teacher_set=()`
   - `v6_task2_dry_run`：`single_loss = None` → `teacher_set=()`
3. S3 `build_residual_records`：`if not record.teacher_set: continue` →
   `residual=[]`、`reuse=[]`。
4. S5 `if not residual:` → 跳过 candidate training（"no residual -> skip"）。
5. S7 commit 空、S8 Router 空、S11 eval fallback 到 rejected adapter。
6. task2 读取 task1 snapshot → registry 仍空 → 重复 2–5。

即："没有旧专家"被实现成了"不能训练新专家"。正确语义应为：registry 空 →
当前能力基线 = Frozen Backbone → 用 Base loss 判定样本是否充分 →
不充分的样本进入 residual → residual 足够时重新启动 Candidate training（re-bootstrap）。

## 4. 修复前旧 run 处置（2026-08-07 实况）

- `scripts/v6_ucit/resume_run.sh 42` 监督的 seed_42 旧链仍在运行 task3
  （S1 teacher search），已被 SIGTERM 安全停止于原子 checkpoint
  （task3 state = DATA_READY，仅 `s0_snapshot_load.done`，无半写 stage）。
- 已写 superseded 标记（status INVALID_FOR_FINAL_V6_RESULT，
  reason "empty-registry absorbing state and rejected-candidate fallback"）：
  - `experiments/runs/v6_ucit_engineering/formal/seed_42/SUPERSEDED_EMPTY_REGISTRY_FIX.json`
  - `.../seed_43_INVALID_split_bias/SUPERSEDED_EMPTY_REGISTRY_FIX.json`
  - `.../seed_44/SUPERSEDED_EMPTY_REGISTRY_FIX.json`
- 修复后正式重跑必须从 task0 开始（task0 commit/reject 决定 task1 registry
  与整个后续轨迹），不得从旧 task1/task2 snapshot 续跑。

## 5. 修复方案清单（对应 Stage R1–R10）

| Stage | 动作 |
|-------|------|
| R1 | `ExpertRegistry.get_active_experts()`（lifecycle ∈ {provisional, formal}）+ `get_rejected_candidates()` + `get_all_artifacts()`；所有正式模块改用它 |
| R2 | 拒绝时结构化 artifact：`rejected_candidates/task_XX/candidate_YY/{adapter,key,validation.json,rejection.json,manifest.json}`；注册进 registry（lifecycle=REJECTED，terminal，不 active、不 bump pool_version） |
| R3 | 删除 `_old_expert_checkpoint_chain` 的 cold_start / last_known fallback 与 `_last_known_checkpoint_dir`；eval/teacher 的 checkpoint 只从 active experts 解析 |
| R4 | registry 空时 teacher = base-only：新增空专家 checkpoint 目录（`compose_experts.json` + 空 bin），teacher_set=()、teacher_loss=base_loss（真实测得） |
| R5 | `build_residual_records(base_only_mode=True)`：空 teacher 样本不再跳过，按既有 gain-floor 判据（old_gain=0 < min_old_gain → 不充分 → residual），不新增阈值 |
| R6 | residual >= min_residual_samples 时进入同一 Candidate Trainer（无 old checkpoint 即为 re-bootstrap）；`mode=rebootstrap | residual_expansion` 记入日志/JSON |
| R7 | 不引入一次性 flag；registry 空时每个任务都可重试 re-bootstrap |
| R8 | Router 只枚举 active experts；pool=0 → 保存 empty-router checkpoint，不视为异常 |
| R9 | snapshot 增加 `active_expert_ids` / `rejected_candidate_ids` / `rebootstrap_allowed`；resume 只从 registry lifecycle 恢复 active |
| R10 | 状态机增加 `NO_EXPANSION_REQUIRED` 阶段；skip 原因统一为 `insufficient_residual`，不再有 `empty_registry` 语义 |
