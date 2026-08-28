# Oracle / Best-Set 数据资产审计（Phase A / A4）

日期：2026-08-28 ｜ 模式：只读 ｜ 审计对象：两个相关 run 的全部 oracle/bestset 资产

## 0. 总览

| 资产 | 对应池 | 专家数 | 候选集/样本 | 关键状态 |
|---|---|---|---|---|
| router-bestset run `query_cache/*.pt` | pool_0011（12 专家） | 12 | — | **可直接用**：128-d 查询向量，样本对齐已验证 |
| router-bestset run `oracle_audit/{t3,t4}` | pool_0011（12 专家） | 12 | 79（1+12+66） | **半成品**：候选集枚举在，`nll.json`（best-set 标签）从未生成 |
| router-bestset run `baseline_reproduction` | pool_0011（12 专家） | 12 | — | 完成（router/oracle 各 3000 test） |
| no_router_oracle run 全部 | **旧 pool_0009（10 专家）** | 10 | 56（1+10+45） | **不能用于 12-expert 实验** |

## 1. router-bestset run（`compose_ucit_v62_router_bestset_seed42_20260828`）

### 1.1 query_cache（可直接复用）

- `t3_train.pt` (2000)、`t3_val.pt` (200)、`t4_train.pt` (2000)、`t4_val.pt` (200)。
- 结构：`{queries: (N,128) f32, sample_ids: (N,) str "{record_id}#{question_id}", split, ordered_sample_ids_sha256, query_encoder_hash: d2bc62e…, query_encoder_provenance: {module_hash, query_dim 128, seed 42, text_dim 768, visual_dim 768}, official_test_used: false}`。
- **样本对齐已验证**：四个 sha 与修复后正式 run 的 `task{3,4}/data/router_supervision_train.json`（2000）和 `teacher_val.json`（200）**逐字节一致**；record_id 零泄漏（t4 仅有 8 张图在 train/val 之间 image 层重叠，record/question 层干净）。
- 数据来源映射：train = router_supervision_train，val = teacher_val（即本实验 Phase D 的 router-train/val split 定义一致）。

### 1.2 oracle_audit（半成品——缺标签）

- `manifest.json`：各 500 样本，split=router_train，`expert_ids=[0..11]`（与 pool_0011 一致），candidate_count=79，official_test_used=false。
- `records.json`：500 条 question 文件（IconQA train / CLEVR train），id 格式 `{record_id}#{question_id}`。
- `selections.json`：500 个 key（= records id 完全一致），每 key 79 个候选 → 专家 id 列表（`empty`、`single_0..11`、`pair_i_j` ×66）。**无任何 loss / best 字段。**
- **关键缺失**：`jobs.json` 声明输出 `oracle_audit/{t3,t4}/nll.json`（P2_T3/T4_exhaustive_500，checkpoint=pool_0011，gpus 4,5/6,7）——**文件在整个 run 目录及 `/data/ckpt/.../Hyper-LLaVA-runs/` 下都不存在**。即"每样本 79 候选 NLL → best-set 标签"从未计算。
- 500 个 oracle_audit 样本全部 ∈ 对应任务 train 2000（∩val=0），在 train 中有序散布（非前 500 切片）。

### 1.3 baseline_reproduction（R0/R3 对照证据）

| 目录 | Result | 备注 |
|---|---|---|
| router_t3 | 37.37% | 3000 test |
| router_t4 | 2.30% | 3000 test |
| oracle_t3 | 72.73% | 3000 test |
| oracle_t4 | 39.43% | 3000 test |

- baseline_audit：status=BASELINE_REPRODUCED，base_run=修复后 run，base_commit=c2ffe7fd，official_test_used_for_training=false。
- **t3**：SetExactAcc=0.224，PairTeacherRecall=0.0，SingleTeacherRecall≈0.43，route_count {0:45, 1:2955, 2:0} —— **router 从不输出 pair**。
- **t4**：SetExactAcc=0.024，PairTeacherRecall=0.0，SingleTeacherRecall=0.041/0.055，**teacher 侧 pair 率 0.619** —— router_t4 仅 2.30% vs oracle 39.43% 的直接原因。
- 这正是"one-shot key + oracle 监督"要解决的缺口，也是 R0 的对照基线。

## 2. no_router_oracle run（`compose_v62_seed42_no_router_oracle`）——不可用于本实验

三重证据证明其基于**旧 pool_0009（10 专家）**：

1. `answers.jsonl` metadata `checkpoint` = 旧 run 路径 `.../formal_seed42/task5/committed/pool_0009`，git_commit `1b2de8cc…`。
2. `nll.jsonl` 的 `candidate_expert_ids` 仅覆盖 0–9（56 候选 = 1+10+45；12 专家应为 79）。
3. `sample_oracle_summary.json`：`expert_count: 10, candidate_sets_per_sample: 56`；`sample_oracle_audit.json`：`commit_and_experts_match: true`。

- 每任务 `nll.jsonl`：3000 行 × 56 候选，loss 级完整（cache_audit: COMPLETE (LOSS-LEVEL)）；metric 级仅 task0/1 完整。
- 每任务 `answers.jsonl`：target-answer teacher-forcing oracle（`selection_source: "target_answer_teacher_forcing_oracle"`，uses_target_answer=true）——**泄漏型，非测试时方法**，仅作 10 专家旧池的对照参考。
- `fixed/final_pool/task{0..5}/test/candidate_00..55` 也是旧池。
- `combination_matrix.json`：336 行（6 任务 × 56），最大专家 id 9。
- 结论：该 run 只能说明 10 专家旧池的 oracle 上限（如 IconQA 84.43、CLEVR 86.5 loss-域 oracle），**不能**作为 pool_0011 的 one-shot key 监督数据。

## 3. 可用性清单

**可直接用：**
1. `query_cache/{t3,t4}_{train,val}.pt` —— 查询向量 + 样本对齐（one-shot key 训练输入）。
2. `oracle_audit/{t3,t4}/selections.json` + `records.json` —— 79 候选枚举 + question 文件（NLL 打分输入）。
3. `baseline_reproduction` 全部 —— R0/R3 对照。
4. `frozen_expert_checksums_before.json` —— pool_0011 冻结证据（key 元数据绑定依据）。

**必须重算：**
1. **pool_0011 上的每样本 79 候选 exhaustive NLL（best-set 标签）** —— 这是 Phase C 监督标签的唯一正确来源；oracle_audit 的 nll.json 从未落地。
2. **task0/1/2/5 的查询向量缓存**（现有 feature 缓存 `taskN/features/` 可复用，query encoder frozen 已验证，hash 一致即可对齐）。
3. **router-val split 的 oracle 标签**（Phase G 阈值选择用）。

## 4. 发现的不一致（均不阻断，但须记录）

1. Run 2 全链路池代差：10 专家旧池，与正式 12 专家池混用风险（路径上还同时出现 `/data/ckpt/...` 与 `/home/.../experiments/runs/...` 两种写法）。
2. Run 1 oracle_audit 是"半成品"：jobs 声明了 nll.json 但从未产出。
3. Run 1 无 run_manifest.json（environment.json + baseline_audit 佐证基线 commit c2ffe7fd）。
4. query_encoder_hash（d2bc62e…）在仓库文本文件中无显式记录——复用 query_cache 的前提是实验代码 query encoder 与 c2ffe7fd 提交一致。
5. 旧辅助缓存 `compose/oracle/task1_task4_experts01_rank8_seed42/oracle_cache.jsonl`（6000 行、4 候选、旧 commit）未并入任何正式审计，仅作历史参考。
