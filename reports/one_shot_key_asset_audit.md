# One-shot Learnable Expert Key —— Phase A 资产审计报告（GO/NO-GO 判定）

日期：2026-08-28 ｜ 阶段：Phase A（只读审计，未修改任何代码/checkpoint/run）
规范：One-shot Learnable Expert Key 可行性验证（18 节）
目标：确认"Frozen Query + Frozen Expert Pool + Oracle 监督 → 学一次 key → 永久冻结 → cosine 检索路由"的全部前提资产就绪，并明确必须补算的部分。

---

## 1. 审计范围与证据来源

| 审计项 | 内容 | 结果 |
|---|---|---|
| A1 | git 版本核对（正式 run 代码基线、本实验代码基线） | 完成 |
| A2 | 正式 expert pool 完整性（12 专家 LoRA checkpoint 逐项 sha256） | 完成（子代理，见 audit/expert_pool_audit.md） |
| A3 | query/router 资产（query encoder frozen、historical keys frozen、router 结构、key-only 实现） | 完成 |
| A4 | oracle/bestset 资产（两份 run 全部 oracle 缓存） | 完成（子代理，见 audit/oracle_asset_audit.md） |
| A5 | R0 正式结果（6-task matrix） | 完成 |
| A6 | 数据 splits、train/val NLL cache 覆盖、标签缺口 | 完成 |

证据文件：`audit/expert_pool_manifest.json`、`audit/expert_pool_audit.md`、`audit/oracle_asset_audit.md`。

---

## 2. 强制审计要点逐一回答

### 2.1 选择哪个正式 expert pool？为什么？

**`compose_ucit_v62_formal_seed42_repaired_20260823/task5/committed/pool_0011`（12 专家，全 frozen）。**

- 它是 V6.2 repaired 正式实验的**最终累积池**：expert 0–11，`status=frozen`、`lifecycle_status=formal`、`active 12/12`。
- 它是后续所有任务（task5 结束时）实际使用的池；R0/R3 对照、RMS calibration 均基于它。
- 它的 bin sha256（`0c2b3a63…`）被 router-bestset run 的 `frozen_expert_checksums_before.json` 再次确认冻结。

### 2.2 为什么选择它（vs no_router_oracle 的 pool_0009）？

- pool_0009 是**旧 run 的 10 专家中间池**（commit 1b2de8cc），比 pool_0011 少 expert 10/11；本实验要验证的是"12 专家正式池"上 one-shot key 能否逼近 oracle，必须用与 R0/R3 同一池。
- pool_0009 相关 oracle 资产（56 候选）只能作旧池对照参考，不能混入训练/评测。

### 2.3 expert 数量和创建顺序？

**12 个，连续无缺。创建顺序（key 冻结顺序）：0 → 1 → 2 → 3,4,5,6 → 7,8,9,10 → 11**（按 creation_task 0→5）。task3 新增 4 个、task4 新增 4 个、task5 新增 1 个，权威记录在 `task5/snapshots/task5/expert_registry.json`（pool_version=13）。

### 2.4 每个 Expert LoRA checkpoint？

- 每个专家有 `taskN/committed/pool_XXXX/compose_experts.bin`（累积池）与 `taskN/lora/cluster_training/expert_XXXX.pt`（新增专家 adapter）。
- 12 个 pool 的 bin 实测 sha256 与 assembly.json / snapshot registry / expert_pool_after lora_hash **三方一致**（12/12 PASS）。
- registry 中每专家 checkpoint_path 指向其所在 task 合入后的最终 pool 目录（task3→pool_0006、task4→pool_0010、task5→pool_0011），目录均存在、sha 均匹配。
- torch 2.3.1 实际加载验证：5376/5376 张量，missing 0 / unexpected 0（239,861,760 参数，rank-8 bf16）。
- 权威 per-expert 元数据（checkpoint_sha256/creation_task/rank/alpha）须从 snapshot registry 读取（pool JSON 中为 null）。

### 2.5 Query Encoder 来源？

- 正式 router 的 `ComposeRouter.query_encoder`（768→128），`module_hash=d2bc62e…`，seed 42，text/visual dim 768。
- **跨任务 frozen 已验证**：module_hash 全任务相同 + 权重 torch.equal 全 True。
- **historical keys frozen 已验证**：keys.0 在 task0 与 task5 快照中相同。
- 已缓存 128-d 查询向量：`query_cache/{t3,t4}_{train,val}.pt`（t3/t4 = IconQA/CLEVR-Math 的 router-train 2000 + val 200），sample_ids 与正式 run 数据文件逐字节 sha 对齐、零泄漏。

### 2.6 Current Router checkpoint？

- `task5/keys/final_router_checkpoint.pt`：query-key cosine + expert keys + 学习式 cardinality head（set_router），top_m=8、max_active=2、pair_threshold=0.65。
- R0 最终矩阵（6-task）：**83.00 / 92.77 / 51.82 / 37.37 / 2.30 / 54.40**，MFN 53.61，BWT −2.492。
- **关键事实**：正式 router 并非"纯 MLP 路由"——它已是 query-key cosine 检索 + cardinality 判定；用户"无 MLP router"的 one-shot key 方案与已存在的 `exp/v62-router-best-set-supervision` 实现（key_only_bestset.py：frozen query + learnable keys/bias/temperature）同构，**该实现可复用/扩展，R1 尚未运行**。

### 2.7 哪些 Oracle cache 可以直接复用？

1. `query_cache/{t3,t4}_{train,val}.pt` —— 查询向量 + 样本对齐（Phase D/F 训练输入）。
2. `oracle_audit/{t3,t4}/selections.json` + `records.json` —— pool_0011 上 79 候选枚举 + question 文件（NLL 打分输入）。
3. `baseline_reproduction`（router_t3 37.37 / oracle_t3 72.73 / router_t4 2.30 / oracle_t4 39.43）—— R0/R3 对照。
4. `frozen_expert_checksums_before.json` —— pool_0011 冻结证据。
5. R3 已有正式 pool oracle 矩阵（88.97 / 94.97 / 51.17 / 72.73 / 39.43 / 52.38）。

### 2.8 哪些必须重新计算？

1. **pool_0011 上的每样本 79 候选 exhaustive NLL（best-set 监督标签）** —— oracle_audit 的 `nll.json` 在 jobs.json 中声明但**从未产出**；现有 train/val NLL caches 是 teacher_train（40000）上的 empty+singles + top-6 pairs（11 old experts 近似），**不是** router-train split 上的 12-expert 穷举。这是 Phase C 的唯一正确标签来源。
2. **task0/1/2/5 的查询向量缓存** —— 目前只有 t3/t4。可从 `taskN/features/` 缓存用 frozen query encoder 提取（hash 一致即可对齐）。
3. **router-val split 的 oracle 标签**（Phase G 全局阈值 tau_empty/tau_pair 选择用）。
4. expert-11 的 singles/pairs NLL（并入第 1 项）。

### 2.9 train / val / test 是否完整？

- splits：router_supervision_train 2000/任务、teacher_train 40000、teacher_val/validation/calibration 200/任务、test 3000/任务，6 任务全部存在。
- **泄漏检查 PASS**：data_contract 验证 router_train/val 与 official test 的 record_id 零重叠；query_cache sha 对账一致；t4 仅 image 层 8 张图 train/val 重叠（record/question 层干净）。
- 不完整处：仅 t3/t4 有查询缓存（其余 4 任务待补）；12-expert 穷举 NLL 标签不存在（待补）。

### 2.10 是否发现 checkpoint / sample-id / embedding 不一致？

- **未发现**阻断性不一致。记录在案的：
  1. no_router_oracle 全链路为旧 pool_0009（10 专家），与本实验正式池不同代（已排除）。
  2. pool 的 compose_experts.json 中专家级字段为 null（权威值在 registry）。
  3. committed 与 cluster_training 的 JSON 视图不同（bin 相同）。
  4. query_encoder_hash 在仓库文本中无记录（复用 query_cache 的前提是与 c2ffe7fd 提交一致）。
  5. oracle_audit 500 样本 ⊂ train 2000（散布位置，非前 500 切片）——可作 Phase C 抽检子集。

---

## 3. NO-GO 条件逐项检查

| 用户规定的 NO-GO 条件 | 判定 |
|---|---|
| 专家 LoRA 不完整 | **否**（12/12 完整、sha 三方一致、可加载） |
| expert-registry 对应关系无法确认 | **否**（registry 与磁盘逐项匹配） |
| oracle 数据来自不同 expert checkpoint | **否**（usable assets 全部来自 pool_0011；旧池资产已隔离） |
| train/test 泄漏 | **否**（data_contract + sha 对账 PASS） |
| query feature 无法和 sample_id 对齐 | **否**（query_cache sample_ids 与正式数据逐字节一致） |
| 专家形成顺序无法恢复 | **否**（creation_task 完整：0→1→2→3×4→4×4→5） |

**全部 6 项 NO-GO 条件均不触发 → 判定 GO。**

---

## 4. GO 判定与 Phase B+ 前提清单

**VERDICT: GO**

Phase B+ 可开始，但必须先完成以下补算（按依赖顺序）：

1. **Phase B（oracle 基准）**：在选定 split 上评估 P_base / P_best_single / P_oracle / P_current_router。R0/R3 已有 6-task test 数字；需补充 router-train/val split 上的 oracle 集合分布（empty/single/pair 率）作为监督标签统计。
2. **Phase C（监督标签）**：在 pool_0011 上对 router-train split（2000×6）计算 79 候选 exhaustive NLL → S_i* ∈ {empty, {E_a}, {E_a,E_b}}，输出 `cache/one_shot_key/train_labels.jsonl`。可复用 oracle_audit 的候选枚举与 records 格式；500 样本抽检子集先行验证流水线。
3. **Phase D（查询缓存补齐）**：提取 task0/1/2/5 的查询向量（frozen query encoder，hash 校验），与 t3/t4 现有缓存合并为 6-task 全量。
4. 之后按规格进入 Phase E（centroid baseline R1）→ F（one-shot learnable R2，sequential key freezing）→ G（纯 cosine 检索 + 全局 val 阈值）→ 指标矩阵 + continual simulation + 专项分析 → 最终报告。

**实现复用**：`exp/v62-router-best-set-supervision` 分支的 `key_only_bestset.py`（key_only mode、multilabel targets、probabilistic selection）与 `test_key_only_bestset.py`（4 个现成测试）作为 Phase E/F 基础，新增：centroid init baseline、sequential per-expert one-shot 训练（previous keys frozen）、6-task teacher labels、6-task query cache、continual simulation + OldToNewHijackRate、完整指标。

**Git 管理**：新建独立 branch/worktree `exp/v63-one-shot-frozen-key`（不从正式实验 branch 直接修改）；三个 commit：① audit（本次产出）② implementation + tests ③ experiment configs。大型 checkpoint 不入库。

**GPU**：仅物理 GPU 4、5、6、7。

**下步动作**：等待用户批准后创建 branch/worktree 并开始 Phase B。
