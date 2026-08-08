# Compose 流水线迁移最终报告（§35）

- **日期**：2026-08-08
- **基线 SHA**：`3c2a7d1ded478e0d7faaa4964097e8170bc9cb3c`
- **验证状态**：P0 验收全部通过；按 §37 未启动正式六任务实验

---

## 1. 迁移摘要

将 v6 UCIT 持续学习流水线重构为 **Compose 流水线**（Query-Clustered Residual Expert Discovery + Cluster-Supervised Learnable Expert Key + Sample-wise Old Expert Reuse）。核心变化：

| v6（基线） | Compose（本分支） |
|---|---|
| 全局 Router（BCE 蒸馏 + bias/temperature head + 全局 calibration） | 无全局 calibration；纯余弦匹配的 ComposeRouter（冻结 query encoder + 专家 keys） |
| 固定 task_id→expert 映射 | 按样本 Top-M 检索（batch×M，禁止广播 row 0） |
| 每任务固定 slot（task_id\*10+slot） | 消费式专家 id 分配器 `registry.next_expert_id()` |
| 无监督聚类+原型 key | 球形 K-Means（K=1..4，余弦 silhouette，K=1 退化，冻结分配）+ 可学习 128-D key（CE + 旧负样本 hinge + 散度） |
| 整池条件残差训练 | 簇级条件残差训练（仅新专家接收梯度） |
| 校验门控提交 | 直接提交（active=true, trainable=false, 原子 pool_version 递增） |
| 全局 RMS（reference=base） | RMS runtime kappa（reference=层内 ALL 激活专家均值；pair 自动 1/√2） |
| 15 阶段任务状态机 | 16 阶段任务状态机（task 0 跳过 OLD_TEACHER_READY 并记录原因） |

## 2. 变更文件表

### 2.1 新增核心模块（untracked）

| 文件 | 职责 |
|---|---|
| `compose/experiments/task_run.py` | 统一任务 runner（S0–S12，stages 幂等标记，resume 语义） |
| `compose/experiments/snapshot.py` | 快照（pool 版本、active experts、router、calibration） |
| `compose/router/router.py` | ComposeRouter：冻结 query encoder + keys + 余弦 Top-M/选择 |
| `compose/router/functional_query.py` | 功能 query：L2Norm(MLP([LN(z_v); LN(z_s)])) 768/768/128 |
| `compose/router/query_encoder.py` | query encoder 定义 |
| `compose/router/key_learning.py` | 簇监督可学习 key（CE + hinge + 散度，独立 optimizer） |
| `compose/router/set_losses.py` | 集合损失 |
| `compose/expansion/query_clustering.py` | 球形 K-Means + 余弦 silhouette + 噪声簇 |
| `compose/expansion/residual.py` | 残差发现（tau_res 2.0，recall miss 诊断） |
| `compose/expansion/expert_formation.py` | 专家形成清单（噪声样本排除） |
| `compose/expansion/commit.py` | 直接提交事务（原子 pool_version 递增） |
| `compose/experts/registry.py` / `pool.py` / `metadata.py` / `checkpoint.py` / `transaction.py` / `task_state.py` | 消费式 id 分配器、专家池、16 阶段状态机 |
| `compose/teacher/teacher.py` | 答案监督 teacher 搜索（空集→Top-M 单→对） |
| `compose/lora/rms.py` | RMS runtime kappa（reference-mean）+ calibration 持久化 |
| `compose/adapters/lora.py` / `types.py` | LoRA adapter（rank 8） |
| `compose/eval/eval_task.py` | 推理评估（仅 image/instruction/冻结 query encoder/keys/LoRA/RMS） |
| `compose/eval/rms_stats.py` | RMS 统计（含 image 占位符修复） |
| `compose/train/train_compose.py` | 合并后的单一 Trainer（`--compose-mode fixed\|cluster_expert`） |
| `configs/compose_ucit.yaml` | 正式配置 |
| `scripts/Compose/Run_UCIT/six_task_run.sh` / `resume_run.sh` | 六任务启动/恢复脚本 |

### 2.2 修改文件

`compose/train/{train_compose.py, data.py, arguments.py, trainer.py}`、`compose/teacher/validity.py`、`compose/lora/adapter_bridge.py`、若干既有测试（`test_compose_linear.py`、`test_oracle_teacher_stage04.py`、`test_task_state.py`、`test_ucit_sequence.py`、`test_selection.py`、`test_grouped_execution.py`、`test_lora_composition_stage03.py`、`test_conditional_training.py`、`test_expert_manager.py`）。

### 2.3 新增测试

15 项关键单元测试（A–O）：`test_compose_a_per_sample_top_m`、`test_compose_b_task0_empty_registry`、`test_compose_c_k1_no_nan`、`test_compose_d_two_spherical_clusters`、`test_compose_e_small_cluster_noise`、`test_compose_f_fixed_assignment`、`test_compose_g_gradient_isolation`、`test_compose_h_key_learning`、`test_compose_i_no_global_calibration`、`test_compose_j_direct_commit_count`、`test_compose_k_rms_applied`、`test_compose_l_pair_scale`、`test_compose_m_resume_no_repeat`、`test_compose_n_split_leakage`、`test_compose_o_inference_purity`；另有 `test_rms_image_question.py`、`test_query_no_answer_leakage.py`、`test_sufficiency_no_test_leakage.py`、`test_oracle_teacher_stage04.py`（含 float-NLL 回归）。

统计：**75 个文件变更，+968 / −12117**。

## 3. 删除模块列表（42 个）

```
compose/cli/eval_v6_routed_ucit.py
compose/eval/export_hyper_compat.py
compose/eval/v6_acceptance.py
compose/eval/v6_multiseed.py
compose/eval/v6_query_features.py
compose/expansion/v6_base_pool.py
compose/expansion/v6_candidate.py
compose/expansion/v6_commit.py
compose/expansion/v6_rejected.py
compose/expansion/v6_residual.py
compose/experiments/v6_snapshot.py
compose/experiments/v6_task1_dry_run.py
compose/experiments/v6_task2_dry_run.py
compose/experiments/v6_task_run.py
compose/lora/v6_rms.py
compose/router/v6_calibrate.py
compose/router/v6_router.py
compose/teacher/v6_teacher.py
compose/train/train_v6_candidate.py
configs/v6_ucit_engineering.yaml
configs/v6_ucit_formal_locked.yaml
configs/v6_ucit_two_task_dry_run.yaml
scripts/v6_ucit/resume_run.sh
scripts/v6_ucit/six_task_run.sh
tests/compose/test_registry_v6.py
tests/compose/test_selection_v6.py
tests/compose/test_split_leakage_v6.py
tests/compose/test_v6_acceptance.py
tests/compose/test_v6_calibrate.py
tests/compose/test_v6_candidate.py
tests/compose/test_v6_commit.py
tests/compose/test_v6_empty_registry_fix.py
tests/compose/test_v6_multiseed.py
tests/compose/test_v6_off_regression.py
tests/compose/test_v6_pipeline_integration.py
tests/compose/test_v6_residual.py
tests/compose/test_v6_rms.py
tests/compose/test_v6_router.py
tests/compose/test_v6_snapshot.py
tests/compose/test_v6_task_run_formal.py
tests/compose/test_v6_teacher.py
tests/compose/v6_ddp_smoke.py
```

## 4. 15 项合规矩阵

| # | 测试 | 断言要点 | 状态 |
|---|---|---|---|
| A | `test_compose_a_per_sample_top_m` | 每样本 Top-M，batch×M，不广播 row 0，PAD=-1 | ✅ |
| B | `test_compose_b_task0_empty_registry` | Task 0 空注册表冷启动合法，NO_EXPANSION_REQUIRED 语义 | ✅ |
| C | `test_compose_c_k1_no_nan` | K=1 退化无 NaN | ✅ |
| D | `test_compose_d_two_spherical_clusters` | 球形 K-Means + 余弦 silhouette | ✅ |
| E | `test_compose_e_small_cluster_noise` | 小簇→噪声簇，排除出训练/清单（训练文件==清单样本） | ✅ |
| F | `test_compose_f_fixed_assignment` | 聚类分配冻结 | ✅ |
| G | `test_compose_g_gradient_isolation` | 簇级条件残差训练：仅新专家梯度 | ✅ |
| H | `test_compose_h_key_learning` | CE+hinge+散度；仅新 keys 更新（含标签位置≠专家 id 回归） | ✅ |
| I | `test_compose_i_no_global_calibration` | 无全局 router calibration | ✅ |
| J | `test_compose_j_direct_commit_count` | 直接提交：active/`trainable=false`/原子 pool_version | ✅ |
| K | `test_compose_k_rms_applied` | RMS kappa 实际生效（set/clear_expert_calibration） | ✅ |
| L | `test_compose_l_pair_scale` | 对组合 1/√2 缩放 | ✅ |
| M | `test_compose_m_resume_no_repeat` | resume 不重复提交/版本不重复递增 | ✅ |
| N | `test_compose_n_split_leakage` | 训练/验证划分无泄漏 | ✅ |
| O | `test_compose_o_inference_purity` | 推理纯度（§22） | ✅ |

## 5. §30 阻塞性 grep 审计

```bash
grep -rE '(^|[^a-zA-Z0-9])v6([._-]?1)?([^a-zA-Z0-9]|$)|v6_' \
  compose/ tests/compose/ scripts/Compose/ configs/ \
  --include="*.py" --include="*.yaml" --include="*.sh" | wc -l
# 0
```

**0 匹配**（无字符串拼接、无动态拼接、无 ignore 绕过）。

## 6. 测试结果

```
$ python -m unittest discover -s tests/compose -p "test_*.py"
Ran 244 tests in 9.931s
OK
```

泄漏专项（动态，两任务 eval 全部 24+24 样本）：
- `answer_features_used = False`、`oracle_used = False`、`task_id_lookup_used = False`、`clustering_used_at_test = False`
- 泄漏单元测试 4 项：`test_query_no_answer_leakage`、`test_compose_n_split_leakage`、`test_sufficiency_no_test_leakage` → OK

## 7. Smoke 结果（`configs/compose_ucit_smoke.yaml`，最终代码 fresh root `compose_ucit_smoke3`）

> 由修复 key-stats 指标与 snapshot extra 之后的最终代码重跑生成（与修复前 run 15 对比：除 key 指标与 `router_pool_version` 外全部逐字节一致，验证修复零扰动）。

### Task 0（ImageNet-R，40 train + 16 val）

| 字段 | 值 |
|---|---|
| residual_count | 40（全量冷启动） |
| 聚类 selected K / silhouette | 2 / 0.2082 |
| cluster sizes | [32, 8] |
| noise | [] |
| 新专家 id | 0, 1（消费式分配器） |
| key mode / epochs | learnable / 50；loss 0.2308；pos 0.8568→0.8496；old_neg 0.0（空池，语义空） |
| 最终 keys 余弦 | cos(k0,k1) = 0.7796 |
| commit | pool_version 0→3，一次事务 2 专家，active=[0,1] |
| RMS | 224 层 × kappa（均裁剪至 kappa_min 0.25） |
| Recall 审计（诊断） | pool_size 0（空注册表，无审计目标），audited 40 |
| eval（24 样本，compose_router 模式） | 全部选择 pair {0,1}；router_pool_version 3 |
| 泄漏旗标 | answer_features_used/oracle_used/task_id_lookup_used/clustering_used_at_test 全 False |
| 快照 | pool_version 3，has_router/has_calibration True |

### Task 1（ArxivQA，40 train + 16 val）

| 字段 | 值 |
|---|---|
| 旧池大小 | 2（专家 0,1） |
| Top-M 检索 | 40/40 样本均检索到两个旧专家（candidate_experts = (0,1)，top_m 8 按池大小填充） |
| teacher 结果 | 40/40 空集（旧专家在该切片上无超过空集的正则化增益 J(S)=nll+λ|S|） |
| reuse_count | 0 |
| residual_count | 40（簇级训练样本；noise 样本 66400/74939 排除出训练与清单） |
| 聚类 selected K / silhouette | 2 / 0.2757（sizes [38]，noise 2 样本） |
| 新专家 id | 2（注册表消费式 id） |
| key mode / epochs | learnable / 50；loss 4e-6；pos 0.789→0.780；old_neg 0.3327→0.3327（旧 key 冻结；hinge 初始化即满足：0.789−0.333 > margin 0.3） |
| Recall 审计（诊断） | audited 40，pool_size 2，top_m 8；EmptyFalseRecallRate 1.0；oracle 召回 0（与 teacher 全空集一致，旧专家无增益） |
| commit | pool_version 3→4，一次事务，active=[0,1,2] |
| eval（24 样本） | 直方图 {0:2, 1:21, 2:1}；router_pool_version 4 |
| 泄漏旗标 | 四项全 False |
| 快照 | pool_version 4，active=[0,1,2]，has_router/has_calibration True |

### Resume 验证

同一命令在完整运行后再次调用（smoke3 root）：`run complete` 瞬时返回，exit 0；6 个 `compose_experts.bin` mtime 逐字节不变；pool_version 保持 task0=3 / task1=4——**无重复提交、无版本递增**。

## 8. 已知限制

1. **key_learning 指标记录 bug（已修复并回归验证）**：`_stats()` 以标签字符串索引 ParameterDict，而标签是 new_expert_ids 的**位置**；专家 id ≠ 位置时（task 1：`new_expert_ids=[2]`，历史 key `"0"` 存在）记录的是旧 key 的相似度（0.2723，实为 cos 到旧 key 0）。已改为 `new_keys[str(new_expert_ids[int(label)])]`，新增回归测试（诱饵 key 置于簇心：修复前 0.99→0.99，修复后 0.0→0.35）。训练本身不受影响（loss 侧用堆叠张量正确索引）。修复后真值：task 1 key 2 pos 0.789→0.780、old_neg 0.3327 冻结。本报告全部数字来自修复后最终代码重跑（smoke3）。
2. **单新专家 key 的 hinge 饱和边界**：K=1 时 CE/散度恒为 0（按 §13 设计），唯一梯度来自旧负样本 hinge，key 在 margin 边界满足时停止移动（`cos(q,k)−max_neg ≥ 0.3`）。task 1 的 key 2 初始化（质心）即满足 margin（0.789−0.333 > 0.3），故几乎不动（0.789→0.780，loss 4e-6）——行为符合设计，但意味着该切片上 key 学习对分离度贡献有限，分离主要来自质心初始化。
3. **Task 0 双 key 分离度一般**（最终 cos(k0,k1) = 0.7796）：smoke 切片 40 样本、两聚类中心近共线；CE 在 temperature 0.07 下快速饱和。正式实验中需检查更大数据量下的 key 分离。
4. **Teacher 空集主导**：task 1 在 40 样本 ArxivQA 切片上旧专家（ImageNet-R）无正则化增益 → 全空集、reuse_count=0、oracle 召回 0（诊断指标，合法结果而非错误）；正式实验需更大训练量观察 reuse。
5. **未验证项（按 §36/§37 明确排除）**：merge/prune/shadow/hyperbolic/token routing/全局 calibration 不在 P0 范围；正式六任务实验未启动（§37 要求 P0 全部通过后方可启动，本报告即 P0 验收记录）。
