# 代码修改报告：One-shot Learnable Expert Key 可行性验证（Phase A–G 实现）

> 日期：2026-08-29 ｜ 实验：`one_shot_learnable_frozen_key_feasibility`（seed 42）
> 规范约束：不重训 V6.x、不重训 expert pool、base MLLM / expert LoRA / query 全冻结、
> 删除/绕过 MLP Router、历史 keys 永不更新、正式实验仅用物理 GPU 4-7。

## 1. 提交拓扑

| Commit | 仓库 / branch | 内容 | 行数 |
|--------|--------------|------|------|
| `028e7d2` | 主 repo `codex/task0-capacity-chain` | Commit 1：Phase A 只读资产审计（GO 判定） | +399 |
| `545efa7` | worktree `exp/v63-one-shot-frozen-key`（基于 `1403155`） | Commit 2：oracle 标签流水线 + R1/R2/G 实现 + 单元测试 | +941 |
| `f2ff880` | worktree 同上 | Commit 3：实验配置（复现契约） | +76 |

worktree 工作区干净（无未提交改动）；主 repo 无未提交 diff。

## 2. 修改文件清单（git 内 8 个文件，新增 1416 行）

### Commit 1 — Phase A 审计（只读，GO 判定）
| 文件 | 用途 |
|------|------|
| `audit/expert_pool_manifest.json` | 12 专家 pool_0011 结构化审计清单（creation order、verification、gaps、verdict GO） |
| `audit/expert_pool_audit.md` | 12 pool sha256 三方交叉验证表（bin/registry/manifest） |
| `audit/oracle_asset_audit.md` | oracle 资产可用性（t3/t4 query cache 可复用；pool_0009 排除） |
| `reports/one_shot_key_asset_audit.md` | 综合审计报告 + VERDICT: GO（6 项 NO-GO 条件全不触发） |

### Commit 2 — 实现 + 测试
| 文件 | 行数 | 用途 |
|------|------|------|
| `compose/experiments/one_shot_key_oracle_labels.py` | 192 | Phase B/C：`prepare` 生成 12 任务目录（records/selections/manifest + ordered_ids_sha256）+ 4 批 NLL jobs（train_batch_1/2、val_batch_1/2，GPU 轮转 [4,5,6,7]）；`labels` 从 79 候选 exhaustive NLL 推导 S_i* ∈ {empty, {E_a}, {E_a,E_b}} + 汇总 + val_raw_losses |
| `compose/router/one_shot_key.py` | 495 | Phase E/F/G 核心（见下） |
| `tests/compose/test_one_shot_key.py` | 254 | 8 个单元测试（见 §4） |

`one_shot_key.py` 模块结构：
- `_load_aligned`：query_cache + labels 对齐（sample_id 顺序强校验、跨 task query_encoder_hash 一致性校验）
- `centroid_keys`（R1）：窗口内 positive 均值归一化，`task <= CREATION_TASK[expert]` 时序过滤，<2 个 positive 则 raise
- `r2_train`（R2）：按 creation order 逐专家训练；previous keys `requires_grad=False` 永久冻结；centroid init；窗口内 val early stop（score = SetExactAcc, -MeanOracleRegret, -P90OracleRegret）；**best-epoch key/bias 恢复**（共享 log_temperature 不回滚）；bce + anchor 损失
- `evaluate`（G）：纯 cosine 检索 + 全局阈值 tau_none/tau_second，输出 SetExactAcc/regret/RetrievalRecallTop2 等指标
- `simulate`：逐 task 窗口评估 + OldToNewHijackRate（future-expert 路由率）

### Commit 3 — 实验配置
| 文件 | 用途 |
|------|------|
| `compose/experiments/configs/one_shot_frozen_key.json` | 复现契约：RUN root、base_run/pool_0011/base_router（task5 12 专家 creation_task 已验证一致）、splits、job 批次、query hash、R1/R2 超参、阈值网格、watchdog 约定 |

## 3. 非 git 运行资产（RUN root `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/one_shot_frozen_key_seed42/`）

| 路径 | 状态 |
|------|------|
| `oracle_nll/t{0..5}_{train,val}/`（12 目录 × records/selections/manifest） | ✅ prepare 完成 |
| `oracle_nll/jobs/` 4 批文件 | ✅ 就绪 |
| `query_cache/t{0..5}_{train,val}.pt`（12 个） | ✅ Phase D 完成，query_encoder_hash 全部 `d2bc62e…` |
| `watch/phase_b_watcher.sh` | ✅ 幂等 watchdog：GPU 4-7 空闲 → 按序派发 4 批；FAILED/死 worker 自动清理重试；12 个 nll.json 齐 → labels → r1/r2/evaluate/simulate |

## 4. 测试验证

- `tests/compose/test_one_shot_key.py`：**8/8 通过**（centroid 窗口、boundary 索引、顺序冻结回归断言、检索基数、oracle regret、label/query 对齐、hijack rate）
- 全量 compose 套件：413 通过 / 3 失败（`test_ucit_evaluator_parity` 缺 `java` 环境，与本次改动无关）
- `bash -n` watchdog 语法校验 ✅

## 5. 代码审核修复记录（审核后修复，已入 Commit 2）

| 严重度 | 发现 | 修复 |
|--------|------|------|
| 严重 | early stop 保留的是 final-epoch 权重（与注释/规范"best epoch"矛盾） | 逐 epoch 保存 best key/bias，结束恢复 |
| 中 | `simulate` 硬编码 0.5/0.5 阈值 | 加 `--tau-none/--tau-second` |
| 中 | `set_name`/regret 与 bestset_teacher、key_only_bestset 重复实现 | 改为 import 复用 |
| 中 | `evaluate` 对 R1 state（无 bias 键）KeyError | R1 state 补字段 + `.get` 兜底 |
| 中 | 测试未真实验证"历史 key 冻结" | 新增 final-keys == snapshot-keys 断言 |
| 顺手 | val_indices 每 epoch 重算 / labels 每 task 重读 / 跨 task encoder hash 无校验 | 循环外提、读一次、加校验 |

## 6. 实验推进状态

- ✅ Phase A（GO）→ B 准备（jobs 就绪）→ C 数据准备 → D（query cache 完成）
- ⏳ **Phase B/C NLL 计算：阻塞**——GPU 4-7 被 libolin 用户进程占用（~14 GB/卡，已 21.5h，利用率长期 0%），按用户决定"等待释放"，watchdog（cron `a1975bdf`，每半小时）自动检查并派发
- ⏳ Phase E/F/G 与最终报告：依赖 labels 产出后自动执行（watchdog 覆盖，纯 CPU）
