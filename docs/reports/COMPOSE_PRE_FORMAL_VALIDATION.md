# Compose Pre-Formal Fast Validation 报告（§1-§28）

- **日期**：2026-08-08
- **基线 SHA**：`3c2a7d1ded478e0d7faaa4964097e8170bc9cb3c`
- **验证环境**：conda `hyper`（torch 2.3.1+cu118, transformers 4.33.3），8× RTX 4090（本验证用 GPU 4/5），CUDA_VISIBLE_DEVICES 隔离
- **结论**：**GO_FOR_FULL_SEED42_PILOT**（附条件，见 §25 摘要第 8/9 点）

---

## A. 冻结状态与静态检查（§1-4）

| 检查 | 结果 |
|---|---|
| §1 代码冻结 | BLOCKING 遗留：核心 compose 模块全部 untracked（迁移后未提交）；功能完整可用。建议正式实验前冻结提交（见摘要第 8 点） |
| §2 命名残留 | v6 残留 0 处；无 BLOCKING |
| §3 静态检查 | compileall / bash -n / 全模块导入全 PASS（hyper env） |
| §4 单元测试 | 266/266 PASS（含 §15-19 回归 33 项） |

## B. BLOCKER 复核（§5-6）

- **BLOCKER-1 RMS runtime kappa**：`kappa = clip(R_bar/(R_k+eps), 0.25, 4.0)`，reference=层内激活专家均值；pair 自动 1/√2、triple 1/√3、single 1.0（代码常量已断言）；真实模型重跑 backward 448 个 kappa 参与计算，无 NaN/Inf，无梯度泄漏（§13 亦复核）。
- **BLOCKER-2 pool_version 语义**：初始 1 + 每次 commit 恰好 +1；snapshot-restore 保底保留；crash-recovery resume 永不 bump；7/7 单测 + 真实 artifact 全验证（§15-16）。

## C. 数值 / 确定性（§7-10, §13-14, §21-22）

- Query 生成 bitwise 确定（seed 42）；双模态 10/10。
- Top-M 检索：逐样本 batch×M（无 row-0 广播），tau_res 边界判定（`>` residual / `<=` reuse）全验证；recall miss 记诊断、永不转为能力专家。
- 梯度隔离：真实模型 backward，224/224 新专家 lora_B finite-nonzero、lora_A step-0 零梯度（LoRA 初始化契约）、base/vision/projector/旧专家/未选中新专家全部无梯度；峰值显存 15.38 GB。
- Key learning：真实 path 复现 bitwise（final_loss 3.62e-06，pos 0.7890→0.7802）；CE 标签为 new_expert_ids 位置（K>1 非连续 id 验证）；旧 key bitwise 不变；收敛探针证明梯度方向正确；prototype 模式完全跳过训练（epochs=0，key 不动）。
- **METHOD_OBSERVATION**：质心初始化的 key 在正式 lr/epochs 下几乎不移动（loss 3.6e-06）——符合设计（CE 近饱和 + margin 已满足），分离度主要来自质心初始化。K=1 时 CE/散度恒为 0，hinge 满足 margin 时梯度恰为 0（key 不动）——零梯度退化路径行为符合设计。

## D. 语义链验证（§11-12, §15-19）

- 聚类：K=1/2/3/noise 全覆盖，silhouette 真实诊断，manifest 与 residual/features 0 不一致。
- Direct commit + Snapshot/Resume：真实 smoke3 artifacts 上 A/B/C1/C2/D/D2 全 PASS（hash 复核、恰好一次 bump、幂等 already_committed、crash 恢复不回滚已注册专家、snapshot 完整性与 pool_version 保留、状态机无自动前进）。
- Leakage：train/test slice 0 重叠；residual→cluster manifest→features 全 ⊆ train 且 ∩ test=∅；eval 恰好消费 test ids；14 个 stage marker 全齐。
- 推理纯度：eval 在 `torch.inference_mode()` 下运行（CLIP/select/generate），router `.eval()`，eval 源码无 train/backward/zero_grad；router pool_version == registry（task0=3, task1=4）；RMS calibration 从 checkpoint 加载应用（不重训）；全部 answer 的 4 个 purity flag 均为 False。
- Pair 退化：真实 histogram {task0: 0/0/24, task1: 2/21/1}（空/单/双全语法覆盖，task1 三者兼有）；无 reuse 退化（teacher_set 全空、teacher_loss==empty_loss）；K=2 选中但 1 个不足最小样本簇（formed 1 + 2 noise）按设计处理。

## E. 两任务 Quick Chain 与资源（§20, §23-24）

### Quick chain（configs/compose_ucit.yaml 仅 override：样本数 + 输出根 + §20 规定 1 epoch）

- 路径：`experiments/runs/compose_ucit_quickchain/`；slice：`/tmp/compose_quickchain_slices/`（每任务 512 train + 128 val + 128 test，test 与 train id 全 disjoint；test 记录经 llava 格式转换，schema 与原 smoke 一致）。
- 已通过 diff 归一化证明：除 3 项 sanctioned override 外，quick chain config 与正式 config 逐字节一致（tau_res 2.0、silhouette 0.15、router 0.5/0.5、key lr 3e-4/50ep、RMS 1e-8/0.25/4.0、lora 8/16、lr 2e-4 等全部未动）。

| 指标 | task0 (ImageNet-R) | task1 (ArxivQA) |
|---|---|---|
| 样本 | 512 train / 128 val / 128 test | 512 train / 128 val / 128 test |
| 训练 epoch | 1 | 1 |
| teacher_set 大小分布 | 全空（cold start，设计如此） | train {0: 313, 1: 199, 2+: 0} / val {0: 80, 1: 48}——首次出现非空 teacher_set（单专家偏好，无 pair 胜出） |
| residual / 聚类 | 512/512 residual（tau_res 2.0）；K=2，silhouette 0.317，sizes [276,236]，0 noise | 474 residual / **38 reuse**（历史专家首次真实复用）；K=1，silhouette None（K=1 无 silhouette，设计如此），sizes [474] |
| recall audit | —（cold start 无 oracle） | OracleMemberRecall@1=0.333, @8=1.0, MRR=0.667（诊断信息，非门控） |
| 形成专家 | 0, 1（learnable，key loss 0.227，pos 0.793） | 2（learnable，key loss 4.89e-4，pos 0.750，50 epochs） |
| pool_version 终值 | 3（1+2 commits） | 4（+1 commit，active [0,1,2]） |
| eval 选择 histogram | {0:1, 1:36, 2:91}（128 test） | {0:2, 1:126, 2:0}（128 test，K=1 单簇一致） |
| 阶段耗时 | 总 19 min（S0-2 4、S3-6 8、S7-8 2、S9 4、S10-11 4） | 总 38 min（S2 teacher search 20、S3-6 10、S7-8 3、S9 3、S10-11 3） |
| RMS | validation，224 layers，calibration sha cdc214d2 | validation，224 layers，calibration sha 2dfc371d |

Quick chain 语义链结论：**两任务各自全 14 阶段完成、无 NaN/Inf、无泄漏；历史上第一次走通「teacher_set 非空 → reuse 判定 → 部分样本复用 → 其余进 residual → 新专家形成」的完整真实路径**（task1：199 样本 teacher 偏好 → 38 复用 + 474 residual → expert 2）。教师搜索代价在真实规模（4 selections/样本）下实测 20 min，是后续任务的最主要耗时项。

### 资源估算（§24）

实测锚点（1× RTX 4090，quick chain）：task0 冷启动 19 min（teacher search 1 sel/样本 2 min）；task1 4 sel/样本 teacher search **20 min**（0.0078 min/selection-sample）；S3-S6 residual+聚类 8-10 min @512 residual × 1 epoch；S7-S8 2-3 min；S9 RMS 3-4 min；S10-S11 eval 3-4 min @128 gen。

正式六任务（2000 train/200 val teacher search、3000 test eval、3 epochs、单 GPU 串行）外推：

| 任务 | 旧专家数 | teacher search | 任务总耗时（估） |
|---|---|---|---|
| task0 (ImageNet-R) | 0（cold start） | ~17 min | ~3.3 h |
| task1 (ArxivQA) | 2 | ~69 min | ~4.1 h |
| task2 (VizWiz) | 3 | ~121 min | ~5.0 h |
| task3 (IconQA) | 4 | ~189 min | ~6.1 h |
| task4 (CLEVR) | 5 | ~207 min | ~6.4 h |
| task5 (Flickr30k) | 6 | ~224 min | ~6.7 h |

- **六任务合计 ≈ 31.7 h（约 1.3 天，单 GPU 串行）**；瓶颈是 S2 teacher search（合计 ~14 h，pair 数受 max_pairs=6 封顶）与 S11 eval（3000 gen ≈ 1.4 h/任务）。
- 显存：teacher search 峰值 16.2 GB / 训练 15.4 GB（quick chain 实测），单卡 4090 24 GB 安全。
- 磁盘：features ~50 KB/样本；committed pool ~120 MB/专家；预计全链 ~2.9 GB（smoke3 双任务 706 MB 为实测参照）。

### §23 配置审计（已修复）

- **NON_BLOCKING 工程 bug（§27 流程）**：tasks 2-5 `test_instructions` 指向不存在的 `test.json`（实际文件为 `test_3000.json`）。
  - 原始症状：`configs/compose_ucit.yaml` 6 任务中 4 个 test 路径 `FileNotFoundError`。
  - Root cause：配置文件使用了 `test.json`，数据集目录仅提供 `test_3000.json`。
  - 最小修改：4 行路径修正（[configs/compose_ucit.yaml](configs/compose_ucit.yaml)）。
  - 回归测试：新增 [tests/compose/test_ucit_formal_config.py](../tests/compose/test_ucit_formal_config.py)（3/3 PASS：路径存在性 + 6 任务顺序 + 算法契约值）。
  - 修改文件：`configs/compose_ucit.yaml`、`tests/compose/test_ucit_formal_config.py`。
  - 重跑：回归测试 PASS；quick chain 仅涉及 tasks 0/1 不受影响。

---

## 摘要（9 点）

1. **GO 状态**：**GO_FOR_FULL_SEED42_PILOT**。§1-§24 全部完成：266/266 单测 PASS（含 33 项 §15-19 回归），quick chain 两任务全 14 阶段干净完成（无 NaN/Inf、无泄漏、无 kappa=0.25 异常），无遗留 BLOCKING。唯一前置条件：正式启动前完成冻结提交（见第 8 点）。
2. **BLOCKING 数量**：0 个遗留 BLOCKING（§1 冻结为提交动作，非代码缺陷；§23 的 config 路径 bug 已按 §27 流程修复并附回归测试）。
3. **RMS 正确性**：运行时 kappa 计算、pair 1/√2 缩放、梯度隔离全部验证通过（真实 backward 448 参数有限非零）；quick chain 两任务 RMS calibration sha 均生成且从 checkpoint 加载（224 layers）。
4. **pool_version 语义**：commit 恰好 +1、snapshot 保底、resume 不 bump，7/7 单测 + 真实 artifact 验证；quick chain 终值 task0=3、task1=4，与 commits 数一致。
5. **两任务 quick chain 数字**：task0 512 residual 冷启动 → K=2（silhouette 0.317）→ 专家 0,1 → eval {0:1, 1:36, 2:91}，19 min；task1 首次真实复用（38/512 reuse，teacher_set 199 非空）→ 474 residual → K=1 → 专家 2 → eval {0:2, 1:126, 2:0}，38 min（S2 teacher search 4 sel/样本 20 min）。pool 从 3→4，全部 14 阶段完成。
6. **测试泄漏**：train/test 0 重叠；训练链（residual/manifest/features/teacher）全 ⊆ train；eval 恰消费 test ids；quick chain 两任务切片亦 id-disjoint；无泄漏。
7. **NaN/Inf**：所有审计路径（梯度、keys、loss 项、RMS、quick chain 全阶段产物）未出现 NaN/Inf。
8. **提交就绪度**：功能验证全部通过；核心 compose 模块仍 untracked（迁移后未提交）。正式六任务启动前必须完成一次冻结提交（含 configs/compose_ucit.yaml 修复 + tests/compose/test_ucit_formal_config.py）。
9. **推荐下一步命令**（§28：本验证不启动；该命令留给正式可行性阶段执行）：
   ```
   python -m compose.experiments.task_run \
     --config configs/compose_ucit.yaml \
     --root experiments/runs/compose_ucit_formal_seed42 \
     --first-task 0 --last-task 5 --gpus 4
   ```
   预计单 GPU 串行 ~31.7 h（约 1.3 天），峰值显存 16.2 GB。运行中若任一任务出现全 expert/layer kappa=0.25 的 smoke 现象，按规则立即 STOP 并回报。
