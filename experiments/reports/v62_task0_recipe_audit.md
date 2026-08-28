# V6.2 正式实验 Task0 / E0 训练配置严格审计

审计日期：2026-08-23  
审计对象：旧 V6.2 Task0/E0、当前受控 `single_r8`、Hyper-LLaVA Task0 LoRA  
审计性质：只读历史审计；未启动训练、推理或 GPU 复现实验

## 1. Executive Verdict

**结论：INCORRECT。**

旧 V6.2 Task0/E0 不是当前 `single_r8` 的等价历史实现。旧配方同时存在三个足以改变结论的实质问题：

1. **训练数据错误（P0）**：先对原始、按类别排序的 ImageNet-R 训练 JSON 做 `[:2000]`，没有先 shuffle 或分层采样。E0 因而只看到了 23,998 条中的前 2,000 条，只覆盖 **16/200** 个类别。
2. **优化对象与保存对象不一致（P0）**：旧训练把 token embedding 参数显式设为可训练并实际交给优化器，训练时共有 151,060,480 个可训练参数；但保存和正式评测只保留 19,988,480 个 LoRA 参数，训练中学到的 131,072,000 个 embedding 参数被丢弃。
3. **正式 Router 推理抑制专家（P1）**：正式 Task0 测试的 3,000 个样本中，2,922 个选择空专家，E0 仅激活 78 次，即 **97.4% base-only / 2.6% E0-active**。历史固定直连 E0 的同一测试得分为 23.13，而正式 Router 路径只有 17.13。

此外，旧流程训练使用 `image_aspect_ratio=square`，评测强制改成 `pad`，构成训练/评测预处理不一致（P1）。LoRA dropout 0.0 与当前 0.05 的差异属于 P2，不足以解释主要差距。

因此，旧 V6.2 的约 17–20 分不能被解释为“rank-8 单专家学习能力只有 17–20 分”。它衡量的是：**16 类偏置子集上的非纯 LoRA 训练、丢弃已训练 embedding 后的 E0、再叠加高空路由率**。当前全量、纯 LoRA、统一 pad 预处理的 `single_r8=89.63` 才是受控的单 rank-8 专家容量测量。

## 2. Three-Recipe Comparison

| 项目 | 旧 V6.2 Task0 / E0 | 当前受控 `single_r8` | Hyper-LLaVA Task0 LoRA |
|---|---:|---:|---:|
| 运行根目录/检查点 | `/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs/compose_ucit_v62_formal_seed42/task0` | `/data/ckpt/zhaozhuofan/Hyper-LlaVA-experiments/runs/task0_capacity_chain_seed42/single_r8` | `/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/06_18/Task1` |
| 源码提交 | `fb5e9083d52cc996cdc2ab5a58634a8c489c5b6b` | `37e4b3ee...` | 当前脚本与检查点内 `training_args.json`；提交号未随检查点序列化 |
| 原始训练集 | 23,998 | 23,998 | 23,998 |
| 实际唯一训练样本 | **2,000** | **23,998** | **23,998** |
| 实际类别覆盖 | **16/200** | **200/200** | 200/200（使用同一完整 train.json） |
| 取样方式 | `records_all[:2000]`，无 shuffle/分层 | 全量 | 全量 |
| 被排除样本 | 21,998 | 0 | 0 |
| epoch / 样本暴露数 | 3 / 6,000 | 1 / 23,998 | 1 / 23,998 |
| world size | 1 | 1 | 4 |
| micro batch / GPU | 1 | 1 | 8 |
| gradient accumulation | 8 | 64 | 2 |
| 有效全局 batch | 8 | 64 | 64 |
| optimizer steps | 750 | 374 | 375 |
| 学习率 | 2e-4 | 2e-4 | 2e-4 |
| weight decay | 0（默认/启动配置） | 0 | 0 |
| warmup | 0.03，约 23 step | 0.03，约 12 step | 0.03，约 12 step |
| scheduler | cosine | cosine | cosine |
| 精度 | bf16 + tf32 | bf16 + tf32 | bf16 + tf32 |
| 最大长度 | 2048 | 2048 | 2048 |
| gradient checkpointing | true | true | true |
| 图像预处理（训练） | **square** | pad | pad |
| 图像预处理（评测） | **pad** | pad | pad |
| LoRA 结构 | 单专家，r=8, alpha=16 | 单专家，r=8, alpha=16 | 6 专家 Hyper-LoRA，r=48, alpha=96 |
| LoRA scaling | 2 | 2 | 2 |
| LoRA dropout | **0.0** | 0.05 | 0.05 |
| target modules | q/k/v/o + gate/up/down | 相同 | 相同 |
| 纯 LoRA 参数量 | 19,988,480 | 19,988,480 | 126,763,008（Hyper 全部可训练参数） |
| 训练时实际可训练参数 | **151,060,480** | **19,988,480** | **126,763,008** |
| 额外可训练参数 | **token embedding 131,072,000** | 无；仅启用 input grads，不训练 embedding | backbone 冻结；无该旧问题 |
| 保存内容 | 仅 E0 LoRA，embedding 未保存 | E0 LoRA | Hyper-LoRA adapter + non-LoRA trainables |
| train loss | 0.108877 | 0.214427 | 0.182207 |
| val loss | **未执行** | 0.107023 | 未执行 |
| Task0 得分 | Router 17.13；固定 E0 23.13 | **89.63** | **91.83** |
| seed | 42 | global 42；expert 4201 | **UNVERIFIED**（未写入保存的命令/参数文件） |

说明：Hyper-LLaVA 检查点中的 `adapter_config.json` 明确记录 r=48、alpha=96、dropout=0.05、expert_num=6 和七类 target modules；`training_args.json` 明确记录四卡、全量数据、pad、batch、epoch、优化器相关启动参数；`trainer_state.json` 记录最终 step=375、train_loss=0.182207。Hyper seed 未被命令或保存参数明确记录，因此不作推断。

## 3. Effective Runtime Diff

### 3.1 数据进入 E0 的真实路径

旧提交的 `compose/experiments/task_run.py` 执行顺序是：

```python
records_all = json.load(...)
records_all = _assign_unique_record_ids(records_all)
teacher_train = records_all[:train_limit]
teacher_val = records_all[train_limit:train_limit + val_limit]
```

旧配置中 `teacher_search_train_samples=2000`、validation=200。切片之前没有随机打乱或按类别分层。逐 ID 比对结果如下：

| 检查项 | 结果 |
|---|---:|
| 旧 `teacher_train.json` 行数 | 2,000 |
| 旧 `residual_train.json` 行数 | 2,000 |
| 当前 `train.json` 行数 | 23,998 |
| 旧/当前 ID 交集 | 2,000 |
| 旧集合独有 ID | 0 |
| 当前集合独有 ID | 21,998 |
| 旧集合是否为原始文件精确前缀 | 是 |
| 旧集合类别数 | 16 |
| 当前集合类别数 | 200 |

旧前缀各类别计数为：178、166、147、84、183、150、133、45、195、149、62、140、160、60、142、6。第 16 类只进入 6 条，后续 184 类完全没有进入 E0。

这不是 Task0 residual threshold 丢样本。Task0 初始专家池为空，2,000 条全部以 cold-start residual 进入 E0；实际聚类模式是 `single_bootstrap_no_clustering`，一个 cluster 大小正好为 2,000。错误发生在 residual/聚类之前的数据截断阶段。

### 3.2 步数与“是否训练不够”

旧 E0 完成 750 optimizer steps，当前 `single_r8` 完成 374 steps，Hyper 完成 375 steps。旧实验在数字步数上并不少，但它是对 2,000 条、16 类偏置数据重复 3 epoch；当前两条高分配方各覆盖完整 23,998 条、200 类一次。

所以问题不是简单的“step 太少”。更准确的描述是：

- 旧实验唯一数据覆盖仅为当前的 8.33%；
- 旧实验有效 batch 为 8，梯度噪声及更新轨迹与当前有效 batch 64 不同；
- 旧实验反复拟合类别偏置前缀，低 train loss 不能代表 200 类泛化；
- 旧实验没有 validation loss，无法在训练期发现该泛化失败。

### 3.3 可训练参数、优化器与保存边界

旧运行日志明确输出：

```text
Trainable parameter count: 151060480
```

其中：

- E0 LoRA：19,988,480；
- `model.model.embed_tokens.weight`：32,000 × 4,096 = 131,072,000；
- 合计：151,060,480。

旧 `train_compose.py` 在冻结基础模型和启用 E0 后，又执行：

```python
model.model.embed_tokens.weight.requires_grad_(True)
```

代码注释声称 embedding 不进入 optimizer，但旧 `llava_trainer.py:create_optimizer` 会收集所有 `requires_grad=True` 参数。因此 embedding 确实参与优化。另一方面，`expert_state_dict` 只导出 LoRA expert tensors。

检查点审计进一步确认：旧独立 E0 与提交到 pool 的 E0 均为 448 个 tensor、19,988,480 个参数，逐 tensor 完全一致；正式评测也报告 expected=448、loaded=448。由此可排除“LoRA 保存路径错误”，但确认了“训练时优化 embedding、保存时丢弃 embedding”的契约错误。

当前受控训练只通过 `enable_input_require_grads()` 满足 gradient checkpointing 对输入梯度的需求，不把 embedding 参数本身设为可训练；实际 trainable count 精确为 19,988,480。

### 3.4 LoRA 是否实际更新

旧 E0 不是零更新或完全失效：

| 检查项 | 旧 E0 | 当前 `single_r8` |
|---|---:|---:|
| LoRA tensors | 448 | 448 |
| LoRA 参数 | 19,988,480 | 19,988,480 |
| A 总范数 | 26.5156 | 26.2322 |
| B 总范数 | 10.4550 | 8.7233 |
| 非零 B tensors | 224/224 | 224/224 |
| LoRA-B 有限梯度模块 | 224/224 | 224/224 |

旧监督审计还记录 zero-supervision=0。因此 label masking、LoRA 注入和反向传播不是主故障点。旧 train loss 很低但测试差，主要符合偏置小样本记忆、embedding 协同训练后被丢弃、以及预处理失配的组合特征。

### 3.5 训练与评测前向差异

旧 Task0 启动命令没有传 `--image_aspect_ratio`，保存配置为 `square`；旧 `compose/eval/load_compose.py::_foundation` 在评测时显式设为 `pad`。当前 `single_r8` 和 Hyper-LLaVA 均从训练命令开始使用 `pad`，评测也使用 `pad`。

提示模板和生成设置不是差距来源：旧/当前均为 Vicuna v1，使用同一 ImageNet-R 3,000 条测试集、确定性生成、`do_sample=false`、`num_beams=1`、`max_new_tokens=128` 和同一 scorer。

单专家 RMS 系数在各层约为 1.0，且固定单专家路径的 gate=1，因此 Task0 的 RMS 缩放不是主要原因。

## 4. Root Cause Ranking

| 优先级 | 根因 | 证据强度 | 对 17–20 vs 89.63 的判断 |
|---|---|---|---|
| **P0-1** | 先截断未打乱、按类别排序的数据，只训练 2,000 条/16 类 | 直接代码、逐 ID 和类别统计 | **主因**。E0 从未学习其余 184 类，无法代表 ImageNet-R Task0 容量 |
| **P0-2** | embedding 被优化但未保存，训练/部署参数集合不一致 | 直接代码、日志参数量、checkpoint tensor 审计 | **主因**。训练 loss 部分依赖最终不存在的参数状态 |
| **P1-1** | Router `tau_none` 导致 97.4% 样本空路由 | 正式 selection histogram + 历史固定 E0 对照 | **确定的推理损失**：固定 E0 23.13 → Router 17.13，约 -6.00 点 |
| **P1-2** | 训练 square、评测 pad | 保存配置和加载代码 | 显著风险；具体独立贡献未做消融，不能伪精确归因 |
| **P1-3** | 小 batch、三轮重复偏置子集且无验证 | runtime 参数和 trainer state | 放大过拟合并掩盖失败，不是单独充分原因 |
| **P2-1** | dropout 0.0 vs 0.05 | adapter/runtime 配置 | 正则化差异，预计为次要因素 |
| 排除 | LoRA target/rank/alpha/scaling 错误 | 结构和 checkpoint 审计 | 旧 E0 与当前均为七模块 r8/alpha16/scaling2 |
| 排除 | LoRA 没有梯度或没有保存/加载 | 梯度摘要、范数、逐 tensor、load summary | 448/448 全部保存并加载，B 全部非零 |
| 排除 | scorer、prompt 或测试集不同 | 评测命令与运行摘要 | 不是主要差距来源 |
| 排除 | 单专家 RMS 缩放错误 | layer-wise kappa | 单专家约为 1.0 |

不能把 72.50 分差距逐项做无依据的线性拆分。但现有历史对照足以给出边界：Router 本身解释正式 17.13 到固定 E0 23.13 的 6.00 分；固定 E0 仍远低于 89.63，故剩余主要差距已经存在于训练产物，首要由数据截断/类别覆盖错误与优化-保存契约错误解释。

## 5. Training-vs-Inference Diagnosis

**诊断：MIXED FAILURE，以训练配方失败为主，Router 推理失败为辅；不是单纯 inference/evaluator failure。**

证据链如下：

1. Base 得分为 16.20。
2. 旧 E0 在固定直连、绕过 Router 的历史评测中得 23.13，说明 E0 有学习但能力有限。
3. 同一 E0 经正式 Router 仅得 17.13，因为 2,922/3,000 样本没有激活任何专家。
4. 即使完全修正 Router，历史 E0 的已知上界对照仍只是 23.13，远非当前 `single_r8` 的 89.63。
5. 因而 Router 不是唯一原因；旧 E0 训练产物本身已经失败。

旧 Router 的确定性阈值逻辑为：若最高相似度 `< tau_none`，返回空专家集合；否则才选择第一专家。正式直方图说明该条件在 97.4% 测试样本上触发。这个行为在只有一个 E0 的 Task0 尤其不合理：系统大多数时间退化为 Base。

结论可概括为：

```text
旧正式 17.13
  = 弱 E0（固定直连仅 23.13）
  + Router 97.4% 空路由（额外下降约 6 点）

弱 E0
  = 2,000 条按类排序前缀（仅 16/200 类）
  + 训练了但未保存 token embedding
  + square→pad 预处理失配
  + 小 batch/重复拟合且无验证
```

## 6. Recommended Minimal Fix

若修复旧 V6.2 训练逻辑，最小必要改动应是：

1. Task0 不得用 `records_all[:train_limit]` 构造容量实验训练集。正式容量测量直接使用完整 23,998 条；若必须限样本，必须先按 seed 固定地分层采样，并在 manifest 中写出实际类别覆盖和样本 ID 哈希。
2. 明确训练/保存参数契约：纯专家实验只允许 LoRA 参数进入 optimizer。使用 input-gradient hook 支持 gradient checkpointing，不把 token embedding 参数设为 trainable；启动时断言 trainable count=19,988,480。
3. 训练和评测统一 `image_aspect_ratio=pad`，把该值写入 checkpoint manifest，并在加载时校验而不是静默覆盖。
4. Task0 单专家容量评测使用固定 E0 路径。Router 指标应作为另一个实验单独报告；至少同时报告 empty-route rate、expert-active score 和 fixed-E0 score。
5. 保留当前已验证的 r8/alpha16、七 target modules、LR 2e-4、有效 batch 64、一轮全量数据和 validation。训练结束强制记录唯一样本数、类别覆盖、optimizer steps、trainable 参数清单、train/val loss、checkpoint tensor 数和加载完整性。

### 是否需要最小复现

**不需要。** 历史证据已经包含所需 A/B：同一旧 E0 的 Router 结果 17.13 与固定直连结果 23.13；当前全量纯 LoRA `single_r8` 结果为 89.63。数据 ID、训练参数、checkpoint、加载和前向路径也均可从已有产物闭环验证。新增 GPU 复现不会改变本审计对旧配方是否正确的判断。

## Evidence Index

- 旧正式运行：`/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs/compose_ucit_v62_formal_seed42`
- 旧 Task0 报告：`task0/report/task_report.md`
- 旧正式评测摘要：`task0/eval_output/run_summary.json`
- 旧持续学习矩阵：`evaluation/continual_matrix.csv`
- 历史无 Router 固定 E0 对照：`/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs/compose_v62_seed42_no_router_oracle/cache_audit/combination_matrix.csv`
- 当前受控运行：`/data/ckpt/zhaozhuofan/Hyper-LlaVA-experiments/runs/task0_capacity_chain_seed42/single_r8`
- Hyper 检查点：`/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/06_18/Task1`
- Hyper 保存命令：上述检查点内 `training_args.json`
- Hyper adapter 配置：上述检查点内 `adapter_config.json`
- Hyper trainer state：上述检查点内 `trainer_state.json`
- Hyper 当前启动脚本：`scripts/Hyper/Train_UCIT/Task1.sh`
- 旧源码提交：`fb5e9083d52cc996cdc2ab5a58634a8c489c5b6b`

**AUDIT COMPLETE / NO GPU REPRODUCTION REQUIRED**
