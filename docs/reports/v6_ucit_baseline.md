# V6 Stage 01：Hyper-LLaVA UCIT 基线复现报告

- 日期：2026-08-02
- 分支：`feat/v6-ucit-staged`
- 实验目录：`experiments/runs/v6_ucit_staged/stage01_baseline/`
- 状态：**完成**（两套 full UCIT seed42 基线均复现并冻结）

## 1. UCIT 任务顺序（锁定）

ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k（cur_task 0–5），权威来源 `scripts/Hyper/Train_UCIT/Task{1..6}.sh`，与 Stage 00 审计一致。

## 2. 原脚本 batch 设置

| Task | cur_task | 数据集 | 脚本配置 | 脚本 global batch |
|---|---|---|---|---|
| 1 | 0 | ImageNet-R | 4gpu×8×2 | 64 |
| 2 | 1 | ArxivQA | 4gpu×4×2 | 32 |
| 3 | 2 | VizWiz | 4gpu×8×2 | 64 |
| 4 | 3 | IconQA | 4gpu×8×2 | 64 |
| 5 | 4 | CLEVR | 8gpu×8×1 | 64 |
| 6 | 5 | Flickr30k | 8gpu×8×1 | 64 |

所有参数从 Task 脚本逐字解析（`configs/*/task_params.json` 含原脚本 SHA-256），LoRA rank 48 / alpha 96、epoch 1、lr 2e-4、cosine、warmup 0.03、bf16、expert_num 6、seed 42 全部与脚本一致。

## 3. 两套基线定义

| 名称 | global batch | 用途 |
|---|---|---|
| `hyper_llava_original_batch` | Task1/3/4/5/6=64, Task2=32 | 原实现复现判断（**不**称其为严格原样复现：Task5/6 原用 8 卡，本次按 4 卡重缩放；Task2 原 32 保留） |
| `hyper_llava_gb24_matched` | 全部 24 | 后续 V6 公平比较基线 |

两者不混合计算；V6 主实验必须与 gb24_matched 比较。

## 4. 2–4 GPU 实际配置

全程使用 **4 张卡**（受用户约束，串行两套）：

| 配置 | 4 GPU（实际） | per-device | grad accum | global batch |
|---|---|---|---|---|
| gb24 全部任务 | ✓ | 2 | 3 | 24 |
| original_batch=64（T1/3/4/5/6） | ✓ | 2 | 8 | 64 |
| original_batch=32（T2） | ✓ | 2 | 4 | 32 |

启动前由 `make_launch.py` 自动断言实际 global batch。3/2 卡配置已验证可整除（未实际使用）。

## 5. Smoke 结果（Step 2，ImageNet-R，30 步）

| 检查项 | smoke_orig | smoke_gb24 |
|---|---|---|
| loss | 1.71 → 0.96（有效下降） | 1.80 → 1.31 |
| NaN/Inf | 无 | 无 |
| checkpoint | adapter_model.bin / stats.json / non_lora_trainables.bin / config.json 齐全 | 同左 |
| 路由统计 | adaptive_w_img=[0.5]（单任务均匀） | 同左 |
| checkpoint 加载 + evaluator | 128/128 预测成功 | — |
| 4 卡 DeepSpeed 退出 | 0 | 0 |

## 6. mini2 accuracy matrix（Step 3-4，512 train / 128 test，seed 42，两套同数据）

| 配置 | ImageNet-R (t1后) | ImageNet-R (t2后) | ArxivQA (t2后) |
|---|---|---|---|
| original_batch | 46.09 | 46.09 | 77.34 |
| gb24_matched | 47.66 | 50.00 | 80.47 |

MAA/MFN/MFT/BWT：orig 53.90/61.72/61.72/0.00；gb24 56.45/65.24/64.07/+2.34。mini2 仅 512 训练样本，数值不代表 full 水平，仅验证流水线闭环（训练→保存→恢复→评测→矩阵→公式）。

## 7. full 6×6 accuracy matrix（seed 42）

### hyper_llava_original_batch（%）

| | ImageNet-R | ArxivQA | VizWiz | IconQA | CLEVR | Flickr30k |
|---|---|---|---|---|---|---|
| task1 后 | 91.73 | | | | | |
| task2 后 | 90.40 | 94.23 | | | | |
| task3 后 | 89.37 | 94.07 | 61.37 | | | |
| task4 后 | 87.07 | 94.10 | 57.40 | 86.00 | | |
| task5 后 | 87.37 | 94.27 | 56.70 | 78.73 | 79.70 | |
| task6 后 | 86.67 | 94.00 | 58.71 | 79.03 | 64.33 | 58.20 |

### hyper_llava_gb24_matched（%）

| | ImageNet-R | ArxivQA | VizWiz | IconQA | CLEVR | Flickr30k |
|---|---|---|---|---|---|---|
| task1 后 | 91.90 | | | | | |
| task2 后 | 91.03 | 94.83 | | | | |
| task3 后 | 89.63 | 94.43 | 61.77 | | | |
| task4 后 | 86.27 | 93.87 | 57.54 | 84.63 | | |
| task5 后 | 85.90 | 93.97 | 54.60 | 76.83 | 79.57 | |
| task6 后 | 86.80 | 93.83 | 59.06 | 76.17 | 65.37 | 58.19 |

## 8. MFT / MFN / MAA / BWT（公式复用 `summarize_continual_metrics.py`，3×3 手工矩阵独立复核通过）

| 指标 | original_batch | gb24_matched | 差（gb24−orig） |
|---|---|---|---|
| MAA | 83.27 | 83.13 | −0.15 |
| MFN | 73.49 | 73.24 | −0.25 |
| MFT | 78.54 | 78.48 | −0.06 |
| BWT | −6.06 | −6.29 | −0.24 |

## 9. 路由分布

`stats.json` 的 `adaptive_w_img`（每任务更新，两套几乎一致）：
- task1：[0.5]（单任务均匀）
- task2：[~0, ~0]（几乎全图像；ArxivQA 文本主导）
- task3：[0.36, 0.77, 0.34]
- task4：[0.10, 0.96, 0.07, 0.85]
- task5：[0.02, 0.77, 0.01, 0.66, 1.00]
- task6：[~0, 0.78, 0.67, 0.64, 1.00, 0.60]

图像/文本高斯统计每 expert 随训练累计（image_count/text_count）。任务间路由先验随阶段增长，两套 batch 配置分布一致。

## 10. 参数量（每任务 checkpoint，含全部 6 专家槽）

- 每任务 adapter 总参数：119,930,880（6 专家槽 × ~19.99M；rank-48 LoRA，目标模块 q/k/v/o/gate/up/down_proj）
- 每任务实际新增：~19.99M（仅 cur_task 槽训练）
- 最终累计：719,585,280（6×119.93M；含未激活槽位）
- 推理激活：单专家 ~19.99M
- 两套配置完全一致

## 11. 训练时间与显存

| 任务 | orig 训练时长 | gb24 训练时长 |
|---|---|---|
| ImageNet-R | ~43 min | ~45 min |
| ArxivQA | ~1h26m | ~1h27m |
| VizWiz | ~1h11m | ~1h13m |
| IconQA | ~55 min | ~38 min |
| CLEVR | ~1h11m | ~1h11m |
| Flickr30k | ~1h12m | ~1h12m |

- 峰值显存：~10.5 GB/卡（4 卡 bf16 LoRA + grad checkpoint；其他用户占用时 >23GB 会 OOM）
- 评测：3000 样本/数据集，4 卡并行 ~2.5-3 min 生成 + 打分（caption 数据集 METEOR 首次需启动 JVM）

## 12. OOM 与 retry（全部保留证据）

| 事件 | 处理 |
|---|---|
| 首次 smoke：根分区 100% 满（半成品 checkpoint 63G） | 保留日志，删除半成品，checkpoint 迁移 /data |
| full eval：缺 Java（METEOR 指标） | conda 安装 openjdk 25 到 hyper env，run_eval.sh 注入 PATH |
| full_orig task3：GPU 0-3 被其他用户（openpi）占用致 OOM | 保留 OOM 日志；不结束他人进程；换空闲 GPU 4-7 retry；gb/lr/rank 不变；manifest 标记 retry |
| 用户约束仅 4 卡 | 两套串行（gb24 先，orig 后 resume） |

## 13. 两套 batch 差异

两套配置差距极小：MAA/MFN/MFT/BWT 相差 <0.3pp，各单元相差 <1.2pp（IconQA t4: 86.00 vs 84.63 为最大单元差）。无系统性优劣；优化步数不同（gb24 步数更多）但结果趋同。不据此调整任何后续 V6 门槛。

## 14. 与已有复现结果的差异

`runs/results/UCIT/06_18/continual_metrics.json` 参考（R[1][1]=91.87, R[2][1]=83.2, R[2][2]=93.87）。本次 R[1][1]=91.73/91.90（一致）、R[2][2]=94.23/94.83（一致），但 R[2][1]（ImageNet-R 在 task2 后）=90.40/91.03，高于 06_18 的 83.2。可能原因：06_18 批次的数据预处理/版本差异、或旧脚本路径（train_all.sh OUTPUT_DIR 不匹配 bug 期间产物）。本次未修改 evaluator、未调整任何数值。

## 15. V6-off 行为兼容性

- 未启用任何 V6 flag；未新增 V6 代码（Stage 01 纯配置/脚本/报告）
- `git diff HEAD` 零修改：未触碰 `clitmoelora.py`、`train_MOE.py`、`builder.py`、原 Task/Eval 脚本、evaluator、UCIT 数据
- 原 60 个 compose 测试全部通过（8.5s，OK）
- 训练/推理仍走原 Hyper-LLaVA 路径（cur_task 激活、CLIP 高斯路由、adaptive_w_img、adapter_model.bin 格式不变）
- 唯一基础设施修复：eval 用绝对路径 hyper env python（原脚本假设 PATH 已有）、checkpoint 输出目录命名 `<tag>_llava_lora_ours`（builder.py 按 `'llava' in model_name` 分支的既有约定）

## 16. Stage 01 最终状态

**全部验收通过**：
1. ✅ 静态测试全过（compile/shell/60 测试/schema/gb 断言/索引确定性/checkpoint 路径/evaluator import）
2. ✅ original_batch smoke
3. ✅ gb24 smoke
4. ✅ mini2 original_batch（2×2 矩阵）
5. ✅ mini2 gb24（2×2 矩阵）
6. ✅ full UCIT seed42 original_batch（6×6 矩阵）
7. ✅ full UCIT seed42 gb24（6×6 矩阵）
8. ✅ MFT/MFN/MAA/BWT 可计算（公式独立复核）
9. ✅ checkpoint 参数量与任务边界核验
10. ✅ 所有 OOM/retry 保留（日志+manifest）
11. ✅ V6-off 行为未改变
12. ✅ 未使用测试集选 checkpoint（save_strategy epoch + evaluation_strategy no）
13. ✅ 产物 SHA-256 已记录（checksums.json）
14. ✅ 报告明确区分两套 batch

**与任务指令的偏差记录**：任务规定"UCIT 原始有效 global batch 为 24"，审计发现 Task 脚本实际为 32/64，故按任务指令以 gb24_matched 作为 V6 主比较基线、original_batch 作为原实现复现（两套都完成，不混合）。

**命名**：后续 V6 比较一律使用 `hyper_llava_gb24_matched_seed42`；`hyper_llava_original_batch_seed42` 仅用于原实现复现参考。
