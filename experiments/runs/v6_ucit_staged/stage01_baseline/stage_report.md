# Stage 01 报告：复现并冻结 Hyper-LLaVA UCIT 基线

- 日期：2026-08-02
- 分支：`feat/v6-ucit-staged`
- 状态：**完成**

## 两套基线定义

| 名称 | 定义 | global batch | 用途 |
|---|---|---|---|
| `hyper_llava_original_batch` | 严格复现仓库 Task 脚本 global batch | Task1/3/4/5/6=64, Task2=32 | 原实现复现判断 |
| `hyper_llava_gb24_matched` | 六任务统一 global batch 24 | 全部 24 | 后续 V6 公平比较基线 |

原始脚本 global batch 解析（scripts/Hyper/Train_UCIT/Task{1..6}.sh 权威来源）：

| Task | cur_task | 数据集 | 脚本配置 | 脚本 global batch |
|---|---|---|---|---|
| 1 | 0 | ImageNet-R | 4gpu×8×2 | 64 |
| 2 | 1 | ArxivQA | 4gpu×4×2 | 32 |
| 3 | 2 | VizWiz | 4gpu×8×2 | 64 |
| 4 | 3 | IconQA | 4gpu×8×2 | 64 |
| 5 | 4 | CLEVR | 8gpu×8×1 | 64 |
| 6 | 5 | Flickr30k | 8gpu×8×1 | 64 |

V6 任务指令要求 global batch 24 与 2–4 卡；Task5/6 原脚本用 8 卡，本次按 4 卡重缩放（per_device=2, accum=8 → 64）。Task2=32 用 4 卡 per_device=2 accum=4。

## 2–4 GPU 实际配置（本次采用 4 卡）

| 配置 | 4 GPU | 3 GPU | 2 GPU |
|---|---|---|---|
| gb24 | bs2 × accum3 = 24 | bs2 × accum4 = 24 | bs2 × accum6 = 24 |
| original_batch=64 | bs2 × accum8 = 64 | bs2 × accum(64//6→不整除, 需 fallback) | bs2 × accum16 = 64 |
| original_batch=32 | bs2 × accum4 = 32 | — | bs2 × accum8 = 32 |

启动前由 make_launch.py 断言 effective global batch。

## 基础设施

- `configs/{original_batch,gb24_matched}/task_params.json`：从 Task 脚本解析的完整参数 + 原脚本 SHA-256
- `data_indices/`：mini2 确定性索引（seed 42，ImageNet-R/ArxivQA 各 512/128/128，train/val 不重叠，含抽样 ID manifest）
- `scripts/make_launch.py`：从原 Task 脚本复制并覆盖 batch/accum/output/data/seed 的启动生成器
- `scripts/run_eval.sh` / `run_eval_mini.sh`：官方 evaluator 评测（chunk 并行 / mini2 test）
- `scripts/summarize_stage01.py`：MFT/MFN/MAA/BWT（复用官方公式 + 手工矩阵独立复核）
- `scripts/collect_manifest.py`：训练信息 + checkpoint SHA-256 + 环境版本收集

## 环境

- `/home/zhaozhuofan/miniconda3/envs/hyper`：Python 3.10.20, torch 2.3.1+cu118, peft 0.4.0, transformers 4.33.3, deepspeed 0.14.5
- GPU：8×RTX 4090 24GB（空闲）
- 注意：conda env 不在 PATH，启动器固定 hyper env 的 deepspeed 绝对路径

## Smoke（Step 2）

| 检查项 | smoke_orig | smoke_gb24 |
|---|---|---|
| max_steps | 30 | 30 |
| loss 序列 | 1.71 → 0.96 | 1.80 → 1.31 |
| NaN/Inf | 无 | 无 |
| checkpoint | adapter_model.bin + stats.json + non_lora_trainables.bin 等 | 同左 |
| 路由统计 stats.json | adaptive_w_img [0.5]（单任务均匀） | 同左 |
| checkpoint 加载 + evaluator | 128/128 预测成功 | — |
| 退出 | 0（4 卡 deepspeed） | 0 |

**基础设施修复（保留证据，非 evaluator 改动）**：
1. 首次 smoke 崩溃于 `PytorchStreamWriter failed writing file` → 根分区 100% 满（819G，smoke checkpoint 半成品 63G）。已删除失败的半成品 checkpoint（保留日志），checkpoint 输出迁移至 /data（4.2T 可用）。
2. eval 加载走 PyPI peft 分支（KeyError MOE_LORA_Hyper）：builder.py 按 `'llava' in model_name` 分支，原仓库 checkpoint 目录名 `*_llava_lora_ours`。不改 builder.py，输出目录统一命名为 `<tag>_llava_lora_ours`。

## mini2（Step 3-4，seed 42，同一 mini2 数据）

accuracy matrix（%），行=训练后阶段，列=任务：

| | ImageNet-R | ArxivQA |
|---|---|---|
| original_batch task1 | 46.09 | — |
| original_batch task2 | 46.09 | 77.34 |
| gb24 task1 | 47.66 | — |
| gb24 task2 | 50.00 | 80.47 |

| 指标 | original_batch | gb24_matched |
|---|---|---|
| MAA | 53.90 | 56.45 |
| MFN | 61.72 | 65.24 |
| MFT | 61.72 | 64.07 |
| BWT | 0.00 | +2.34 |

注：mini2 仅 512 训练样本，数值不代表 full 水平，仅验证 pipeline 闭环（训练→保存→恢复→评测→矩阵）。两套 batch 差异存在但方向一致（gb24 略优），不据此调整任何 V6 门槛。

## full UCIT seed42（Step 5-6，完成）

- 全程 4 卡串行（用户约束）：gb24 先（GPU 4-7），orig 后（GPU 4-7；0-3 被其他用户占用后换卡）
- 每任务后评测全部已见任务（官方 test_3000，4 卡 chunk 并行）
- 输出：`evaluations/full_{orig,gb24}/`（21/21 数据集-阶段齐全），checkpoint：`/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints/full_{orig,gb24}_task{1..6}_llava_lora_ours`

### 6×6 accuracy matrix（seed 42）

original_batch：

| | ImageNet-R | ArxivQA | VizWiz | IconQA | CLEVR | Flickr30k |
|---|---|---|---|---|---|---|
| t1 | 91.73 | | | | | |
| t2 | 90.40 | 94.23 | | | | |
| t3 | 89.37 | 94.07 | 61.37 | | | |
| t4 | 87.07 | 94.10 | 57.40 | 86.00 | | |
| t5 | 87.37 | 94.27 | 56.70 | 78.73 | 79.70 | |
| t6 | 86.67 | 94.00 | 58.71 | 79.03 | 64.33 | 58.20 |

gb24_matched：

| | ImageNet-R | ArxivQA | VizWiz | IconQA | CLEVR | Flickr30k |
|---|---|---|---|---|---|---|
| t1 | 91.90 | | | | | |
| t2 | 91.03 | 94.83 | | | | |
| t3 | 89.63 | 94.43 | 61.77 | | | |
| t4 | 86.27 | 93.87 | 57.54 | 84.63 | | |
| t5 | 85.90 | 93.97 | 54.60 | 76.83 | 79.57 | |
| t6 | 86.80 | 93.83 | 59.06 | 76.17 | 65.37 | 58.19 |

### MFT/MFN/MAA/BWT

| 指标 | original_batch | gb24_matched | 差 |
|---|---|---|---|
| MAA | 83.27 | 83.13 | −0.15 |
| MFN | 73.49 | 73.24 | −0.25 |
| MFT | 78.54 | 78.48 | −0.06 |
| BWT | −6.06 | −6.29 | −0.24 |

### 路由与参数量

- adaptive_w_img 先验随任务增长（t1 [0.5] → t6 六值），两套一致
- 每任务 checkpoint 119,930,880 参数（6 专家槽 × ~19.99M）；每任务新增 ~19.99M；累计 719,585,280；激活单专家 ~19.99M

## OOM 与故障记录

| 事件 | 处理 |
|---|---|
| 首次 smoke：根分区 100% 满（半成品 checkpoint 63G） | 保留日志，删除半成品，checkpoint 迁移 /data，重跑成功 |
| eval 加载 KeyError MOE_LORA_Hyper | 输出目录命名加 `_llava_lora_ours`（builder.py 既有约定），重跑成功 |
| full eval：缺 Java（pycocoevalcap METEOR 依赖 java -jar） | conda 安装 openjdk 25（hyper env），run_eval.sh 注入 PATH，resume 只补评测 |
| full_orig task3：GPU 0-3 被其他用户（openpi）占用致 OOM | 保留 OOM 日志；不结束他人进程；换空闲 GPU 4-7 retry；gb/lr/rank 不变；manifest 标记 |
| 用户约束仅 4 卡（12:31） | 暂停 orig（保留 task1/2），gb24 先跑完，orig resume 后跑完 |
| push 认证失败（Stage 00 遗留） | 记录，未强推 |

## V6-off 兼容性

- `git diff HEAD` 零修改（原文件未触碰）；60 compose 测试全过；原训练/推理/路由路径不变

## 最终状态

- 验收 14 项全部通过（详见 docs/reports/v6_ucit_baseline.md §16）
- 后续 V6 主实验与 `hyper_llava_gb24_matched_seed42` 比较
