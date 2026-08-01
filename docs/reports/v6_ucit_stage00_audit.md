# V6 Stage 00：代码与 UCIT 流水线审计

- 日期：2026-08-01
- 分支：`feat/v6-ucit-staged`
- 基线 HEAD：`dc7ad7c9f1f6407549b285f7c420ce8c5a50b78d docs_compose_report_residual_expert_feasibility_evidence`
- 状态：只审计，未修改任何方法行为

## 1. UCIT 六任务顺序（正式）

以 `scripts/Hyper/Train_UCIT/Task{1..6}.sh` 为唯一权威来源（不得按记忆重建）：

| Task 脚本 | cur_task | 数据集 | 指令文件（`ucit_instructions/`） | 评测指标类型 |
|---|---|---|---|---|
| Task1.sh | 0 | ImageNet-R | `ImageNet-R/train.json` | Accuracy |
| Task2.sh | 1 | ArxivQA | `ArxivQA/train_4w.json` | Accuracy |
| Task3.sh | 2 | VizWiz | `VizWiz/train.json` | Average (COCO caption) |
| Task4.sh | 3 | IconQA | `IconQA/train.json` | Accuracy |
| Task5.sh | 4 | CLEVR | `CLEVR/train_4w.json` | Accuracy |
| Task6.sh | 5 | Flickr30k | `Flickr30k/train_brief_4w.json` | Average (COCO caption) |

与 `scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py:11-19` 的 `TASKS` 顺序一致（CLEVR 读作 `CLEVR-Math`）。图像目录：`/data/dataset/zhaozhuofan/UCIT/datasets`。另有 `Train_UCIT_IFRCAV`（IconQA 优先）与 `Train_UCIT_AIRFCV`（ArxivQA 优先）两个变体，本任务固定使用主顺序，不触碰变体。

## 2. 基线启动命令与超参数（原 Hyper-LLaVA）

入口：`llava/train/train_mem_MOE.py` → `llava/train/train_MOE.py`（`train()`）。启动器为 DeepSpeed（非 torchrun）。

- 启动：`deepspeed --include localhost:0,1,2,3 --master_port 29601 llava/train/train_mem_MOE.py --deepspeed ./scripts/zero2.json`
- 模型：`/data/ckpt/zhaozhuofan/models/llava-v1.5-7b`；vision/text tower `clip-vit-large-patch14-336`；`mm_projector_type mlp2x_gelu`
- LoRA：`--lora_r 48 --lora_alpha 96 --lora_dropout 0.05`，target modules 为 `find_all_linear_names` 收集的 `q/k/v/o/gate/up/down_proj`（`train_MOE.py:196-209`，`llava_arch.py:411` 显式列出）
- 持续学习：`--expert_num 6 --cur_task {0..5} --previous_task_model_path ...`（Task2+ 链式加载）
- 优化：`--num_train_epochs 1 --learning_rate 2e-4 --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine --bf16 True --tf32 True`
- 数据：`--model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --group_by_modality_length True --image_aspect_ratio pad`
- 路由：`--modality_routing_mode task --eval_modality_routing_mode same --router_hidden_dim 32 --router_loss_weight 0.1`（默认无记忆回放）

**global batch 差异记录**：Task 脚本实际配置为 Task1/3/4：4 卡×8×2=64；Task2：4×4×2=32；Task5/6：8 卡×8×1=64。任务指令规定 V6 实验使用 global batch 24（2–4 卡配置表）。差异在 Stage 01 以 V6 的 2–4 卡 + grad-accum=24 配置复现基线时处理，Stage 00 不修改脚本。

## 3. 专家体系（原 Hyper-LLaVA 机制）

- 实现：`Hyper/peft/tuners/clitmoelora.py` —— `HyperMOELoraConfig` / `HyperMOELoraModel` / `HyperMOELoraLinear`；每个 expert 是低秩 `nn.Linear` 对（`loraA/loraB` 的 `ModuleList` 元素 `HyperMOEExpert`），rank 按 `r // expert_num` 拆分
- 训练激活：`clitmoelora.py:407-410`，`self.training` 时只用 `cur_task` 对应 expert
- 推理路由：`llava_arch.py:677-696` —— 基于每 expert 的 CLIP 图像/文本高斯统计（`llava_llama.py:75-100` buffer，`llava_arch.py:527-575` 在线更新）+ `adaptive_w_img` 模态先验（`compute_routing_weights.py`，log-BC 距离 + softmax）→ argmax 逐样本选 expert
- `HyperMOEGate` / `HyperMOERouter` 为死代码，训练/推理流程未引用
- 注意：训练脚本中 FlashAttention monkey patch 被注释，走 transformers 原生 attention

## 4. Checkpoint / expert 文件结构

每个任务输出目录（`/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/Task{N}_llava_lora_ours`）：

- `adapter_model.bin` —— 全部 expert 的 LoRA 权重（`get_peft_state_maybe_zero_3`，key 形如 `base_model.model.model.layers.*.self_attn.q_proj.lora_A.default.loraA.{i}.mlp.weight`）
- `non_lora_trainables.bin` —— mm_projector、高斯统计、instance_router
- `stats.json` —— 每 expert `image_count/mean/var/text_count/mean/var` + `adaptive_w_img`
- `adapter_config.json` / `config.json` / `training_args.json` / `trainer_state.json`

恢复：`load_model_from_previous_task`（`train_MOE.py:799-841`）；推理加载：`llava/model/builder.py:25-110`。

## 5. 评测入口与持续学习指标

- 逐数据集评测：`scripts/Hyper/Eval_UCIT/eval_*.sh` → `python -m llava.eval.model_answer`（生成，temperature 0）→ `eval_deepseek_r1.py`（VQA 类，`pred.upper() == gt.upper()` 精确匹配）或 `eval_caption.py`（VizWiz/Flickr30k，COCO 指标）；结果 `<result_root>/<Dataset>/hyper-task{N}/Result.text`
- 矩阵与指标：`scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py`（唯一实现）：
  - `MAA` = 各 stage 已学任务均值再平均
  - `MFN` = 最终行 `R[N][·]` 均值
  - `MFT` = 对角线 `R[i][i]` 均值
  - `BWT` = `mean_j (R[N][j] − R[j][j])`，负值表示遗忘
- 已有产物示例：`runs/results/UCIT/06_18/continual_metrics.json`（如 R[1][1]=91.87, R[2][1]=83.2, R[2][2]=93.87）

## 6. 当前 GPU 与预计配置

`nvidia-smi`（2026-08-01 07:29）：8 张 NVIDIA GeForce RTX 4090（24 GB），全部空闲（util 0%，free ~24 GB），无其他用户进程占用。

V6 计划配置（global batch 24）：

| GPU 数 | per-device batch | grad accum | global batch |
|---|---|---|---|
| 2 | 2 | 6 | 24 |
| 3 | 2 | 4 | 24 |
| 4 | 2 | 3 | 24 |
| OOM 2 | 1 | 12 | 24 |
| OOM 3 | 1 | 8 | 24 |
| OOM 4 | 1 | 6 | 24 |

评测优先单卡。

## 7. 需修改的核心文件（V6 后续阶段）

- 新组件（独立 feature flag 隔离，全部配置化）：experts registry、composition 执行、oracle teacher、router、candidate pool、lifecycle、shadow update —— 计划落在 `compose/`（已有 `experts/adapters/oracle/eval` 骨架）与新增 `v6/` 命名空间
- 适配/复用点：`compose/experts/`（pool、checkpoint、assemble、metadata）、`compose/adapters/`（LoRA 注入与管理）、`compose/oracle/`（per-sample NLL Oracle、cache、candidate sets —— Stage 04 Oracle Teacher 直接复用其思路）、`compose/eval/metrics.py`
- 数据加载与训练循环：复用 `compose/train/`（trainer/data/arguments）
- UCIT 数据读取：沿用 `ucit_instructions/` JSON 格式 + `LazySupervisedDataset` 语义

## 8. 不应修改的兼容接口

- `scripts/Hyper/Train_UCIT*`、`Eval_UCIT*` 原始脚本与 `runs/results/UCIT/06_18` 等既有产物（基线对照用）
- `Hyper/peft/tuners/clitmoelora.py` 与 `llava/model/` 原始 Hyper-LLaVA 推理路径（feature flag off 时行为必须与现状完全一致）
- `llava/train/train_MOE.py` / `train.py` 默认行为
- 用户未提交的 format-controlled 文件（compose/data、compose/eval 新增文件及 docs/reports/format_controlled_*）—— 不得加入任何 V6 提交
- `evaluate_controlled_ab.py` 的 answer_pos-1 修复（见 §10）

## 9. 现有测试状态

- 运行方式：`python -m unittest discover -s tests/compose -v`（pyproject 无 pytest 配置，测试为 unittest 风格，无 pytest fixture 依赖）
- 环境：`/home/zhaozhuofan/miniconda3/envs/hyper`，Python 3.10.20，torch 2.3.1+cu118，peft 0.4.0，transformers 4.33.3
- 结果：**60 个测试全部通过（OK），0 失败**，无既有失败测试需记录

## 10. 工作区 V6/F2 改动检查

- `compose/eval/eval_controlled_ab.py:185-195` 确认存在 answer_pos-1 修复（LM 约定 `logits[t]` 预测 token t+1，`answer_pos = answer_token_pos - 1` 后取 logits），:239-240 同时记录 `answer_token_position` 与 `answer_logits_position` —— 修复仍在
- V6 / Shadow Update / registry 实现痕迹：全仓库无（唯一 "V6" 出现在 `docs/reports/0730_residual_expert_feasibility_summary.md` 标题，属既有 compose 总结）
- 受控实验结论保持不变：`controlled_direct_sum_status = FAILED`（STOP_COMPOSITION_CONFIRMED）；本任务新状态 `full_v6_ucit_status = NOT_YET_TESTED`
- 既有 60 测试通过，无需静默修复任何代码

## 11. Git 状态

- 新分支：`feat/v6-ucit-staged`（从 `dc7ad7c` 创建）
- 远端：`origin` = https://github.com/Fandy-Zhao/Hyper-llava.git；`upstream` = https://github.com/Ghy0501/HiDe-LLaVA.git
- 未跟踪文件（用户 format-controlled 产物，**不提交**）：`compose/data/` 5 个、`compose/eval/` 5 个、`docs/reports/format_controlled_*` 7 个、`experiments/data/`、`sample_instructions/` 多任务 JSON、`docs/reports/format_controlled_*` HTML

## 12. 冻结的分阶段计划（Stage 01 → Stage 12）

1. Stage 01：V6-off 基线复现（global batch 24，2–4 卡），mini2 + full seed42
2. Stage 02：Expert Registry 与激活上下文（feature flag）
3. Stage 03：direct_sum / rms_calibrated 组合执行
4. Stage 04：答案监督 Oracle Set Teacher
5. Stage 05：双模态 Query + 可学习欧氏 Expert Key
6. Stage 06：0/1/2 专家 Set Router（含安全回退）
7. Stage 07：旧专家充分性判断 + Residual Buffer
8. Stage 08：两槽 Keyed Candidate Pool（Top-1 稀疏更新）
9. Stage 09：贡献验证 + provisional/formal/archived 生命周期
10. Stage 10：Conditional Shadow Update
11. Stage 11：完整 V6 UCIT 闭环 + 中断恢复
12. Stage 12：消融矩阵 + 三 seed 主实验（42/43/44）+ 最终报告

约束重申：不使用双曲 Key/Poincaré 路由、不加入超过 2 专家激活、不用任务 ID 监督 Key、不用测试答案做真实推理、不临时改变任务顺序、每阶段先测试再提交。
