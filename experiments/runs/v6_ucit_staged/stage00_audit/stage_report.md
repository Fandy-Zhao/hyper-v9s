# Stage 00 审计报告

- 日期：2026-08-01
- 分支：`feat/v6-ucit-staged`（从 `dc7ad7c` 创建，基线 HEAD 未移动）
- 类型：audit-only，零行为修改

## 审计结论

1. **UCIT 六任务顺序（已锁定）**：ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k（cur_task 0–5），权威来源 `scripts/Hyper/Train_UCIT/Task{1..6}.sh`，与 `summarize_continual_metrics.py` 一致。
2. **基线启动命令与超参**：deepspeed 启动 `llava/train/train_mem_MOE.py`，LLaVA-v1.5-7B，LoRA rank 48/alpha 96，epoch 1，lr 2e-4，cosine，warmup 0.03，bf16。Task 脚本实际 global batch 为 32–64（非 24）；V6 实验按任务指令固定 global batch 24（2/3/4 卡配置见 run_manifest）。
3. **专家体系**：`Hyper/peft/tuners/clitmoelora.py`，expert = 低秩 LoRA 对，训练按 `cur_task` 激活，推理按 CLIP 高斯统计 + `adaptive_w_img` argmax 路由。
4. **指标**：`scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py` 为 MFT/MFN/MAA/BWT 唯一实现，读 `Result.text` 生成 task×stage 矩阵。
5. **checkpoint 结构**：`adapter_model.bin` + `non_lora_trainables.bin` + `stats.json` + 配置文件。
6. **工作区**：无任何 V6 实现痕迹；`eval_controlled_ab.py` answer_pos-1 修复仍在（:185-195, :239-240）；format-controlled 结论保持 `controlled_direct_sum_status = FAILED`。
7. **测试**：60/60 通过，无既有失败测试。
8. **GPU**：8×RTX 4090 全部空闲。

## 需修改 / 不应修改

- 修改面（后续阶段）：新增 V6 组件落在 `compose/` 骨架之上，全部 feature flag + 配置化；复用 `compose/experts`、`compose/adapters`、`compose/oracle`、`compose/train`、`compose/eval/metrics`。
- 保护面：`scripts/Hyper/Train_UCIT*`、`Eval_UCIT*`、`Hyper/peft/tuners/clitmoelora.py`、`llava/model/`、`llava/train/train_MOE.py`、用户未提交的 format-controlled 文件一律不动、不提交。

## 提交

- commit：`chore(v6): audit UCIT pipeline and freeze staged plan`
- 内容：审计报告 + stage00_audit 目录产物；不含任何用户未跟踪文件。
- push 状态：**失败**（`fatal: could not read Username for 'https://github.com'`，无 credential helper / ssh key）。本地 commit `e240510` 保留，未强推、未修改远端配置；后续阶段开始时重试 push。
