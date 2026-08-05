# V6 UCIT Engineering Closure — Stage E0：仓库与 UCIT 路径审计

- 日期：2026-08-05
- 分支：`exp/compose-root-cause-diagnosis`
- 审计基线 HEAD：`ed35b4844bcc4605f6f81a5c27b82fbc18def2f9`
- 类型：audit-only，零行为修改
- 产物：本报告 + `artifacts/v6_ucit_engineering/stage_e0/component_matrix.json` + `artifacts/v6_ucit_engineering/stage_e0/task_sequence.json`

## 1. UCIT 正式任务序列（已确认，多源一致）

**正式序列：ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k（cur_task 0–5）**

证据（4 个独立来源一致，无冲突）：

| 来源 | 角色 | 一致性 |
|---|---|---|
| `scripts/Hyper/Train_UCIT/Task{1..6}.sh` | 权威运行脚本 | 一致 |
| `scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py:11-18` | 指标矩阵硬编码 | 一致 |
| `scripts/Hyper/Train_UCIT_LlaVANext/Task{1..6}.sh` | 基线方法训练脚本 | 一致 |
| `experiments/runs/v6_ucit_staged/stage00_audit/stage_report.md`（2026-08-01 锁定） | 既有审计 | 一致 |

`Train_UCIT_AIRFCV`（ArxivQA 优先）与 `Train_UCIT_IFRCAV`（IconQA 优先）是独立变体实验序列，不属于正式运行，不构成任务书第三条第 1 点的"配置不一致"；本批次与下一批正式运行均固定使用主序列，**两任务 Dry Run 即为 ImageNet-R → ArxivQA**（与任务书默认一致，无需改序）。

任务序列明细见 `artifacts/v6_ucit_engineering/stage_e0/task_sequence.json`。

## 2. 数据与计算资源

- **数据根**：`/data/dataset/zhaozhuofan/UCIT/`（`instructions/` 6 任务指令 + `datasets/` 6 任务图片；仓库 `ucit_instructions` 为符号链接）。
- **格式**：train = LLaVA conversations 格式 `{id, image, conversations:[{human}, {gpt}]}`；test = `{question_id, image, text, answer}`（`image` 为相对路径，相对 `datasets/`）。
- **规模**：ImageNet-R train 23,998 / test 3,000；ArxivQA train 40,000 / test 3,000。图片总量约 54 GB。
- **GPU**：8×RTX 4090 24GB。GPU 0/1 完全空闲；GPU 2 有 ~14GB 空闲；GPU 3–7 被其他用户（libolin）占用。**Dry Run 使用 GPU 0,1（2 卡 DDP）**。
- **磁盘**：`/data` 63T，空闲 3.9T（94% 已用）。六任务 checkpoint + 特征 + 快照估计 < 1T（见交接包 resource_estimate），充足。
- **内存**：503G 总量，available 464G，充足。
- **运行进程**：无本项目进程；libolin openpi 训练占用 GPU 3–7。

## 3. 原 Hyper-LLaVA 训练 / 评估入口（复用基准）

- **训练入口**：`scripts/Hyper/Train_UCIT/Task{1..6}.sh` → `deepspeed --include ... llava/train/train_mem_MOE.py`（`llava/train/train_MOE.py::train()`）。超参：LLaVA-v1.5-7B、LoRA r48/α96、lr 2e-4、cosine、warmup 0.03、bf16、epoch 1、global batch 32–64（任务书固定 V6 为 24）。
- **Compose 变体训练入口**：`compose/train/train_compose.py::train()`（HfArgumentParser + HF Trainer 体系，deepspeed 兼容；`compose_rank=8/compose_alpha=16` 默认与 V6 一致）。Task1.sh 前置校验：mm_projector.bin 存在、数据条数、输出目录为空。
- **评估入口**：`scripts/Hyper/Eval_UCIT/eval_{imagenet,arxivqa,...}.sh` → `llava.eval.model_answer`（`llava.model.builder.load_pretrained_model`，PEFT 格式：`adapter_model.bin` + `non_lora_trainables.bin` + `stats.json`）+ `llava.eval.eval_deepseek_r1`（产出 `Result.text`：`Samples: N\nAccuracy: xx.xx%`）；持续指标 `summarize_continual_metrics.py`（MFT/MFN/MAA/BWT，目录结构 `result_root/<dataset>/hyper-task<N>/Result.text`）。
- **Compose 变体评估入口**：`compose/eval/eval_task.py`（`--adapter-kind compose`，已支持 `--expert-ids` 空/单/双显式选择）+ `compose/eval/metrics.py`（case-insensitive exact match，`answer` 字段直接兼容 UCIT test 格式）；路由评估 `compose/cli/model_answer_routed.py`（按 route manifest 逐样本决定 base_only/single/direct_sum）。
- **关键差异**：原 `llava.eval.model_answer` 不认 compose checkpoint 格式，也不支持路由选择。E10 snapshot 需导出原入口可读的格式（compose manifest / PEFT adapter）以满足 E12 验收第 14 条"原 Hyper eval 可读取两个 snapshot"。

## 4. 组件复用审计结论（详细矩阵见 component_matrix.json）

### READY（可直接复用，共 11 项）

| 组件 | 关键实现 | 位置 |
|---|---|---|
| UCIT 数据根 | 6 任务指令+图片 | `/data/dataset/zhaozhuofan/UCIT/` |
| UCIT 正式序列 | 4 源一致 | `task_sequence.json` |
| 原 Hyper 训练入口 | train_MOE + Task1-6.sh | `llava/train/`, `scripts/Hyper/Train_UCIT/` |
| 原 Hyper 评估入口 | model_answer + eval_deepseek_r1 + summarize | `llava/eval/` |
| checkpoint/日志结构 | compose manifest + registry（原子） | `compose/experts/checkpoint.py` |
| LoRA adapter manager | ExpertManager / ExpertPool / train_only 冻结 | `compose/adapters/`, `compose/experts/pool.py` |
| RMS 统计与校准 | fp64 Welford、all-reduce、hash 绑定、kappa clip | `compose/lora/statistics.py`, `rms_composition.py` |
| 答案教师搜索 | empty/single/pair + lambda_expert + 条件增益 + 缓存 18 字段绑定 | `compose/teacher/oracle_set.py`, `cache.py`, `candidate_search.py` |
| Residual Buffer | 答案教师判定（非 Router）、原子 shard、train-only | `compose/expansion/residual_buffer.py` |
| Candidate 池 | kmeans++/正交 key、top-1 稀疏分配、只算选中槽、独立优化器 | `compose/expansion/candidate_pool.py` |
| Router 基础 | 128-D 冻结、Query/Key checkpoint 原子+严格校验、tau_none/tau_second 空选择、锚点记忆 | `compose/router/*` |
| 特征提取 | 冻结 CLIP-L/14@336 image+question，L2 normalize | `compose/cli/extract_query_features.py` |
| 受控格式隔离 | 受控假设集中 4 文件，与 UCIT 不兼容，V6 不复用 | `eval_controlled_ab.py` 等 |

### NEEDS_ADAPTER（6 项）

1. **ComposeSelection**（E1）：`[batch,top_k]` 固定 1/2，`positive_count==0` 被拒，无法表示 `selected_experts=[]`；forward 中 `-1` 槽会 `KeyError`。
2. **ExpertRegistry**（E2）：缺 `created_task_id`、`key_path`、`key_sha256`、`rms_stats_path`、`mean_conditional_gain`、`key_accuracy`、`pool_version`；无 candidate/provisional/formal 生命周期；`EXPERT_STATUS` 只有 REGISTERED/FROZEN/TRAINABLE/ARCHIVED。
3. **Router**（E3）：无 training_retrieval/inference_selection 双模式；`set_losses.py` 的 `lambda_empty` 定义了未消费；router checkpoint 需补 pool_version / feature extractor version / config hash 绑定。
4. **Teacher cache**（E4）：cache key 已绑定 18 字段，需补 pool_version、router_version 绑定（任务书第 8 节字段）。
5. **CandidatePool**（E6）：`CandidatePoolConfig.slot_count` 强制 2，任务 1 需 1（`candidate_count=1`）；`compose/expansion/__init__.py` 未导出 `CandidateExpertPool`（模块导入失败）。
6. **恢复/分布式**（E10）：HF Trainer/DDP/原子写齐备，但无 V6 任务状态机与阶段级恢复。

### MISSING（6 项，全部为本批次新建）

- `configs/` 目录与三个 YAML（engineering / two_task_dry_run / formal_locked）
- 任务状态机（NOT_STARTED → COMPLETED，16 状态单向迁移）
- 专家生命周期状态机（candidate → provisional → formal → archived）
- Router 双模式
- task-boundary Snapshot + 恢复（E10）
- 交接包（批次末尾）

### BLOCKED：无

## 5. 既有验证基线

- 测试套件 `tests/compose`：**191 passed + 8 subtests**（2026-08-05 实测，环境 `hyper` conda：torch 2.3.1+cu118, python 3.10.20, pytest 可用；默认 `miniconda3/bin/python` 无 pytest）。
- Stage 00 已确认：多 LoRA 前向 `y = W0(x) + α·LoRA_A(x) + β·LoRA_B(x)`、scaling 一次、无 merge 污染、双 adapter 激活正确、checkpoint 一致、30 个正式评估 bitwise 复现。

## 6. 保护面（本批次不触碰）

- `scripts/Hyper/Train_UCIT*`、`Eval_UCIT*`（含 AIRFCV/IFRCAV/LlaVANext 变体）、`llava/model/`、`llava/train/train_MOE.py`、`Hyper/peft/`。
- 未提交的 controlled-format / p1_real 文件（`compose/data/controlled_format_v1.py` 等 25+ 个 `??` 文件）与 `docs/reports/format_controlled_*`、`experiments/runs/format_controlled_composition*` 一律不动、不提交。
- `outputs/`、`experiments/data/` 等未跟踪实验数据不提交。

## 7. 本批次工程计划（由审计驱动）

E1 统一 ComposeSelection（空/单/双）→ E2 Registry+状态机 → E3 Router 双模式 → E4 教师缓存扩展 → E5 Residual 拆分 → E6 Candidate 参数化+残差训练 → E7 验证与事务提交 → E8 全局教师+Router 校准 → E9 RMS → E10 Snapshot → E11 测试验收 → E12 两任务 Dry Run（GPU 0/1）→ 交接包。每 Stage 独立 commit，不 push。
