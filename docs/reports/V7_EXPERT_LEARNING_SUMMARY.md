# V7 六任务专家学习情况汇总（Task0→Task5 正式 run）

Branch `main`（= `feat/0903-v7-throughput-equivalence`，HEAD `a8e6bbb`）· 2026-09-07

数据来源 run root：
`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_gpu01_cached_query_formal_20260903`
（正式链 S0–S5 全完成；21-cell 矩阵于 2026-09-07 02:51 全部写全）

## 0. 速览

- 专家池随任务单调增长：**4 → 8 → 12 → 16 → 20 → 23（提交池）/ 24（key store）**；每个任务新增一个 4 专家 cohort（rank-8 LoRA，每专家 **19,988,480** 参数，每轮 +79,953,920）。
- 新增专家从上一任务 committed 状态续训（`--previous-checkpoint`/compose 基础），历史专家提交后冻结（`status=frozen`）；学习内容沉淀在三个载体：**LoRA 权重 + 1536 维路由 key + per-layer RMS κ 标定**。
- 六个任务的 S5 剪枝中 **t1–t4 均 0 移除**（candidate 数与提交数一致）；**t5 移除 expert #22**（origin Flickr30k，key store 以 `lifecycle='pruned'` 留档、manifest 剔除 → 提交 23 位活专家）。
- 官方 gate（对角）：88.23（ImageNet-R）/ 94.67（ArxivQA）/ 59.76（VizWiz）/ 78.67（IconQA）/ 66.00（CLEVR-Math）/ 58.04（Flickr30k），全部 PASS。

## 1. 每任务专家学习情况

数据集→任务映射：0=ImageNet-R，1=ArxivQA，2=VizWiz，3=IconQA，4=CLEVR-Math，5=Flickr30k。
新增专家由 S2（candidate 创建）加入候选池、S3 以 strict global batch 64 训练（4-rank DDP × per-device 1 × accum 16）、S4 RMS κ 标定、S5 剪枝评分后提交。

| 阶段 | 活专家 | 新增 cohort（expert_id） | 来源数据集 | S3 训练步数 | S5 剪枝 job 数 | 移除 | 官方 gate |
|---|---|---|---|---|---|---|---|
| t0 | 4 | 0–3 | ImageNet-R | 370 | 5 | 0 | 88.23 |
| t1 | 8 | 4–7 | ArxivQA | 621 | 5 | 0 | 94.67 |
| t2 | 12 | 8–11 | VizWiz | 618 | 5 | 0 | 59.76 (Avg) |
| t3 | 16 | 12–15 | IconQA | 463 | 5 | 0 | 78.67 |
| t4 | 20 | 16–19 | CLEVR-Math | 621 | 5 | 0 | 66.00 |
| t5 | 23 | 20–23（22 被剪） | Flickr30k | 609 | 9（两轮） | **#22** | 58.04 (Avg) |

（数字均直接读自各 `taskN/committed/compose_experts.json` 的 `experts[].trained_steps`、`taskN/pruning/` 的 job 产物与 `taskN_eval_control/taskN_eval_gate.done.json`。）

## 2. 专家"学到什么"——三个载体

1. **LoRA 低秩适配器权重**（task5 committed bin 实测：`lora_A (8, 4096)` / `lora_B (4096, 8)` → rank=8；manifest `adapter.alpha=16.0, dropout=0.0`；作用在全部 32 层 `q/k/v/o_proj + mlp gate/up/down_proj`）。每专家恰好 19,988,480 参数（weights 逐张量累加实测，与 manifest `adapter_parameter_count` 459,735,040 ÷ 23 一致）。S3 训练即把「上一 committed 冻结历史 + 本任务新专家」的 compose 前向在目标任务数据上端到端优化 —— 新专家学的是**在历史专家上下文下的增量任务能力**，历史专家权重不动。
2. **1536 维路由 key**（`taskN/committed/v7_keys.pt`，`query_dim=1536`）：每个专家一条 key，训练期随 Answer+Key loss 学习；测试/评测期 query 由固定 CLIP backbone 投影后 global-top2 选专家（fixed-query cache 模式下 selections 预计算、`encoder_calls=0`）。
3. **per-layer RMS κ 标定**（S4，`compose_experts.json → rms_calibration`，commit 时冻结）：每个 (层, 模块) 给每个活专家一个 RMS 缩放标量，把不同专家前向的激活幅度对齐到统一数值尺度，供 compose 前向确定性合并。

训练设置（规格 §5）：global batch 64 精确、S3 为单 epoch（每任务整轮步数见上表，数据量/64 决定）、seed 全局 42 —— 注意 **per-expert `created_seed` 字段在 committed manifest 中未落盘（null）**，无法逐专家回溯独立采样种子。

## 3. S5 剪枝轨迹

- **t1–t4：全保留。** 每任务 pruning 目录 5 个评分 job（leave-one-out NLL + official metric 产物 `selections_N.json/nll_N.json/answers_N.jsonl/official_metric_N.json`），移除数 0：`state/candidate_keys.pt` 与 `committed/v7_keys.pt` 行数相等（8/12/16/20）。
- **t5：一轮两轮共 9 个 job，最终移除 #22。** 证据链：
  - 提交前快照 `task5/state/candidate_keys.pt` = 24 行（20 历史 + 20–23 四个新候选）；
  - `task5/committed/v7_keys.pt` = 24 行但 metadata 中 expert **#22** 记 `lifecycle='pruned'`（origin_task 5）——key store 留档、不复用；
  - `task5/committed/compose_experts.json` manifest = 23 位活专家（id 0–21 + 23，**无 22**），全部 `extra.v7_lifecycle='historical'`；
  - 移除后的 reroute 在 {20, 21, 23} 上收敛，随后 committed checkpoint 以 23 专家过官方 gate（Flickr30k Average 58.04 PASS）。
  - （t5 S5 曾因他人占用计划 GPU 而 OOM，经 a8e6bbb 调度收窄修复 + `--resume-contract-rebind` 续跑，属调度事件，不改变方法/剪枝语义。）

## 4. 学习效果侧影——21-cell 行为矩阵

最终矩阵（`evaluation/continual_matrix.json` / `.md`；行=模型 stage，列=数据集，对角=官方 gate）：

| Stage | ImgNetR | ArxivQA | VizWiz | IconQA | CLEVR | Flickr |
|---|---|---|---|---|---|---|
| t0 | 88.23 | — | — | — | — | — |
| t1 | 88.23 | 94.67 | — | — | — | — |
| t2 | 88.23 | 94.67 | 59.76 | — | — | — |
| t3 | 88.23 | 94.50 | 59.76 | 78.67 | — | — |
| t4 | 88.23 | 94.50 | 59.76 | 78.67 | 66.00 | — |
| t5 | 88.23 | 94.50 | 59.74 | 78.67 | 66.00 | 58.04 |

路由行为观察（设计内确定性，非管线错误）：
- **同列分数大量重复**：如 ImageNet-R 列全 88.23、IconQA 列全 78.67、CLEVR 列 66.00×2 —— 因 fixed-query cache 的 selections 跨任务模型**逐字节相同**（selections.json sha256 已核对，如 a221bb6…），同 top2 → 同文本 → 同分；answers.jsonl 实际各异（仅 checkpoint 路径不同），selection audit 21/21 全绿（count=3000 / `encoder_calls=0` / `sequence_matches_cache=True`）。
- **VizWiz 59.74（t5 非对角）≠ 59.76** 证明路由并非退化（不同模型在长答案集上确实走出了不同结果）。
- 缓存 top2 与 1536D key 状态配合：路由完全确定性、可审计、零 encoder 调用，训练与评测两侧语义一致。

## 5. 数据完整性备注

- `compose_experts.json` 中 per-expert 的 `key_accuracy / support_count / mean_conditional_gain / positive_contribution_count` 为 v6 遗留字段，本 run 中均为 0.0 / 未填充（修剪决策依据的是 S5 leave-one-out NLL + official metric job，而非这些字段）。
- key store 与 manifest 的差异是**有意设计**：`lifecycle='pruned'` 的 key 留档以保持 id 空间完整（t5: 24 rows vs 23 live）。

## 附录 A：证据文件清单

- 池演进 / 参数 / κ / 每专家元数据：`task{N}/committed/compose_experts.json`（+ `.bin` weights，含 `rms_calibration`）
- 路由 key 状态：`task{N}/committed/v7_keys.pt`、`task{N}/state/candidate_keys.pt`
- S5 剪枝 job 产物：`task{N}/pruning/{selections,nll,answers,official_metric}_{0..N}.json(l)`、`task{N}/logs/pruning_*.log`
- 官方 gate：`task{N}_eval_control/task{N}_eval_gate.done.json`（含 predictions sha256 + selection audit）
- 21-cell：`evaluation/continual_matrix.{json,md}`、`evaluation/selections/t{stage}/task{task}/selections_audit.json`（21 个全 PASS）、`evaluation/predictions/` 逐 cell answers.jsonl
- 续跑/评测编排：`task0_eval_control/continue_tasks1_to5.py`（v4，`EXPECTED_SHA=a8e6bbbe…`，每次最多 4 卡、只排空闲 GPU）
