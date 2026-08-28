# Expert LoRA Pool 完整性审计（Phase A / A2）

日期：2026-08-28 ｜ 模式：只读（未修改任何文件） ｜ 审计对象：V6.2 repaired 正式实验

实验根目录：`/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_ucit_v62_formal_seed42_repaired_20260823`

## 1. 结论摘要

- **选定的正式 expert pool：`task5/committed/pool_0011`（12 个专家，全部 frozen）**，作为 one-shot key 实验的唯一专家来源。
- 12 个 pool（pool_0000 → pool_0011，累积式）的 `compose_experts.bin` 实测 sha256 与 `assembly.json`、snapshot 注册表、`expert_pool_after.json` 的 lora_hash **三方交叉一致**。
- 12 个专家的权重完整可加载：torch 2.3.1 实际加载验证，5376/5376 张量，missing 0 / unexpected 0，共 239,861,760 参数（rank-8 bf16，32 层 × 7 模块 × 12 专家）。
- **专家形成顺序可完整恢复**（见 §4），满足 one-shot key 按 creation order 训练的前提。
- 唯一遗留门：task0 quality_gate `formal_pool_oracle_at_least_87 = false`（oracle 86.5 < 87.0），已通过 `COMPLETE_WITH_TASK0_GATE_WAIVER` 正式豁免，不影响本实验（本实验不使用该阈值做判定）。

## 2. 选择 pool_0011 的理由

1. 它是 V6.2 repaired 正式实验**最终累积池**，包含全部 12 个专家（expert 0–11）。
2. 它是 **task5 结束时的 frozen 状态**：expert_registry.json（pool_version=13）记录 active 12/12、trainable 空、rejected 空、全部 `status=frozen`、`lifecycle_status=formal`。
3. one-shot key 的 `key_metadata.checkpoint_sha256` 绑定、RMS calibration、runtime_config 均在 pool_0011 顶层 JSON 中完整存在。
4. 它的 bin sha256（`0c2b3a63…`）被 `frozen_expert_checksums_before.json`（router-bestset run）再次确认已冻结。

## 3. 逐 pool 校验表

| Task | Pool | bin 大小 | assembly.json 记录 sha | 实测 sha256 一致性 |
|---|---|---|---|---|
| task0 | pool_0000 | 40,129,746 B | 3f552c92…d1208 | 一致 |
| task1 | pool_0001 | 80,259,538 B | 354300bc…3f62 | 一致 |
| task2 | pool_0002 | 120,389,930 B | 414bc766…a6d | 一致 |
| task3 | pool_0003 | 160,520,618 B | 292e33c7…bb5e | 一致 |
| task3 | pool_0004 | 200,651,242 B | 264f88f9…ada | 一致 |
| task3 | pool_0005 | 240,781,866 B | 6dce5507…97a | 一致 |
| task3 | pool_0006 | 280,912,554 B | 4dd8fe99…488 | 一致 |
| task4 | pool_0007 | 321,043,178 B | 47f094a5…97f | 一致 |
| task4 | pool_0008 | 361,173,802 B | 534aa856…86d | 一致 |
| task4 | pool_0009 | 401,304,490 B | d2f62f6e…dab | 一致 |
| task4 | pool_0010 | 441,435,562 B | 041c6ae3…cfc0 | 一致 |
| task5 | pool_0011 | 481,566,634 B | 0c2b3a63…4c00 | 一致 |

## 4. 专家清单与创建顺序（权威来源：task5/snapshots/task5/expert_registry.json）

| expert_id | creation_task | rank | alpha | checkpoint 目录 | registry.checkpoint_sha256 vs 磁盘 | 最终池 |
|---|---|---|---|---|---|---|
| 0 | 0 | 8 | 16.0 | 存在 | 匹配 | pool_0000 |
| 1 | 1 | 8 | 16.0 | 存在 | 匹配 | pool_0001 |
| 2 | 2 | 8 | 16.0 | 存在 | 匹配 | pool_0002 |
| 3–6 | 3 | 8 | 16.0 | 存在 | 匹配 | pool_0006 |
| 7–10 | 4 | 8 | 16.0 | 存在 | 匹配 | pool_0010 |
| 11 | 5 | 8 | 16.0 | 存在 | 匹配 | pool_0011 |

- 创建顺序（one-shot 训练的 key 冻结顺序）：**0 → 1 → 2 → 3,4,5,6 → 7,8,9,10 → 11**，与 task 时序一一对应。
- checkpoint_path 语义：registry 中每个专家指向**其所在 task 合入后的最终 pool 目录**（task3 四专家共用 pool_0006，task4 四专家共用 pool_0010），这些目录均存在且 bin sha 匹配。

## 5. 与 pool 一致性的补充验证

- `expert_pool_after.json` 的 pool_version 与记录专家数与对应 pool 文件一致（task0:2/1 专家 … task5:13/12 专家）。
- 每专家在 expert_pool_after 中的 `lora_hash` 等于其所在 task 最终 pool 的 bin sha256。
- `lora/cluster_training/` 下每 task 一个目录，其中 `compose_experts.bin` 与 committed 最终 pool 完全一致（task0–5 全部验证）。
- 单专家 adapter 文件 `expert_XXXX.pt`（40,140,738 B）存在且 task0 的 sha 与 quality_gate 的 `expert_lora_sha256["0"]` 一致。
- RMS：`rms_calibration.json` 对 224 模块 × 12 专家共 2688 个 kappa 值全部有值、无 null；provenance checkpoint_hash = pool_0011 bin sha，自洽。

## 6. 注意事项（均不构成数据缺失，不阻断 GO）

1. 组合视图 `compose_experts.json` 中专家级字段（checkpoint_sha256/creation_task 等 12 个）为 null —— 权威值从 snapshot 注册表读取。
2. 组合视图 rank/lora_alpha 为占位默认（rank=1/alpha=1.0），真实值 rank=8/alpha=16.0。
3. committed 与 cluster_training 的 JSON 元数据视图不同（bin 相同）—— 提交时改写，非数据变动。
4. 中间 pool（pool_0003~0005、pool_0007~0009）不被注册表引用，仅作累积快照。
5. task4 失败重试残留（`logs/FAILED.task4_s6_*` 两个 partial chunks）不在任何正式池/注册表中。

## 7. 对 Phase B+ 的加载指引

- 权重：`task5/committed/pool_0011/compose_experts.bin`（torch.load 直接可用，纯 state_dict，键 `model.layers.{0-31}.{q,k,v,o_proj|gate,up,down_proj}.experts.{0-11}.lora_{A,B}.weight`）。
- 元数据：从 `task5/snapshots/task5/expert_registry.json` 读取权威 per-expert 字段；RMS 与 runtime_config 从 pool_0011 JSON 顶层读取。
- 加载后校验：per-expert sha256 对照 registry（本审计已三方验证，加载代码中保留 checksum 断言）。
