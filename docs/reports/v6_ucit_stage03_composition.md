# Hyper-LLaVA V6 Stage 03：双专家执行与 RMS 校准

## 结论

Stage 03 **PASSED**。Compose 已提供统一的 0/1/2 专家执行入口，支持 `base_only`、`single`、`direct_sum` 与 `rms_calibrated`。受控 direct-sum 对 4 个专家组合、每组 8 个冻结测试样本与旧实现逐项完全一致；4-rank bf16 pair smoke、ImageNet-R single smoke 和 ImageNet-R→ArxivQA mini2 均闭环通过。没有进入 Stage 04。

## 目标与架构边界

V6 方法逻辑全部保留在 `compose/`。`CompositionRuntime` 串联 `ExpertRegistry`、`ExpertActivationContext`、`AdapterBridge` 和 `ExpertComposer`；桥接层只暴露底层 adapter 的参数、独立 delta 与可恢复运行状态。Hyper 不导入 Compose，本阶段未修改任何 Hyper 源文件，也没有在 Hyper 中加入路由、候选、生命周期或 RMS 决策语义。

新增模块：

- `compose/lora/composer.py`：类型安全的模式分派、输入校验、统一诊断返回。
- `compose/lora/direct_sum.py`：固定系数直接相加，并匹配旧 bf16 `index_add_` 累加拓扑。
- `compose/lora/rms_composition.py`：冻结、对称、无学习参数的 RMS 系数。
- `compose/lora/statistics.py`：在线矩、DDP 聚合、原子 JSON、来源哈希校验。
- `compose/lora/runtime.py`：上下文进入/退出、hook 安装和异常恢复。
- `compose/lora/adapter_bridge.py`：同时桥接 Hyper LoRA 与旧 `ComposeLinear`，并保证一个模型内层类型一致。

通用 Hyper hook 修改：无。现有 Hyper 层已经能通过 Compose 侧 hook 计算指定 adapter delta，因此没有必要侵入 Hyper。V6-off 回归确认仅构造 bridge 不改变原 forward。

## Composer API 与公式

`ExpertComposer.forward(module, hidden_states, active_expert_ids, mode, runtime_context)` 返回 composed output、排序后的专家 ID、逐专家系数、delta RMS、pair output RMS、delta cosine 与运行诊断。调用者不直接拼接 LoRA A/B。

- `base_only`：`h = h_base`。
- `single(i)`：严格沿用 Stage 02/Hyper 单专家 forward 与原 scaling。
- `direct_sum(i,j)`：`h = h_base + delta_i + delta_j`，系数固定为 `(1,1)`，无归一化、clipping 或 gate。
- `rms_calibrated(i,j)`：`h = h_base + (1/sqrt(2)) * (c_i,l delta_i + c_j,l delta_j)`；令训练校准集上的层级 delta RMS 为 `r_i,r_j`，则 `c_i,l=sqrt(r_j/r_i)`、`c_j,l=sqrt(r_i/r_j)`，clip 为 `[0.25,4.0]`，epsilon 为 `1e-8`。任一 RMS 不大于 epsilon 时两个系数都回退到 1 并记录 fallback。

专家 ID 在组合前规范化为升序，因此 `[0,1]` 与 `[1,0]` 数值一致；重复 ID、超过两个、未注册或归档专家均显式拒绝。

冻结配置 SHA-256：`29eacd7c25718e085a864a742b3c0d8ab343759b2be958202b11cc846923d0f2`。RMS 公式实现 SHA-256：`6891823fafc6fd8936b88cc37a1a5defc927b9757aa886b21572e4555c312da5`。

## RMS 统计与防泄漏

每条统计以 `expert_id/layer_name/module_name/target_module_type/statistic_version` 为键，保存 sample count、delta/output/base-output RMS、均值、方差、dtype、校准 split 及 checkpoint/dataset/config 哈希。张量批归约使用 fp32，跨批 Welford 标量合并使用 fp64；支持 `all_reduce` 后重建全局矩。JSON 通过同目录临时文件、fsync 和 `os.replace` 原子落盘。

正式统计只取每个 pair 对应训练数据 `train_eval.json` 的前 16 个预注册样本，split=`train_calibration`、seed=42；样本 ID 与 split hash 均写入产物。测试集只在冻结配置后评估一次，未用于公式、pair scale、clip 或参数选择。加载时 checkpoint、dataset manifest 或 composition config 任一哈希变化都会使缓存失效；test split 在构造时拒绝；未注册或归档专家的统计在加载时拒绝。

预注册后曾在任何测试评估前，将极慢的逐激活 fp64 物化改成 fp32 批归约，Welford fp64 合并、样本、公式和配置均未改变，并保留原始失败日志与修订 manifest。正式诊断后增加了“归档专家拒绝加载”校验，并使 DDP `sample_count` 与数值矩一样做全局求和；正式单进程 RMS 数值、公式与配置均不变。最终 `statistics.py` SHA-256 为 `9525c8c98eb2c52053dfb1066094aa94a4db79de5bbb74f9d98923d1f2312618`。

## active/trainable 与测试

`trainable ⊆ active`。空集、冻结单专家、训练单专家、冻结 pair、旧专家冻结且只训练新专家，以及两个专家同时训练均受支持。上下文精确快照并恢复 adapter 启用状态、选择与每个参数的 `requires_grad`；异常与嵌套上下文均恢复。base 参数保持冻结，旧专家无梯度，新专家有非零梯度。

完整 Compose unittest：**83/83 PASS**（包括新增 Stage 03 的 9 个测试）。覆盖 base/single/direct 数学回归、顺序交换、零 delta、输入拒绝、梯度隔离、双 trainable、bf16 finite、在线矩、epsilon、原子 round-trip、三类哈希失效、test split、归档/未注册统计与 mocked DDP。最终真实 4-rank NCCL pair smoke 进一步确认 4 个 rank 的统计完全一致，全局 sample count 都为 16。

## Controlled direct-sum 回归

复用既有 A、Independent B、C checkpoint，并从既有专家组装新的 `a_residual_b` checkpoint；没有重训或覆盖旧 checkpoint。每个组合使用 8 个冻结测试样本，比较完整 logits、A/B answer logits、answer-token NLL、prediction 与 pair output。

| Pair | max abs logit diff | mean abs diff | A/B diff | NLL diff | agreement |
|---|---:|---:|---:|---:|---:|
| A + Independent B | 0 | 0 | 0 | 0 | 1.0 |
| A + Residual B | 0 | 0 | 0 | 0 | 1.0 |
| Independent B + C | 0 | 0 | 0 | 0 | 1.0 |
| Residual B + C | 0 | 0 | 0 | 0 | 1.0 |

首轮实现曾因 bf16 顺序加法产生约 1 的最大 logit 差；失败产物完整保留。修正为与旧实现相同的零缓冲 `index_add_` 拓扑后正式回归完全一致。这只修复 direct-sum 的执行等价性，没有修改受控实验结论。

## RMS controlled diagnostic

每个组合对 400 个冻结测试样本评估一次。表中 synergy 指相对最佳单专家的 NLL 改善量；RMS 变好不是本阶段通过条件。

| Pair / mode | Acc. | NLL | mean / median synergy | positive | worst 10% | harmful |
|---|---:|---:|---:|---:|---:|---:|
| A+Independent B / direct | 54.25% | 0.70547 | -0.07201 / -0.06670 | 6.50% | -0.17948 | 93.50% |
| A+Independent B / RMS | 51.50% | 0.71162 | -0.07816 / -0.06967 | 10.00% | -0.19493 | 90.00% |
| A+Residual B / direct | 100% | 0.00021 | 0.75873 / 0.78784 | 100% | 0.51054 | 0% |
| A+Residual B / RMS | 100% | 0.03474 | 0.72420 / 0.78091 | 100% | 0.35307 | 0% |
| Independent B+C / direct | 50.00% | 0.67766 | -0.27798 / -0.25535 | 25.00% | -0.72331 | 75.00% |
| Independent B+C / RMS | 57.50% | 0.62129 | -0.22161 / -0.21206 | 23.75% | -0.56965 | 76.25% |
| Residual B+C / direct | 74.75% | 0.79454 | -0.30481 / 0.11897 | 68.75% | -2.82373 | 31.25% |
| Residual B+C / RMS | 67.50% | 1.02667 | -0.53694 / 0.08242 | 52.25% | -3.00974 | 47.75% |

四组 RMS 系数 `(min,max,mean)` 分别为 `(0.5096,1.9625,1.0185)`、`(0.3070,3.2569,1.0945)`、`(0.6441,1.5526,1.0108)`、`(0.3196,3.1289,1.0474)`；224 层均无 epsilon fallback。每层 RMS 与系数保存在各 pair 的 `rms_statistics.json` 和 `summary.json`。四组 accuracy/NLL 方向一致性检查均为 true。结果显示 RMS 对 Independent B+C 改善、对 A+Independent B 与 Residual B+C 恶化、A+Residual B 准确率不变但 NLL 恶化，因此本阶段只确认机制正确，不声称修复 V6。

## Smoke 与 mini2

所有 GPU 任务显式使用物理 GPU 4,5,6,7（逻辑 0–3）。启动时 GPU 0–3 被其他用户进程明显占用，未结束或抢占任何其他用户进程。训练参数为 4 GPUs、per-device batch 2、gradient accumulation 3、effective global batch 24、seed 42；未发生 OOM，也未启用 batch fallback。

Single ImageNet-R smoke 运行 30 optimizer steps，train loss `0.767935`（Stage 02 为 `0.766878`），checkpoint 保存/重载成功，128 样本准确率 **53.91%**，与 Stage 02 smoke 相同。第一次 evaluator 因旧路径名启发式未识别多模态 checkpoint 而空跑；保留失败日志后只增加兼容 symlink 重试，未修改 evaluator，重试通过。

Pair runtime smoke 在真实 HyperMOELoraLinear、bf16、4-rank NCCL 下通过：direct/RMS 均 finite，active=`[0,1]`、trainable=`[1]`，旧专家冻结无梯度，新专家有非零梯度，4 rank 统计一致且 checkpoint 可恢复。

Mini2 single-only matrix：

```
[[49.22,   —  ],
 [50.00, 79.69]]
```

对应 MAA=57.0325、MFN=64.845、MFT=63.675、BWT=2.34。Stage 02 为 `[[47.66,—],[50.00,78.91]]`，Stage 01 为 `[[47.66,—],[50.00,80.47]]`；变化为 `+1.56 / 0.00 / +0.78` 个百分点，没有同向系统性漂移。task1/task2 train loss 分别为 `0.951044`、`0.353816`。

## 性能、显存和存储

同一 4090、相同 batch/sequence/warmup/measurement 条件下，四个 pair 的均值范围如下：base `85.1–86.6 ms/sample`（11.55–11.76 samples/s，峰值 19.03–19.09 GB）；single `128.6–131.3 ms/sample`（7.62–7.77 samples/s，19.69–20.41 GB）；direct `211.1–214.1 ms/sample`（4.67–4.74 samples/s，21.00–21.08 GB）；RMS `214.4–216.9 ms/sample`（4.61–4.66 samples/s，21.66–21.74 GB）。RMS 相对 direct 的层级校准额外开销约 2.8–3.5 ms/sample。逐模式 p50/p95 已保存在 pair summary；RMS stats 每组约 445 KB。single smoke checkpoint 为 296,696,033 bytes；新组装 controlled checkpoint 为 80,274,232 bytes。

## 失败、重试与已知限制

保留了四类失败：controlled attempt1 缺 `PYTHONPATH`（未 forward）；attempt2 暴露 bf16 累加不等价及缺失 residual pair；RMS attempt1 在任何测试评估前停止过慢 fp64 激活物化；single eval attempt1 为旧路径名启发式空结果。只终止了本任务确认归属的 RMS 进程组；没有 OOM。

已知限制：只支持最多两个专家；同一模型不能混用 Hyper 与 legacy ComposeLinear；RMS 是固定校准算子而不是 router，结果并不普遍优于 direct；正式 V6 全量 UCIT 仍未测试；pair smoke 不计作 UCIT benchmark；性能数据来自受控 4090 诊断而非生产吞吐承诺。

## 产物、前置条件与状态

所有代码、配置、manifest、逐样本结果、统计、日志、失败记录、smoke、mini2 与 metrics 位于 `experiments/runs/v6_ucit_staged/stage03_composition/`。`checksums.json` 给出除自身外每个文件的 SHA-256；大 checkpoint 保存在 `/data/ckpt/zhaozhuofan/v6_ucit_staged/stage03_composition/`，没有写入 Git。

Stage 04 前置条件已满足：Stage 03 工程验收通过、single 路径无系统性漂移、controlled direct-sum 精确复现、RMS 无测试泄漏且失败记录完整。但 Stage 04 必须由用户单独授权，本阶段在独立原子 commit 和一次 push 尝试后停止。

最终状态：**STAGE03_PASSED**。
