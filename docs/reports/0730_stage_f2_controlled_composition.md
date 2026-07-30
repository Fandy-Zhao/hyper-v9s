# Stage F2：受控功能组合验证

## 结论

Stage F2 未通过。五类预注册条件在三个 seed 上均为 `0/3`。Residual B 的参数与激活具有稳定身份，且在 B-only 上带来约 19.75–21.75 个百分点的生成准确率提升，但其 NLL 改善的逐样本 bootstrap 95% CI 在三个 seed 上都跨越 0；更关键的是，A+Residual B 在已见 A+B 上没有超过等参数 rank16，而 Residual B+C 在未见 B+C 上虽有小幅准确率优势，却产生稳定且显著的负 NLL synergy。自动汇总决策为 `STOP_COMPOSITION`。

## 数据、训练与公平性

- 功能：A=形状识别，B=计数，C=左右空间关系。
- 数据集：A-only、B-only、C-only、A+B、B+C；每个集合独立 train/val/test，分别为 1,600/200/400 样本，生成 seed 为 730，模板不跨 split 泄漏。
- 数据恢复后 36 个 instruction/eval JSON 与原 manifest 逐文件哈希完全一致；恢复 manifest SHA-256 为 `67cd556d09254d6e1c35473ae667cfb4207ab434a97faafe2770e34ce25b1883`。
- 三个训练 seed：42、43、44。每个 seed 训练 Expert A、Independent B、Expert C、Residual B、rank16 A+B、task A+B、upper B+C，共 21 个正式 GPU run。
- 所有正式训练均为 1 epoch、global batch 64、AdamW、learning rate 2e-4、cosine scheduler、warmup 0.03、bf16、LoRA scale 2。
- 单 rank-8 参数量 19,988,480；两个 rank-8 参数量 39,976,960；rank-16 参数量 39,976,960。rank16 的数据、epoch、optimizer、seed 与 scale 均匹配。
- Conditional Residual 训练始终启用并冻结 Expert A，仅更新 Residual B；无正交损失、Router、Shadow Update 或学习式 gate。

## 三个 seed 的核心结果

Synergy 定义为 `NLL(best single) - NLL(pair)`；正值才表示组合优于逐样本最佳单专家。

| Seed | B-only Residual 准确率提升 | B-only NLL 改善（95% CI） | A+B best single / pair / rank16 acc | A+B mean / median synergy | B+C best single / pair / independent pair / rank16 acc | B+C mean / median synergy |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 42 | +21.25 pp | +0.02263 `[-0.03768, 0.08051]` | 68.00 / 67.75 / 72.50% | -0.00729 / +0.00034 | 32.25 / 34.00 / 32.25 / 33.00% | -0.27521 / -0.05214 |
| 43 | +21.75 pp | +0.02114 `[-0.03831, 0.07903]` | 67.00 / 68.75 / 71.50% | -0.00602 / -0.00043 | 32.75 / 34.00 / 31.75 / 33.50% | -0.27492 / -0.04390 |
| 44 | +19.75 pp | +0.03252 `[-0.02605, 0.08963]` | 67.50 / 67.50 / 72.75% | -0.00597 / +0.00057 | 32.75 / 34.50 / 32.50 / 33.00% | -0.26840 / -0.03411 |

跨 seed 均值±总体标准差：

- A+B pair accuracy：68.00% ± 0.54 pp；best single：67.50% ± 0.41 pp；rank16：72.25% ± 0.54 pp。
- A+B mean synergy：`-0.006426 ± 0.000613`；median synergy：`+0.000158 ± 0.000426`。
- A+B PairExclusiveCorrectRate：`0.4167% ± 0.2357 pp`。
- B+C pair accuracy：34.1667% ± 0.2357 pp；best single：32.5833% ± 0.2357 pp；independent pair：32.1667% ± 0.3118 pp；rank16：33.1667% ± 0.2357 pp。
- B+C mean synergy：`-0.272845 ± 0.003142`；median synergy：`-0.043384 ± 0.007370`。

## 预注册条件审计

| 条件 | 通过 seed 数 | 结果 |
| --- | ---: | --- |
| 已见组合：A+Residual B 的 mean/median synergy > 0，且 accuracy 至少高于 best single 1 pp | 0/3 | 失败 |
| 功能迁移：Residual B 在 B-only 上 accuracy 提升且 NLL 改善 95% CI 下界 > 0 | 0/3 | 失败；三个 CI 均跨 0 |
| 未见组合：Residual B+C 同时超过 best single、Independent B+C、rank16，并有一致 NLL 优势 | 0/3 | 失败 |
| Residual B 在已见与未见两个组合中都有正条件边际贡献 | 0/3 | 失败；B+C 的 C 条件边际 NLL 均为负 |
| NLL 与生成准确率方向一致 | 0/3 | 失败 |

准确率与 NLL 明显不一致。B+C 的 pair accuracy 在三个 seed 上都略高于 best single、Independent B+C 和 rank16，但相对 Independent B+C 的平均 NLL 改善分别为 `-0.11832`、`-0.11279`、`-0.11125`，mean synergy 也稳定为约 `-0.27`。因此不能把小幅离散准确率提升解释为稳定、可路由的组合优势。

## 隔离、几何与功能身份

- 三个 seed 的旧 Expert A 均有 448 个 tensor 在训练前后逐位相等；tensor checksum 一致。
- 固定 8 个输入、16 个目标 token 位置、32,000 词表的 float32 logits 在三个 seed 上逐位相等，最大绝对差均为 0。
- 224 层 LoRA 增量中，旧 A 与 Residual B 的平均余弦分别为 `-0.000336`、`-0.000356`、`-0.000388`；Residual B 平均 delta RMS 约为 `5.03e-5`、`5.02e-5`、`4.98e-5`。
- Residual B 激活方向在 B-only、A+B、B+C 间高度一致。B-only/A+B 平均余弦为 0.99396/0.99119/0.99381，B-only/B+C 为 0.98556/0.98454/0.98710，A+B/B+C 为 0.99006/0.98924/0.99145。

这些结果证明隔离和表示身份成立，但行为组合门槛仍全部失败：表示相似不是可复用功能收益的充分条件。

## 延迟与显存

以下为 81 个成功配置的 summary 聚合；延迟是单样本平均生成秒数，显存为每次评测的 CUDA 峰值。

| 配置 | 平均生成延迟 | 平均峰值显存 | 峰值范围 |
| --- | ---: | ---: | ---: |
| Base | 0.1475 s | 16.96 GiB | 15.47–17.73 GiB |
| Residual B | 0.5730 s | 16.70 GiB | 15.51–17.76 GiB |
| A+Residual B | 0.7331 s | 16.98 GiB | 15.51–17.72 GiB |
| Residual B+C | 0.7376 s | 16.64 GiB | 15.53–17.76 GiB |
| rank16 | 0.6295 s | 16.26 GiB | 15.51–17.75 GiB |
| supervised B+C upper bound | 0.6166 s | 16.23 GiB | 15.49–17.71 GiB |

组合没有显存优势，并且双 expert 的生成延迟高于 rank16。最初 batch=8 的 `seed43/B_plus_C/independent_b` 发生 OOM；失败证据保留在 `evaluation_v2`。所有不完整项在新目录 `evaluation_retry_v1` 以 batch=4 重试，39/39 成功；合并结果由 42 个 primary 成功和 39 个 retry 成功组成，不覆盖原失败。

## 失败案例

最严重失败集中在 B+C 的计数 5 样本，并跨 seed 重复。例如 `controlled/B_plus_C/test/158` 的目标为 `5`，三个 seed 的 Residual B+C 都预测 `4`，synergy 分别约为 `-2.06`、`-2.08`、`-2.04`；`test/66` 和 `test/39` 也在三个 seed 上重复出现严重负 synergy。这不是单个异常 seed，而是稳定的组合干扰模式。

## 证据路径与哈希

- checkpoint：`/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1`
- 阶段根目录：`experiments/runs/0730_residual_expert_feasibility/stage_f2`
- 成功合并矩阵：`evaluation_merged`，81/81 配置、每项 400 条逐样本记录；来源为 primary 42 项、retry 39 项。
- 最终汇总 SHA-256：`7b443133b5f75c4404f2f4aec52f961b85bcad788f9015ba9214bb239fd93efb`
- 失败案例 SHA-256：`6a1366b129836678448138ce164d2aebc6ed57e49fa3f804557b876a15aeb42a`
- 每样本边际贡献 SHA-256：`5a893c8dccec67214ba6b184811f5e72061d2784766bfb757889f541832195b8`

## 决策

`STOP_COMPOSITION`

条件残差与受控组合均未形成稳定、超过等参数基线且 NLL/准确率一致的组合收益。停止自动专家组合方向；不实现 Query-Key Router、Set Router、Candidate Slot、Shadow Update 或其他被禁止的后续机制。
