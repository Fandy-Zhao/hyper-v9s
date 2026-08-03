# Compose（原 V6）P1--P3 正式测试报告

状态：**P1 正式失败；P2/P3 仅完成门禁允许的单种子工程诊断；禁止进入完整 continual benchmark。**

## 结论

- P1：`FAIL_COMPOSITION`。12/12 正式任务完成，checkpoint seeds 42/43/44 明确映射到分析 seeds 0/1/2，没有重标种子。
- P2：`FAIL_SLOT_SPECIALIZATION`，`formal=false`。仅运行 seed-0 cached-query 两槽诊断；一个槽被拒绝，一个槽 provisional，未形成槽间专门化证据。
- P3：`ROUTER_QUERY_INSUFFICIENT`，`formal=false`。仅运行 seed-0 cached-feature R0/R2/R3；R3 set exact accuracy 为 0.50，pair recall/precision 均为 0，router-oracle gap 为 0.50。
- 最终：`allow_full_continual_benchmark=false`。

## P1 正式结果

P1 在物理 GPU 4/5/6/7 上运行，用户已明确授权在 0--3 被占用时使用 4--7。12 项均退出码 0，无 OOM；每项约 1,023--1,035 秒，峰值 17.72--17.77 GiB，总记录 3.434 GPU-hours。

| Unseen pair | 配置 | Accuracy mean | 相对最佳单专家 | Mean synergy | Bootstrap 最小 95% 下界 | 门禁 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Independent B+C | C2 | 0.625833 | -0.246667 | -0.167291 | -0.246797 | FAIL |
| Independent B+C | C3 | 0.777500 | -0.095000 | -0.069636 | -0.109074 | FAIL |
| Residual B+C | C2 | 0.669167 | -0.304167 | -0.425426 | -0.735533 | FAIL |
| Residual B+C | C3 | 0.659167 | -0.314167 | -0.120580 | -0.261724 | FAIL |

四个候选均未满足“至少 2/3 种子胜过最佳单专家、平均准确率增益至少 1pp、平均 synergy 为正、所有种子 bootstrap 下界非负”。Residual B+C 的 C2/C3 还未稳定优于 C0。层贡献占比约束通过，因此失败不是由单专家贡献超过 90% 这一条单独造成。

## 实现与验证

- `compose/lora/rms_composition.py`：将校准冻结为逐层专家 RMS 的算术均值，并支持 C3 专家标量。
- `compose/eval/compose_p1.py` 与聚合器：验证集校准、测试集隔离、C0--C5、逐样本记录、2,000 次 bootstrap 和冻结门禁。
- `compose/expansion/candidate_pool.py`：最多两槽、独立优化器、0/1/2 provisional commit 逻辑。
- `compose/router/multilabel_router.py`：answer-free 独立多标签 Query-Key 路由、验证集阈值、最多两专家和时间可见性 mask。
- `compose/experiments/` 与启动脚本：任务 manifest、GPU 0--3 优先/4--7 授权回退、完整性检查、OOM 失败现场归档、门禁诊断和最终产物。
- 测试：`159 passed, 8 subtests passed`；`compileall`、`bash -n`、`git diff --check` 通过。

## 产物与唯一下一步

完整配置、逐样本预测、层级 RMS/cosine、日志、聚合 CSV、bootstrap、门禁 JSON、阶段报告和复现脚本位于：

`outputs/compose_p1_p3_20260803T090000Z/`

唯一建议：**不要启动完整 continual benchmark；先重新设计 functional experts，使同一冻结 P1 协议下 unseen B+C 的 C2/C3 同时获得正 synergy、非负 bootstrap 下界并稳定超过最佳单专家，再重跑 P1。**
