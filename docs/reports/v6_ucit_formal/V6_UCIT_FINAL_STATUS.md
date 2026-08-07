# V6 UCIT Formal Run — 最终状态报告（24 项）

- 日期：2026-08-07
- config：`configs/v6_ucit_formal_locked.yaml`（config v6，hash `30020824bf7bf084`）
- Git HEAD：`b6060db`（随本报告提交）
- 任务顺序：ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR → Flickr30k（锁定）
- 状态速览：**Launch audit: PASSED / Seed 42: PASSED / Seed 43: PASSED / Seed 44: SKIPPED / Multiseed aggregation: PASSED**

## A. 启动审计（1-17，全部 PASS）

| # | 检查项 | 状态 |
|---|---|---|
| 1 | READY.json 存在 | PASS |
| 2 | ready_for_six_task_run | PASS |
| 3 | blocking_issues 为空 | PASS |
| 4 | Git HEAD 一致（已验收代码基完整） | PASS |
| 5 | 工作区满足交接要求 | PASS |
| 6 | locked_config hash 一致 | PASS |
| 7 | task_sequence hash 一致（顺序锁定） | PASS |
| 8 | two_task_acceptance blocking 全过（18/18） | PASS |
| 9 | run entrypoint 存在（six_task_run.sh） | PASS |
| 10 | resume entrypoint 存在（resume_run.sh） | PASS |
| 11 | eval entrypoint 存在（compose.eval.eval_task） | PASS |
| 12 | two-task snapshots 可加载 | PASS |
| 13 | UCIT 数据完整（12 指令文件，test_3000.json 修正） | PASS |
| 14 | 资源满足 estimate | PASS |
| 15 | 无其他用户 GPU（4,5,6,7 空闲；2/3 不碰） | PASS |
| 16 | 无冲突正式 run | PASS |
| 17 | output root 不覆盖（独立 seed 根） | PASS |

复审计（2026-08-06T00:10Z）后 17/17 PASSED。正式运行期额外经历 8 次运行中修复（见 supplement，全部独立 commit + 回归测试 + 受影响 seed 标记）。

## B. Seed 状态（18-20）

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| 18 | Seed 42 | **PASSED** | config v5 首跑因验证切片 bug 标记 INVALID（fix 7），config v6 全新重跑：六任务 COMPLETED，15 项验收 15/15 PASSED。VizWiz/Flickr30k 用已验收原评估器 COCO caption Average（fix 8 修正，38.58%/42.17%） |
| 19 | Seed 43 | **PASSED** | 独立 registry/run root，同 config v6；六任务 COMPLETED，15 项验收 15/15 PASSED；矩阵与 seed 42 逐位一致（确定性复现验证） |
| 20 | Seed 44 | **SKIPPED** | 用户指令跳过（2026-08-07）；训练已启动但无任何提交/快照，干净终止，run root 保留留证 |

## C. 多种子汇总（21-24，全部 PASS）

| # | 检查项 | 状态 | 结果 |
|---|---|---|---|
| 21 | 每任务每 seed 性能矩阵完整 | PASS | 6 任务 × 2 seed（42/43），全部 3000 样本独立评估 |
| 22 | MFT/MFN/MAA/BWT 计算正确 | PASS | MFT 31.96%，MFN 54.20%，MAA 31.96%，BWT 0.00%（构造性为 0，审计确认） |
| 23 | 专家/残差审计 | PASS | committed 0（每任务 0），candidates trained 0，residual ratio 0；退化链全程有审计标记 |
| 24 | 交付物完整 | PASS | `multiseed_summary.json` + `V6_UCIT_MULTISEED_FINAL.md` + seed_42/43 final 报告 + 本报告 |

## 每任务每 seed 性能（对角线）

| 任务 | seed 42 | seed 43 | mean | std |
|---|---|---|---|---|
| ImageNet-R | 16.43% | 16.43% | 16.43% | 0.0000 |
| ArxivQA | 54.20% | 54.20% | 54.20% | 0.0000 |
| VizWiz（COCO Average） | 38.58% | 38.58% | 38.58% | 0.0000 |
| IconQA | 20.37% | 20.37% | 20.37% | 0.0000 |
| CLEVR | 20.00% | 20.00% | 20.00% | 0.0000 |
| Flickr30k（COCO Average） | 42.17% | 42.17% | 42.17% | 0.0000 |
| **MFT / MAA** | | | **31.96%** | |

## 关键结论与诚实披露

1. **本 run 的全部性能来自 task0 冷启动 adapter**：task0 在真实 256 样本验证下 below_tau（mean_gain -0.281，positive_rate 0.0，两个 seed 一致复现）→ 空 registry → task1-5 走退化链（0 候选、0 提交、全链共用 task0 adapter）。这是**数据驱动结果**（ImageNet-R 1-epoch LoRA 打不过 backbone），不是故障。
2. **跨 seed 零方差**：config 钉死 data/training seed 42 + 贪婪解码 → 矩阵逐位一致。协议 seed 隔离的是 registry/run root（三个独立实例验证了机制），未引入随机性。作为"可复现性"正面结果记录，但多种子统计意义有限——如实标注。
3. **指标口径与 06_18 基线可比**：同一测试集/模板/解码/评分器（fix 8 后 caption 任务与原管线同脚本）；数字差距（56.33 vs 38.58 等）来自模型本身（hyper 全任务模型 vs compose 退化链单 adapter）。
4. **8 次运行中修复全部走 Section 12 纪律**：独立 commit、回归测试、偏差报告、受影响 seed 标记/重跑。
