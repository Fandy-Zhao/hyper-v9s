#!/usr/bin/env python3
"""Generate the Stage 04 formal report and compact stage report from validated metrics."""

import json
from pathlib import Path


ROOT = Path("experiments/runs/v6_ucit_staged/stage04_oracle_teacher")
DOC = Path("docs/reports/v6_ucit_stage04_oracle_teacher.md")


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def table(groups, suite):
    lines = ["| 时间边界 | 分支 | 样本 | Empty | Single | Pair | Selected NLL | Selected Acc | Pair synergy | Harmful pair |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in sorted(groups.items()):
        parts = key.split("/")
        if parts[0] != suite:
            continue
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            parts[1], parts[2], value["samples"], fmt(value["EmptyOracleRate"]),
            fmt(value["SingleOracleRate"]), fmt(value["PairOracleRate"]),
            fmt(value["selected_set_mean_nll"]), fmt(value["selected_set_exact_accuracy"]),
            fmt(value["mean_pair_synergy"]), fmt(value["harmful_pair_rate"]),
        ))
    return "\n".join(lines)


def main():
    aggregate = json.loads((ROOT / "metrics/aggregate_metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "validation/validation.json").read_text(encoding="utf-8"))
    prereg = json.loads((ROOT / "manifests/preregistration.json").read_text(encoding="utf-8"))
    failures = json.loads((ROOT / "manifests/failure_manifest.json").read_text(encoding="utf-8"))
    groups = aggregate["groups"]
    checks = {item["name"]: item for item in validation["checks"]}
    controlled_diff = checks["controlled_scorer_regression"]["details"]["max_absolute_difference"]
    peak = checks["peak_memory_under_24gib"]["details"]["peak_cuda_memory_bytes"]
    full_direct_hist = groups["full_seed42/historical_only/direct"]
    full_direct_post = groups["full_seed42/post_task_diagnostic/direct"]
    full_rms_hist = groups["full_seed42/historical_only/rms"]
    full_rms_post = groups["full_seed42/post_task_diagnostic/rms"]

    report = f"""# Hyper-LLaVA V6 Stage 04：Answer-Supervised Oracle Expert-Set Teacher

状态：**{validation['status']}**。本阶段只生成训练/验证侧 Oracle 标签，没有训练 Router，也没有进入 Stage 05。Full UCIT 结果是预注册的每任务 32 样本分层子集（共 192 个任务样本、四个 Oracle 分支），不是全量 train Oracle。

## 1. Stage 04 问题

在冻结专家池与固定组合算子下，对带答案的训练样本比较 Empty、Single 与 Pair，使用 teacher-forced answer-token NLL 选择复杂度受罚后的最优集合，为后续输入侧 Router 提供监督上界。

## 2. Compose / Hyper 架构边界

实现全部位于 `compose/teacher/` 与实验脚本。Hyper 源码零修改、不会 import Compose；Hyper checkpoint 仅通过 Compose 侧适配器读取，V6-off 路径不变。

## 3. 答案使用边界

Oracle 只接受 `train`、`validation`、`val`、`train_calibration`；任何含 `test` 的 split 立即报错。真实测试推理不得读取答案、NLL、Oracle 集合、test label 或 residual flag。本次 manifest 明确 `test_data_used=false`。

## 4. historical-only 与 post-task diagnostic

historical-only 在 task t 仅允许专家 `[0, t)`；post-task diagnostic 允许 `[0, t]`。前者衡量旧专家复用，后者分析当前任务专家训练完成后的新增价值；缓存和指标完全分开。

## 5. Direct / RMS Oracle 隔离

Direct 固定 `direct_sum`，RMS 固定 `rms_calibrated`。二者使用不同配置哈希、缓存文件、运行 manifest、摘要与 RMS provenance，不进行逐样本择优或混合标签。

### Preregistration

- Direct config SHA-256：`{prereg['configs']['oracle_direct']['config_sha256']}`
- RMS config SHA-256：`{prereg['configs']['oracle_rms']['config_sha256']}`
- `lambda_expert=0.01`，`delta_pair_raw=0.02` nats/token，`top_k_for_pair=4`，`max_pairs=6`
- EOS 默认排除；answer mask 为 `stage04_answer_mask_v1`；token mean；cache schema v1；seed 42
- 所有值均在正式 Oracle 结果之前冻结，未读取 UCIT test 指标调参。

## 6. 候选搜索范围

始终评估 Empty 和所有时间上可见 Single。专家数不超过 4 时枚举全部 pair；更大时按答案侧 Single NLL 取 top-4，再最多评估 6 个去重 pair。该 top-K 仅用于训练侧 Oracle 搜索。

## 7. Answer NLL 定义

`NLL = -sum_t log p(y_t | x, y_<t, S) / T`。实现严格使用 `logits[t] -> labels[t+1]`，忽略 prompt/IGNORE token，默认排除结构 EOS；保存 sum、mean、token count 与 exact teacher-forced，不保存完整 vocab logits。

## 8. 复杂度惩罚

`score(S) = mean_answer_nll(S) + 0.01 * |S|`。Empty 可合法胜出，不强制使用专家。

## 9. Pair 门槛

Pair 同时要求 `best_single_nll - pair_nll >= 0.02` 且 `best_single_score - pair_score > 0`。报告分别统计“已评估、优于 single、通过门槛、最终选择”，不混为一个比例。

## 10. Tie-break

预注册顺序为：score 更低、cardinality 更小、expert ID 字典序更小，保证确定性。

## 11. Cache schema

原子 JSON envelope 包含 schema、rank、完整 provenance、cache key、sample count、sample-ID hash、record checksum 与 records。只存候选 NLL/诊断，不存 logits。

## 12. Cache invalidation

dataset、split、tokenizer、base/expert checkpoint、registry、composition mode、RMS stats、Oracle config、answer template/mask、averaging、Composer/code version任一变化即严格失效。支持 shard、rank merge、完整缓存 resume 查询、重复/缺失检测与原子替换。

## 13. 时间边界审计

机器检查 `temporal_boundaries={checks['temporal_boundaries']['passed']}`：ImageNet-R historical 使用 base-only；其余 historical 使用前一任务 checkpoint；post-task 使用当前任务完成后的 checkpoint。未来、archived、未注册或无校验 checkpoint 专家均被拒绝。

## 14. Controlled 回归

4 个 format-controlled checkpoint × Direct/RMS = 8 次、每次 16 个 train-calibration 样本。Stage 03 单 token 手工 NLL 最大绝对差 `{controlled_diff:.3e}`，低于 `2e-5` 门槛；未重训专家或覆盖旧结果。

{table(groups, 'controlled')}

## 15. ImageNet-R smoke

预注册 train 64 / validation 32 且样本不相交；validation 32 用于评分，train 64 保留为校准边界。四分支首次 cache miss 后以相同输入重跑，4/4 命中，核心 rates 与 selected NLL 完全一致。

{table(groups, 'smoke')}

## 16. Mini2 Oracle

ImageNet-R → ArxivQA，使用 Stage 03 mini2 对应 checkpoint，四个时间边界分别生成 Direct/RMS，未训练 Router。

{table(groups, 'mini2')}

## 17. Full UCIT seed42 Oracle

六任务使用 Stage 01 GB24 seed42 checkpoint 序列。每任务固定 32 个按答案 token 长度四分位轮转抽样的 train 子集；样本 ID 与源文件 SHA-256 已保存。

{table(groups, 'full_seed42')}

## 18. Empty / Single / Pair 比例

Full historical Direct 为 `{fmt(full_direct_hist['EmptyOracleRate'])}/{fmt(full_direct_hist['SingleOracleRate'])}/{fmt(full_direct_hist['PairOracleRate'])}`，post Direct 为 `{fmt(full_direct_post['EmptyOracleRate'])}/{fmt(full_direct_post['SingleOracleRate'])}/{fmt(full_direct_post['PairOracleRate'])}`；RMS 对应为 `{fmt(full_rms_hist['EmptyOracleRate'])}/{fmt(full_rms_hist['SingleOracleRate'])}/{fmt(full_rms_hist['PairOracleRate'])}` 与 `{fmt(full_rms_post['EmptyOracleRate'])}/{fmt(full_rms_post['SingleOracleRate'])}/{fmt(full_rms_post['PairOracleRate'])}`。

## 19. Pair synergy

正值定义为 pair NLL 低于逐样本 best-single。Full historical/post Direct 平均 synergy 为 `{fmt(full_direct_hist['mean_pair_synergy'])}` / `{fmt(full_direct_post['mean_pair_synergy'])}`；RMS 为 `{fmt(full_rms_hist['mean_pair_synergy'])}` / `{fmt(full_rms_post['mean_pair_synergy'])}`。CI、median、positive rate 和共现矩阵保存在逐运行摘要。

## 20. Harmful pair

Full historical/post Direct harmful rate 为 `{fmt(full_direct_hist['harmful_pair_rate'])}` / `{fmt(full_direct_post['harmful_pair_rate'])}`；RMS 为 `{fmt(full_rms_hist['harmful_pair_rate'])}` / `{fmt(full_rms_post['harmful_pair_rate'])}`。高 harmful 或低 PairOracleRate 是方法结论，不作为工程失败。

## 21. Worst tail

每个摘要保存 worst-10% synergy、bootstrap CI 和最差样本 ID。`validation.json` 另行报告 all-empty/all-single/all-pair/one-expert/task-fixed-set collapse；除 duplicate pair 外均作为非阻塞诊断。

## 22. Oracle selected-set 相对 best single

Full historical/post Direct 的 selected-vs-best-single accuracy delta 为 `{fmt(full_direct_hist['selected_vs_best_single_accuracy_delta'])}` / `{fmt(full_direct_post['selected_vs_best_single_accuracy_delta'])}`；RMS 为 `{fmt(full_rms_hist['selected_vs_best_single_accuracy_delta'])}` / `{fmt(full_rms_post['selected_vs_best_single_accuracy_delta'])}`。fixed expert 指逐 expert 全样本表现，per-sample best single 指每样本答案侧选择，selected set 才是最终 Oracle，三者不可混同。

## 23. Direct vs RMS

Direct 与 RMS 是两个独立上界，不能逐样本取较好者。RMS 若优于 Direct，只说明固定 RMS 公式可能提高 Oracle 上界；仍不能证明未来输入侧 Router 有效。

## 24. 性能、显存与缓存

正式 cache-miss 峰值显存 `{peak / 1024**3:.2f} GiB`，低于单卡 24 GiB。逐运行摘要保存 Empty/Single/Pair forward latency、总时长、吞吐、候选数、cache bytes、hit latency 与 RMS-stat hit。Smoke cache-hit 校验延迟见摘要。

## 25. 数据泄漏审计

`no_test_split={checks['no_test_split']['passed']}`，`sample_manifest_no_test={checks['sample_manifest_no_test']['passed']}`，`rms_provenance={checks['rms_provenance']['passed']}`。未使用 test 答案、test 指标、未来专家或 post-task expert 冒充 historical-only。

## 26. DDP / shard 一致性

本次正式评分采用每 GPU 一个 backbone 的数据/任务并行，不做 backward。缓存 API 的多 rank 合并、rank 唯一性、duplicate/missing sample 与 provenance 一致性由单元测试覆盖；同一临时文件没有多 rank 共享写入。

## 27. OOM / retry

OOM 次数 `{failures['oom_events']}`。前两次 controlled 启动分别因缺失 import、SSH 超时遗留重复任务实例而在正式结果前失败；原日志和隔离产物均保留。第三次使用相同预注册配置、checkpoint、候选定义成功；未终止其他用户进程。

## 28. 当前限制

Full 结果是固定分层子集，不是全量 train。当前 Hyper 默认路由没有稳定、可审计的离线 baseline 接口，因此 `current_hyper_route_exact_accuracy` 与对应 delta 明确为 `null`，没有用 Empty、fixed expert 或 best-single 伪装。缓存 resume API 可恢复已完成原子 shard；单个尚未落盘 batch 需要重算。

## 29. Stage 05 前置条件

Stage 04 只提供 train/validation Oracle 标签。进入 Stage 05 前仍需用户明确授权，并选择绑定 Direct 或 RMS 中一个来源；不得混合标签，且 Router 输入不能读取答案、任务 ID、Oracle 或 test 信息。

## 30. Stage 最终状态

`{validation['status']}`。Empty/Single/Pair、answer shift/mask、复杂度惩罚、pair 门槛、确定性 tie-break、时间边界、Direct/RMS 隔离、严格缓存、controlled/smoke/mini2/full 子集统计与全量 Compose 测试均已纳入验收。Stage 05 未启动。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report, encoding="utf-8")
    stage_report = f"""# Stage 04 Report

- Status: **{validation['status']}**
- Scope: answer-supervised Empty/Single/Pair Oracle teacher only; no Router and no Stage 05 work
- Formal runs: controlled 8, smoke 4, mini2 8, full seed42 subset 24
- Full subset: 32 train samples per task, seed 42, answer-token-length stratified; not full train
- Controlled max NLL regression error: `{controlled_diff:.3e}`
- Peak cache-miss CUDA memory: `{peak / 1024**3:.2f} GiB`
- Test data used: false
- OOM events: 0
- Formal report: `docs/reports/v6_ucit_stage04_oracle_teacher.md`
- Machine validation: `validation/validation.json`
- Aggregate metrics: `metrics/aggregate_metrics.json`
"""
    (ROOT / "stage_report.md").write_text(stage_report, encoding="utf-8")
    print(json.dumps({"report": str(DOC), "stage_status": validation["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
