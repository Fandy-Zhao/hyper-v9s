#!/usr/bin/env python3
"""Generate the final Compose P1-Real experiment report and the
synthetic-vs-real comparison table (spec sections 14-15).

Reads the aggregate/bootstrap/layerwise metrics and the synthetic P1
summary files.
"""

import argparse
import json
import statistics
from pathlib import Path


def load_metric(output_root, name):
    path = output_root / "metrics" / name
    if not path.exists():
        return None
    if name.endswith(".json"):
        return json.loads(path.read_text())
    import csv
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_lookup(rows, mode, metric):
    if not rows:
        return None
    for row in rows:
        if row["mode"] == mode and row["metric"] == metric:
            value = row["value"]
            if value in (None, ""):
                return None
            try:
                return float(value)
            except ValueError:
                return None
    return None


def synthetic_table_row(d):
    best = max(d["modes"]["single_left"]["accuracy"], d["modes"]["single_right"]["accuracy"])
    return {
        "best_single": best,
        "c0": d["modes"]["c0"]["accuracy"],
        "c1": d["modes"]["c1"]["accuracy"],
        "c2": d["modes"]["c2"]["accuracy"],
        "c3": d["modes"]["c3"]["accuracy"],
        "accuracy_delta": d["modes"]["c2"]["accuracy"] - best,
        "mean_synergy": d["modes"]["c2"]["mean_synergy"],
        "worst10": d["modes"]["c2"]["worst_10_percent_synergy"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--synthetic-root",
                        default="/home/zhaozhuofan/Hyper-LlaVA/outputs/compose_p1_p3_20260803T090000Z")
    args = parser.parse_args()
    output_root = Path(args.output_root)

    aggregate = load_metric(output_root, "p1_real_aggregate.csv")
    bootstrap = load_metric(output_root, "p1_real_bootstrap.json")
    decision = json.loads((output_root / "gate_decisions" / "p1_real_decision.json").read_text())
    layerwise = load_metric(output_root, "p1_real_layerwise.csv")
    failure = json.loads((output_root / "metrics" / "p1_real_failure_categories.json").read_text())
    all_runs = load_metric(output_root, "p1_real_all_runs.csv")

    def agg(mode, metric):
        return aggregate_lookup(aggregate, mode, metric)

    lines = ["# Compose P1-Real Experiment Report", ""]
    lines.append("- Date: 2026-08-03")
    lines.append("- Dataset: TextVQA (official annotations), B-only 19,512; "
                 "C-only 11 natural + 2,000 controlled-derived (auxiliary); "
                 "B+C 58 (17 calib + 38 test + 3 official-val)")
    lines.append("- Protocol: frozen synthetic-P1 LoRA config (rank 8, alpha 16, "
                 "dropout 0, q/k/v/o/gate/up/down, epoch 1, global batch 64, "
                 "lr 2e-4, cosine, warmup 0.03, bf16), seeds 0/1/2")
    lines.append("- Gate decision: **{}**".format(decision["decision"]))
    lines.append("")

    lines.append("## C0-C5 comparison (three-seed pooled, official VQA accuracy)")
    lines.append("")
    lines.append("| Mode | accuracy | mean_vqa_score | answer_token_nll | mean_synergy | acc delta vs best single |")
    lines.append("|------|----------|----------------|------------------|--------------|--------------------------|")
    for mode in ("base", "single_left", "single_right", "c0", "c1", "c2", "c3"):
        acc = agg(mode, "pooled_accuracy")
        vqa = agg(mode, "mean_vqa_score_3seed_mean")
        nll = agg(mode, "answer_token_nll_3seed_mean")
        syn = agg(mode, "mean_synergy_3seed_mean")
        delta = agg(mode, "accuracy_delta_vs_best_single_3seed_mean")
        if acc is None:
            continue
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {} | {} |".format(
            mode, acc, float(vqa or 0), float(nll or 0),
            "{:.4f}".format(float(syn)) if syn is not None else "-",
            "{:.4f}".format(float(delta)) if delta is not None else "-"))
    lines.append("")

    lines.append("## Bootstrap (10,000 draws, pooled over seeds)")
    lines.append("")
    for mode, stats in (bootstrap or {}).items():
        syn = stats.get("mean_synergy", {})
        vqa = stats.get("vqa_score_delta_vs_best_single", {})
        lines.append("- {}: mean synergy {:.4f} (95% CI [{:.4f}, {:.4f}], n={}); "
                     "vqa-score delta vs best single {:.4f} (95% CI [{:.4f}, {:.4f}])".format(
                         mode, syn.get("mean", 0), syn.get("ci_lower", 0),
                         syn.get("ci_upper", 0), syn.get("n", 0),
                         vqa.get("mean", 0), vqa.get("ci_lower", 0), vqa.get("ci_upper", 0)))
    lines.append("")

    lines.append("## Conditional gains (three-seed means, nats)")
    lines.append("")
    for mode in ("c0", "c1", "c2", "c3"):
        g_b = agg(mode, "G_B_given_C_3seed_mean")
        g_c = agg(mode, "G_C_given_B_3seed_mean")
        lines.append("- {}: G_B|C = {} ; G_C|B = {}".format(
            mode, "{:.4f}".format(g_b) if g_b is not None else "-",
            "{:.4f}".format(g_c) if g_c is not None else "-"))
    lines.append("")

    lines.append("## Layer diagnostics (L0-L3, single seed 0, C1 scaling)")
    lines.append("")
    lines.append("| Group | B layers | C layers | accuracy (BC_test) |")
    lines.append("|-------|----------|----------|--------------------|")
    for group in ("L0", "L1", "L2", "L3"):
        summary = output_root / "predictions" / "p1_real" / "seed0" / "BC_test_diag_{}".format(group) / "summary.json"
        if not summary.exists():
            lines.append("| {} | - | - | (not run) |".format(group))
            continue
        d = json.loads(summary.read_text())
        lines.append("| {} | {} | {} | {:.4f} |".format(
            group, "-", "-", d["modes"]["c1"]["accuracy"]))
    lines.append("")
    lines.append("Per-layer RMS and delta-cosine: `metrics/p1_real_layerwise.csv`.")
    lines.append("")

    lines.append("## Failure-category counts (per-sample, pooled seeds)")
    lines.append("")
    for category, sample_ids in sorted((failure or {}).items()):
        lines.append("- {}: {} samples".format(category, len(sample_ids)))
    lines.append("")
    lines.append("Data-limited note: the natural B+C population is 58; several "
                 "failure categories cannot reach 100 samples (spec section 10).")
    lines.append("")

    lines.append("## Gate evidence")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(decision.get("evidence", {}), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    report = "\n".join(lines)
    (output_root / "reports" / "p1_real_experiment_report.md").write_text(report, encoding="utf-8")

    # ---- synthetic vs real ----
    syn_lines = ["# Synthetic vs Real: Compose P1 Composition", ""]
    synthetic = {}
    for seed_dir in sorted(Path(args.synthetic_root).glob("predictions/p1/independent_b_c_seed*")):
        summary = seed_dir / "summary.json"
        if not summary.exists():
            continue
        d = json.loads(summary.read_text())
        synthetic[d["checkpoint_seed"]] = synthetic_table_row(d)
    syn_mean = {}
    if synthetic:
        for key in ("best_single", "c0", "c1", "c2", "c3", "accuracy_delta", "mean_synergy", "worst10"):
            syn_mean[key] = statistics.fmean(row[key] for row in synthetic.values())
    real_row = {
        "best_single": agg("single_left", "pooled_accuracy"),
        "c0": agg("c0", "pooled_accuracy"),
        "c1": agg("c1", "pooled_accuracy"),
        "c2": agg("c2", "pooled_accuracy"),
        "c3": agg("c3", "pooled_accuracy"),
        "accuracy_delta": agg("c2", "accuracy_delta_vs_best_single_3seed_mean"),
        "mean_synergy": agg("c2", "mean_synergy_3seed_mean"),
        "worst10": agg("c2", "worst_10_percent_synergy_3seed_mean"),
    }
    syn_lines.append("| Metric | Synthetic B+C | Real B+C (TextVQA) |")
    syn_lines.append("|--------|---------------|--------------------|")
    for key, label in (("best_single", "Best single"),
                       ("c0", "C0 direct sum"), ("c1", "C1 1/sqrt2"),
                       ("c2", "C2 RMS"), ("c3", "C3 scalar"),
                       ("accuracy_delta", "Accuracy delta (pair - best single)"),
                       ("mean_synergy", "Mean synergy"), ("worst10", "Worst-10% synergy")):
        s = syn_mean.get(key)
        r = real_row.get(key)
        syn_lines.append("| {} | {} | {} |".format(
            label,
            "{:.4f}".format(s) if s is not None else "n/a",
            "{:.4f}".format(r) if r is not None else "n/a"))
    syn_lines.append("")
    if bootstrap:
        ci = bootstrap.get("c2", {}).get("mean_synergy", {})
        syn_lines.append("Bootstrap lower bound (real): {:.4f}".format(ci.get("ci_lower", float("nan"))))
    syn_lines.append("")
    syn_lines.append("Note: synthetic C0-C3 accuracies are A/B-format accuracies "
                     "(0.5-0.775), real ones are official VQA accuracies; the "
                     "comparison is directional, not literal.")
    syn_lines.append("")
    syn_lines.append("## Answers to the spec section-15 questions")
    syn_lines.append("")
    answers = [
        ("1. 负迁移是否在真实数据复现", "详见 gate evidence (mean synergy / accuracy delta)"),
        ("2. RMS 校准是否仍然无效", "对比 C0 vs C2 在真实数据上的表现"),
        ("3. 失败是否集中于特定问题类型", "见 failure-category counts 与 question_type 分布"),
        ("4. 是否主要由 OCR 错误导致", "B+C 答案不匹配样本可结合 OCR tokens 审计"),
        ("5. 是否主要由数字推理错误导致", "结合 C 专家在 controlled 测试上的表现"),
        ("6. 是否存在明显 domain shift 混杂", "外部 OCR-VQA/VQAv2 结果提供 domain-shift 证据"),
        ("7. 直接全层叠加是否仍是主要失败点", "L0-L3 层组诊断直接回答"),
        ("8. 是否允许进入 expert redesign", "gate decision 决定"),
        ("9. 是否有任何理由继续当前 P2/P3", "memory: STOP_COMPOSITION_CONFIRMED 禁止 Router 阶段"),
    ]
    for question, pointer in answers:
        syn_lines.append("- {}: {}".format(question, pointer))
    (output_root / "reports" / "p1_real_vs_synthetic.md").write_text(
        "\n".join(syn_lines), encoding="utf-8")
    print("wrote reports/p1_real_experiment_report.md and reports/p1_real_vs_synthetic.md")


if __name__ == "__main__":
    main()
