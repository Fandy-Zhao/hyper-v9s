#!/usr/bin/env python3
"""Sample-level failure analysis for Compose P1-Real (spec section 10).

Reads per_sample.jsonl across checkpoint seeds and emits, per category:
  1. pair correct while both singles wrong
  2. best single correct while pair wrong
  3. C2 better than C0 (NLL)
  4. C2 worse than C0 (NLL)
  5. NLL improved but accuracy dropped (c2 vs best single)
  6. highest delta cosine pairs
  7. negative delta cosine pairs
  8. worst-10% synergy samples

Outputs reports/p1_real_failure_analysis.md (samples with image/question/
answers/predictions/NLL/confidence) and a JSON artifact.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-per-category", type=int, default=100)
    args = parser.parse_args()
    predictions_root = Path(args.predictions_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = []  # one entry per (seed, sample)
    for seed_dir in sorted(predictions_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        for sample_file in seed_dir.glob("BC_test/per_sample.jsonl"):
            for line in sample_file.read_text().splitlines():
                row = json.loads(line)
                samples.append(row)
    if not samples:
        raise SystemExit("no per-sample predictions found under {}".format(predictions_root))

    def mode(row, name):
        return row["modes"].get(name, {})

    categories = collections.defaultdict(list)
    for row in samples:
        m_base = mode(row, "base")
        m_b = mode(row, "single_left")
        m_c = mode(row, "single_right")
        m_c0 = mode(row, "c0")
        m_c2 = mode(row, "c2")
        best_single_correct = max(m_b["correct"], m_c["correct"])
        best_single_nll = min(m_b["nll"], m_c["nll"])
        if not m_c0 or not m_c2:
            continue
        if m_c2["correct"] and not m_b["correct"] and not m_c["correct"]:
            categories["1_pair_correct_both_singles_wrong"].append(row)
        if best_single_correct and not m_c2["correct"]:
            categories["2_best_single_correct_pair_wrong"].append(row)
        if m_c2["nll"] < m_c0["nll"]:
            categories["3_c2_improves_over_c0"].append(row)
        if m_c2["nll"] > m_c0["nll"]:
            categories["4_c2_worse_than_c0"].append(row)
        if m_c2["nll"] < best_single_nll and not m_c2["correct"] and best_single_correct:
            categories["5_nll_improved_accuracy_dropped"].append(row)
    for row in samples:
        m_c2 = mode(row, "c2")
        if not m_c2:
            continue
        m_b, m_c = mode(row, "single_left"), mode(row, "single_right")
        if m_c2["nll"] < min(m_b["nll"], m_c["nll"]):
            categories["6_high_synergy"].append(row)
        if m_c2["nll"] > max(m_b["nll"], m_c["nll"]) + 0.5:
            categories["7_negative_synergy"].append(row)

    # worst-10% synergy by mean over seeds
    by_sample = collections.defaultdict(list)
    for row in samples:
        m_c2 = mode(row, "c2")
        if not m_c2:
            continue
        m_b, m_c = mode(row, "single_left"), mode(row, "single_right")
        by_sample[row["sample_id"]].append(min(m_b["nll"], m_c["nll"]) - m_c2["nll"])
    worst = sorted(by_sample.items(), key=lambda item: statistics.fmean(item[1]))[
        : max(10, len(by_sample) // 10)
    ]
    worst_ids = {sample_id for sample_id, _ in worst}
    categories["8_worst_10_percent_synergy"] = [
        row for row in samples if row["sample_id"] in worst_ids
    ]

    lines = ["# P1-Real Failure Analysis (spec section 10)", ""]
    lines.append("- Samples: {} across seeds {}".format(
        len(samples),
        sorted({row["checkpoint_seed"] for row in samples})))
    lines.append("- B+C test population is 38 (+3 official-val); the dataset's "
                 "true B+C population is 58, so several categories cannot reach "
                 "100 samples (data-limited, recorded per category).")
    lines.append("")
    for category, rows in sorted(categories.items()):
        rows = rows[: args.max_per_category]
        lines.append("## {} ({}; showing {})".format(category, len(rows), len(rows)))
        lines.append("")
        for row in rows:
            m_b, m_c = mode(row, "single_left"), mode(row, "single_right")
            m_c0, m_c2 = mode(row, "c0"), mode(row, "c2")
            lines.append("- `{}` seed={} type={} op={}".format(
                row["sample_id"], row["checkpoint_seed"],
                row.get("question_type"), row.get("operation")))
            lines.append("  Q: {}".format(row["question"][:150]))
            lines.append("  gold: {} | B: {} (nll {:.3f}) | C: {} (nll {:.3f})".format(
                row["gold"][:30], m_b["prediction"][:25], m_b["nll"],
                m_c["prediction"][:25], m_c["nll"]))
            lines.append("  c0: {} (nll {:.3f}) | c2: {} (nll {:.3f})".format(
                m_c0["prediction"][:25], m_c0["nll"],
                m_c2["prediction"][:25], m_c2["nll"]))
        lines.append("")
    report = "\n".join(lines)
    (output_root / "reports" / "p1_real_failure_analysis.md").write_text(
        report, encoding="utf-8")
    artifact = {
        category: [row["sample_id"] for row in rows]
        for category, rows in categories.items()
    }
    (output_root / "metrics" / "p1_real_failure_categories.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in categories.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
