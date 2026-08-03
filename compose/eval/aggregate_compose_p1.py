"""Aggregate Compose P1 jobs, bootstrap uncertainty, and apply the frozen gate."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


PAIR_INPUTS = {
    "a_independent_b": ("A_plus_B", "expert_a", "independent_b", "rank16_ab", False),
    "a_residual_b": ("A_plus_B", "expert_a", "residual_b", "rank16_ab", False),
    "independent_b_c": ("B_plus_C", "independent_b", "expert_c", "upper_bc", True),
    "residual_b_c": ("B_plus_C", "residual_b", "expert_c", "upper_bc", True),
}


def bootstrap_mean(values, draws: int, seed: int):
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return {
        "draws": draws,
        "mean": statistics.fmean(values),
        "lower_95": means[int(0.025 * draws)],
        "upper_95": means[min(draws - 1, int(0.975 * draws))],
    }


def old_control(existing_root: Path, checkpoint_seed: int, pair_name: str):
    dataset, left, right, rank16, _ = PAIR_INPUTS[pair_name]
    payload = json.loads((existing_root / "summaries" / "seed{}.json".format(checkpoint_seed)).read_text())
    configs = payload["configs"][dataset]
    best_name = max((left, right), key=lambda name: configs[name]["accuracy"])
    return {
        "c4": {
            "source": "reused_verified_control",
            "selection_name": rank16,
            "accuracy": configs[rank16]["accuracy"],
            "answer_token_nll": configs[rank16]["mean_answer_token_nll"],
            "brier_score": configs[rank16]["brier"],
            "ece": configs[rank16]["ece_15_bins"],
        },
        "c5": {
            "source": "reused_verified_control",
            "selection_name": best_name,
            "accuracy": configs[best_name]["accuracy"],
            "answer_token_nll": configs[best_name]["mean_answer_token_nll"],
            "brier_score": configs[best_name]["brier"],
            "ece": configs[best_name]["ece_15_bins"],
        },
    }


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--existing-controls",
        default="experiments/runs/format_controlled_composition_v1",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_draws < 2000:
        raise ValueError("formal P1 requires at least 2000 bootstrap draws")
    root = Path(args.output_root)
    summaries = sorted((root / "predictions" / "p1").glob("*/summary.json"))
    if len(summaries) != 12:
        raise RuntimeError("P1 requires exactly 12 complete pair/seed summaries; found {}".format(len(summaries)))

    all_runs = []
    per_sample = {}
    bootstrap = {}
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "COMPLETED" or summary.get("test_used_for_c3_search") is not False:
            raise RuntimeError("invalid or leaky P1 summary {}".format(summary_path))
        pair_name = summary["pair_name"]
        analysis_seed = int(summary["analysis_seed"])
        checkpoint_seed = int(summary["checkpoint_seed"])
        if checkpoint_seed != {0: 42, 1: 43, 2: 44}[analysis_seed]:
            raise RuntimeError("analysis/checkpoint seed provenance mismatch")
        controls = old_control(Path(args.existing_controls), checkpoint_seed, pair_name)
        rows = [json.loads(line) for line in (summary_path.parent / "per_sample.jsonl").read_text().splitlines() if line]
        for mode, metrics in {**summary["modes"], **controls}.items():
            all_runs.append({
                "stage": "p1",
                "pair_name": pair_name,
                "unseen": PAIR_INPUTS[pair_name][4],
                "analysis_seed": analysis_seed,
                "checkpoint_seed": checkpoint_seed,
                "config": mode,
                "accuracy": metrics["accuracy"],
                "answer_token_nll": metrics["answer_token_nll"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "mean_synergy": metrics.get("mean_synergy"),
                "median_synergy": metrics.get("median_synergy"),
                "positive_synergy_rate": metrics.get("positive_synergy_rate"),
                "worst_10_percent_synergy": metrics.get("worst_10_percent_synergy"),
                "left_given_right_mean_gain": metrics.get("left_given_right_mean_gain"),
                "right_given_left_mean_gain": metrics.get("right_given_left_mean_gain"),
                "accuracy_delta_vs_best_single": metrics.get("accuracy_delta_vs_best_single"),
                "mean_latency_seconds": metrics.get("mean_latency_seconds"),
                "peak_memory_bytes": summary["peak_memory_bytes"] if mode not in ("c4", "c5") else None,
                "max_expert_contribution_share": summary["max_c3_expert_contribution_share"] if mode == "c3" else None,
                "checkpoint": summary["checkpoint"] if mode not in ("c4", "c5") else metrics["source"],
            })
        for mode in ("c0", "c1", "c2", "c3"):
            synergy = []
            for row in rows:
                single = min(row["modes"]["single_left"]["nll"], row["modes"]["single_right"]["nll"])
                synergy.append(single - row["modes"][mode]["nll"])
            key = "{}:seed{}:{}".format(pair_name, analysis_seed, mode)
            bootstrap[key] = bootstrap_mean(synergy, args.bootstrap_draws, analysis_seed * 100 + len(bootstrap))
            per_sample[(pair_name, analysis_seed, mode)] = synergy

    metrics_dir = root / "metrics"
    decisions_dir = root / "gate_decisions"
    reports_dir = root / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_runs[0])
    with (metrics_dir / "p1_all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_runs)

    aggregate = []
    for pair_name in PAIR_INPUTS:
        for mode in ("base", "single_left", "single_right", "c0", "c1", "c2", "c3", "c4", "c5"):
            selected = [row for row in all_runs if row["pair_name"] == pair_name and row["config"] == mode]
            if not selected:
                continue
            aggregate.append({
                "pair_name": pair_name,
                "unseen": PAIR_INPUTS[pair_name][4],
                "config": mode,
                "seeds": len(selected),
                "accuracy_mean": mean(selected, "accuracy"),
                "accuracy_std": statistics.pstdev(float(row["accuracy"]) for row in selected),
                "answer_token_nll_mean": mean(selected, "answer_token_nll"),
                "brier_mean": mean(selected, "brier_score"),
                "ece_mean": mean(selected, "ece"),
                "mean_synergy": mean(selected, "mean_synergy") if selected[0]["mean_synergy"] is not None else None,
                "accuracy_delta_vs_best_single": mean(selected, "accuracy_delta_vs_best_single") if selected[0]["accuracy_delta_vs_best_single"] is not None else None,
            })
    with (metrics_dir / "p1_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (metrics_dir / "p1_bootstrap.json").write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n")

    pair_decisions = {}
    for pair_name, (_, _, _, _, unseen) in PAIR_INPUTS.items():
        if not unseen:
            continue
        candidates = {}
        for mode in ("c2", "c3"):
            runs = [row for row in all_runs if row["pair_name"] == pair_name and row["config"] == mode]
            c0 = [row for row in all_runs if row["pair_name"] == pair_name and row["config"] == "c0"]
            ci_lower = min(bootstrap["{}:seed{}:{}".format(pair_name, seed, mode)]["lower_95"] for seed in (0, 1, 2))
            checks = {
                "pair_accuracy_beats_best_single_at_least_2_of_3": sum(float(row["accuracy_delta_vs_best_single"]) > 0 for row in runs) >= 2,
                "mean_accuracy_gain_at_least_1pp": mean(runs, "accuracy_delta_vs_best_single") >= 0.01,
                "mean_synergy_positive": mean(runs, "mean_synergy") > 0,
                "all_seed_bootstrap_lower_nonnegative": ci_lower >= 0,
                "stable_improvement_over_c0": mean(runs, "accuracy") > mean(c0, "accuracy") and mean(runs, "answer_token_nll") < mean(c0, "answer_token_nll"),
                "not_random_level": mean(runs, "accuracy") > 0.55,
                "max_contribution_at_most_90pct": mode != "c3" or max(float(row["max_expert_contribution_share"]) for row in runs) <= 0.90,
            }
            candidates[mode] = {"passed": all(checks.values()), "checks": checks, "bootstrap_min_lower_95": ci_lower}
        pair_decisions[pair_name] = {"passed": any(value["passed"] for value in candidates.values()), "candidates": candidates}

    pass_count = sum(value["passed"] for value in pair_decisions.values())
    decision = "PASS_COMPOSITION" if pass_count == 2 else "BORDERLINE_COMPOSITION" if pass_count == 1 else "FAIL_COMPOSITION"
    nll_only = any(
        mean([row for row in all_runs if row["pair_name"] == pair and row["config"] == mode], "mean_synergy") > 0
        and mean([row for row in all_runs if row["pair_name"] == pair and row["config"] == mode], "accuracy_delta_vs_best_single") <= 0
        for pair in pair_decisions for mode in ("c2", "c3")
    )
    payload = {
        "stage": "p1",
        "decision": decision,
        "nll_only_not_passed_observed": nll_only,
        "pair_decisions": pair_decisions,
        "next_stage_policy": "full_p2" if decision == "PASS_COMPOSITION" else "p2_single_seed_smoke" if decision == "BORDERLINE_COMPOSITION" else "p2_p3_single_seed_smoke_only",
        "gate_definition_frozen_before_results": True,
    }
    (decisions_dir / "p1_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Compose P1 Report", "", "Decision: **{}**.".format(decision), "",
        "Checkpoint seeds 42/43/44 are preserved explicitly and paired with analysis/bootstrap seeds 0/1/2; no checkpoint seed was relabeled.", "",
        "## Unseen-composition gate", "",
    ]
    for pair, value in pair_decisions.items():
        lines.append("- `{}`: {}".format(pair, "PASS" if value["passed"] else "FAIL"))
        for mode, detail in value["candidates"].items():
            failed = [name for name, passed in detail["checks"].items() if not passed]
            lines.append("  - `{}`: {}; failed checks: {}".format(mode, "PASS" if detail["passed"] else "FAIL", ", ".join(failed) or "none"))
    lines += ["", "Full run metrics are in `metrics/p1_all_runs.csv`; aggregates and 2,000-draw bootstrap intervals are adjacent.", ""]
    (reports_dir / "p1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
