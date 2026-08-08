#!/usr/bin/env python3
"""D4: sample-level Oracle gating upper bound (offline, answer-supervised).

Action set per sample: empty (base), B, C, B+C (c1), B+0.5C, B+0.25C.
Oracle = the action with the highest official VQA score on the gold answers.
This is an UPPER BOUND only: it uses the test answers and is never a
deployable result and never enters any Router training.

Inputs:
  - outputs/compose_p1_real_.../predictions/p1_real/seed{}/BC_test/per_sample.jsonl
    (base / single_left / single_right / c0..c3 vqa scores)
  - metrics/d0_metric_audit.csv        (marginal NLL per mode per sample)
  - metrics/d1_causal_controls_seed{}.csv (B+0.5C / B+0.25C vqa score + marginal NLL)

Outputs metrics/d4_oracle_actions.csv and metrics/d4_oracle_summary.json.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    p1_root = Path(args.p1_root)
    output_root = Path(args.output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)

    # d0 marginal NLL per (seed, sample, mode) from per-seed files
    d0_rows = {}
    for seed_file in sorted((output_root / "metrics").glob("d0_metric_audit_seed*.csv")):
        with seed_file.open() as handle:
            for row in csv.DictReader(handle):
                d0_rows[(int(row["checkpoint_seed"]), row["sample_id"], row["mode"])] = float(row["marginal_nll"])

    all_rows = []
    for seed in args.seeds.split(","):
        seed = int(seed)
        sample_file = p1_root / "predictions" / "p1_real" / "seed{}".format(seed) / "BC_test" / "per_sample.jsonl"
        if not sample_file.exists():
            continue
        # d1 controls for this seed
        d1 = {}
        d1_path = output_root / "metrics" / "d1_causal_controls_seed{}.csv".format(seed)
        if d1_path.exists():
            with d1_path.open() as handle:
                for row in csv.DictReader(handle):
                    d1[(row["sample_id"], row["configuration"])] = {
                        "vqa": float(row["vqa_score"]),
                        "mnll": float(row["marginal_nll"]),
                    }
        for line in sample_file.read_text().splitlines():
            sample = json.loads(line)
            sample_id = sample["sample_id"]
            modes = sample["modes"]
            actions = {}
            actions["empty"] = (float(modes["base"]["vqa_score"]),
                                d0_rows.get((seed, sample_id, "base")))
            actions["B"] = (float(modes["single_left"]["vqa_score"]),
                            d0_rows.get((seed, sample_id, "single_left")))
            actions["C"] = (float(modes["single_right"]["vqa_score"]),
                            d0_rows.get((seed, sample_id, "single_right")))
            actions["B+C"] = (float(modes["c1"]["vqa_score"]),
                              d0_rows.get((seed, sample_id, "c1")))
            for name, key in (("B+0.5C", "B+0.5C"), ("B+0.25C", "B+0.25C")):
                hit = d1.get((sample_id, key))
                if hit is not None:
                    actions[name] = (hit["vqa"], hit["mnll"])
            if not actions:
                continue
            # Oracle: highest vqa_score; ties broken by lower marginal NLL
            # (a deployable gate would prefer the expert/pair action)
            best_action = max(actions, key=lambda k: (actions[k][0],
                                                      - (actions[k][1] if actions[k][1] is not None else 1e9)))
            best_score = actions[best_action][0]
            all_wrong = all(v[0] < 2.0 / 3.0 for v in actions.values())
            best_single = max(actions["B"][0], actions["C"][0])
            all_rows.append({
                "sample_id": sample_id,
                "checkpoint_seed": seed,
                "question_type": sample.get("question_type"),
                "operation": sample.get("operation"),
                "all_actions_wrong": int(all_wrong),
                "oracle_action": best_action,
                "oracle_vqa": best_score,
                "oracle_correct": int(best_score >= 2.0 / 3.0),
                "best_single_vqa": best_single,
                "best_single_correct": int(best_single >= 2.0 / 3.0),
                "pair_vqa": actions["B+C"][0],
                "pair_correct": int(actions["B+C"][0] >= 2.0 / 3.0),
                "oracle_marginal_nll": actions[best_action][1],
                "pair_marginal_nll": actions["B+C"][1],
                "B_vqa": actions["B"][0],
                "C_vqa": actions["C"][0],
            })

    if not all_rows:
        raise SystemExit("no oracle rows produced (check inputs)")
    with (output_root / "metrics" / "d4_oracle_actions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    from collections import Counter
    action_counts = Counter(r["oracle_action"] for r in all_rows)
    summary = {
        "samples": len(all_rows),
        "oracle_accuracy": statistics.fmean(r["oracle_correct"] for r in all_rows),
        "best_single_accuracy": statistics.fmean(r["best_single_correct"] for r in all_rows),
        "fixed_pair_accuracy": statistics.fmean(r["pair_correct"] for r in all_rows),
        "oracle_action_counts": dict(action_counts),
        "pair_needed_samples": sum(1 for r in all_rows if r["oracle_action"] in ("B+C", "B+0.5C", "B+0.25C")),
        "all_actions_wrong_count": sum(r["all_actions_wrong"] for r in all_rows),
        "pair_strictly_better_than_best_single": sum(
            1 for r in all_rows if r["pair_vqa"] > r["best_single_vqa"]),
        "oracle_gain_over_best_single_pp": 100 * (
            statistics.fmean(r["oracle_correct"] for r in all_rows)
            - statistics.fmean(r["best_single_correct"] for r in all_rows)),
        "pair_gain_over_best_single_pp": 100 * (
            statistics.fmean(r["pair_correct"] for r in all_rows)
            - statistics.fmean(r["best_single_correct"] for r in all_rows)),
        "oracle_marginal_nll_mean": statistics.fmean(
            r["oracle_marginal_nll"] for r in all_rows if r["oracle_marginal_nll"] is not None),
        "pair_marginal_nll_mean": statistics.fmean(
            r["pair_marginal_nll"] for r in all_rows if r["pair_marginal_nll"] is not None),
        "needs_pair_by_type": dict(Counter(
            "{}::{}".format(r["question_type"], r["oracle_action"]) for r in all_rows)),
    }
    (output_root / "metrics" / "d4_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
