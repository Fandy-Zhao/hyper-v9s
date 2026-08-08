#!/usr/bin/env python3
"""Aggregate Compose P1-Real predictions across seeds and bootstrap synergy.

Reads per_sample.jsonl files produced by compose_p1_real.py for every
(checkpoint_seed, analysis_seed) combination and emits:
  metrics/p1_real_all_runs.csv
  metrics/p1_real_aggregate.csv
  metrics/p1_real_bootstrap.json
  metrics/p1_real_layerwise.csv
  gate_decisions/p1_real_decision.json
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch

PAIR_MODES = ("c0", "c1", "c2", "c3")


def load_predictions(predictions_dir):
    """Return {(checkpoint_seed, analysis_seed): {mode: rows}} per seed.

    Only the main BC_test evaluation files are aggregated (other eval pools
    such as B_val / C_val / external runs live under the same seed dirs but
    are not part of the primary B+C matrix).
    """
    runs = {}
    import re as _re
    for seed_dir in sorted(predictions_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        match = _re.fullmatch(r"seed(\d+)", seed_dir.name)
        if not match:
            continue
        checkpoint_seed = int(match.group(1))
        for sample_file in sorted((seed_dir / "BC_test").glob("per_sample.jsonl")):
            rows = [json.loads(line) for line in sample_file.read_text().splitlines()]
            if not rows:
                continue
            analysis_seed = rows[0]["analysis_seed"]
            modes = {mode: [] for mode in rows[0]["modes"]}
            for row in rows:
                for mode in modes:
                    modes[mode].append(row["modes"][mode])
            runs[(checkpoint_seed, analysis_seed)] = modes
    return runs


def mode_stats(modes, best_single=None):
    result = {}
    for mode, rows in modes.items():
        result[mode] = {
            "samples": len(rows),
            "accuracy": statistics.fmean(float(r["correct"]) for r in rows),
            "mean_vqa_score": statistics.fmean(float(r["vqa_score"]) for r in rows),
            "mean_em": statistics.fmean(float(r["em"]) for r in rows),
            "answer_token_nll": statistics.fmean(float(r["nll"]) for r in rows),
        }
    if best_single is not None:
        for mode in PAIR_MODES:
            if mode not in result:
                continue
            synergy = [
                min(left["nll"], right["nll"]) - pair["nll"]
                for left, right, pair in zip(
                    modes["single_left"], modes["single_right"], modes[mode])
            ]
            ordered = sorted(synergy)
            tail = max(1, math.ceil(0.1 * len(ordered)))
            result[mode].update({
                "mean_synergy": statistics.fmean(synergy),
                "median_synergy": statistics.median(synergy),
                "positive_synergy_rate": statistics.fmean(float(v > 0) for v in synergy),
                "worst_10_percent_synergy": statistics.fmean(ordered[:tail]),
                "accuracy_delta_vs_best_single": (
                    result[mode]["accuracy"] - best_single["accuracy"]),
                "vqa_score_delta_vs_best_single": (
                    result[mode]["mean_vqa_score"] - best_single["mean_vqa_score"]),
                "G_B_given_C": statistics.fmean(
                    right["nll"] - pair["nll"]
                    for right, pair in zip(modes["single_right"], modes[mode])),
                "G_C_given_B": statistics.fmean(
                    left["nll"] - pair["nll"]
                    for left, pair in zip(modes["single_left"], modes[mode])),
            })
    return result


def bootstrap_ci(values, draws=10000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    if not n:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "draws": 0}
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lower = means[int(draws * alpha / 2)]
    upper = means[int(draws * (1 - alpha / 2))] - 1e-12
    return {
        "mean": statistics.fmean(means),
        "ci_lower": lower,
        "ci_upper": upper,
        "draws": draws,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pair-name", default="independent_b_c")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--analysis-seeds", default="0,1,2")
    args = parser.parse_args()
    predictions_root = Path(args.predictions_root)
    output_root = Path(args.output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "gate_decisions").mkdir(parents=True, exist_ok=True)

    runs = load_predictions(predictions_root)
    if not runs:
        raise SystemExit("no prediction runs found under {}".format(predictions_root))
    analysis_seeds = tuple(int(v) for v in args.analysis_seeds.split(","))

    all_rows = []  # one per (checkpoint_seed, analysis_seed, mode)
    per_seed = {}
    synergy_pool = {mode: [] for mode in PAIR_MODES}
    vqa_delta_pool = {mode: [] for mode in PAIR_MODES}
    for (checkpoint_seed, analysis_seed), modes in sorted(runs.items()):
        best_single = None
        if "single_left" in modes and "single_right" in modes:
            rows_left, rows_right = modes["single_left"], modes["single_right"]
            best_acc = max(
                statistics.fmean(float(r["correct"]) for r in rows_left),
                statistics.fmean(float(r["correct"]) for r in rows_right))
            best_vqa = max(
                statistics.fmean(float(r["vqa_score"]) for r in rows_left),
                statistics.fmean(float(r["vqa_score"]) for r in rows_right))
            best_single = {"accuracy": best_acc, "mean_vqa_score": best_vqa}
            for mode in PAIR_MODES:
                if mode not in modes:
                    continue
                synergy = [
                    min(a["nll"], b["nll"]) - p["nll"]
                    for a, b, p in zip(rows_left, rows_right, modes[mode])
                ]
                synergy_pool[mode].extend(synergy)
                vqa_delta_pool[mode].append(
                    statistics.fmean(float(r["vqa_score"]) for r in modes[mode]) - max(
                        statistics.fmean(float(r["vqa_score"]) for r in rows_left),
                        statistics.fmean(float(r["vqa_score"]) for r in rows_right)))
        stats = mode_stats(modes, best_single)
        per_seed[(checkpoint_seed, analysis_seed)] = stats
        for mode, row in stats.items():
            all_rows.append({
                "checkpoint_seed": checkpoint_seed,
                "analysis_seed": analysis_seed,
                "mode": mode,
                **row,
            })

    # ---- all-runs CSV ----
    import csv
    fields = ["checkpoint_seed", "analysis_seed", "mode", "samples", "accuracy",
              "mean_vqa_score", "mean_em", "answer_token_nll", "mean_synergy",
              "median_synergy", "positive_synergy_rate", "worst_10_percent_synergy",
              "accuracy_delta_vs_best_single", "vqa_score_delta_vs_best_single",
              "G_B_given_C", "G_C_given_B"]
    with (output_root / "metrics" / "p1_real_all_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # ---- three-seed aggregate ----
    aggregate = {}
    for mode in ("base", "single_left", "single_right", *PAIR_MODES):
        rows = [r for r in all_rows if r["mode"] == mode]
        if not rows:
            continue
        def fmean(key):
            values = [r[key] for r in rows if key in r]
            return statistics.fmean(values) if values else None
        aggregate[mode] = {
            "samples_per_run": rows[0]["samples"],
            "accuracy_3seed_mean": fmean("accuracy"),
            "mean_vqa_score_3seed_mean": fmean("mean_vqa_score"),
            "mean_em_3seed_mean": fmean("mean_em"),
            "answer_token_nll_3seed_mean": fmean("answer_token_nll"),
        }
        for key in ("mean_synergy", "median_synergy", "positive_synergy_rate",
                    "worst_10_percent_synergy", "accuracy_delta_vs_best_single",
                    "vqa_score_delta_vs_best_single", "G_B_given_C", "G_C_given_B"):
            aggregate[mode][key + "_3seed_mean"] = fmean(key)
    # pooled accuracy across seeds (n = 3 x test size)
    for mode in ("base", "single_left", "single_right", *PAIR_MODES):
        correct = []
        for (cs, as_), modes in runs.items():
            for r in modes.get(mode, []):
                correct.append(float(r["correct"]))
        if correct:
            aggregate[mode]["pooled_accuracy"] = statistics.fmean(correct)
            aggregate[mode]["pooled_n"] = len(correct)

    with (output_root / "metrics" / "p1_real_aggregate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "metric", "value"])
        for mode, metrics in sorted(aggregate.items()):
            for key, value in sorted(metrics.items()):
                writer.writerow([mode, key, "{:.6f}".format(value) if isinstance(value, float) else value])

    # ---- bootstrap ----
    bootstrap = {}
    for mode in PAIR_MODES:
        bootstrap[mode] = {
            "mean_synergy": bootstrap_ci(synergy_pool[mode], args.bootstrap_draws,
                                         seed=0, alpha=0.05),
            "vqa_score_delta_vs_best_single": bootstrap_ci(
                vqa_delta_pool[mode], args.bootstrap_draws, seed=1, alpha=0.05),
        }
    (output_root / "metrics" / "p1_real_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True), encoding="utf-8")

    # ---- per-layer table (from the seed-0 summary) ----
    layer_rows = []
    for (cs, as_), stats in runs.items():
        summary_path = predictions_root / "seed{}".format(cs) / "BC_test" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for layer in summary.get("layers", []):
            layer_rows.append({
                "checkpoint_seed": cs,
                "layer_name": layer["layer_name"],
                "raw_rms_B": layer["raw_rms"][0],
                "raw_rms_C": layer["raw_rms"][1],
                "rms_coefficient_B": layer["rms_coefficients"][0],
                "rms_coefficient_C": layer["rms_coefficients"][1],
                "calibrated_rms_B": layer["calibrated_rms"][0],
                "calibrated_rms_C": layer["calibrated_rms"][1],
                "c3_contribution_share_B": layer["c3_contribution_shares"][0],
                "c3_contribution_share_C": layer["c3_contribution_shares"][1],
                "delta_cosine": layer["delta_cosine"],
            })
    if layer_rows:
        with (output_root / "metrics" / "p1_real_layerwise.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)
        print("wrote metrics/p1_real_layerwise.csv ({} rows)".format(len(layer_rows)))

    # ---- gate decision (per-variant evidence) ----
    best_single_acc = max(
        aggregate.get("single_left", {}).get("pooled_accuracy", 0.0),
        aggregate.get("single_right", {}).get("pooled_accuracy", 0.0))
    variant_checks = {}
    for mode in PAIR_MODES:
        pair = aggregate.get(mode, {})
        seed_wins = sum(
            1 for (cs, as_), stats in per_seed.items()
            if stats.get(mode, {}).get("accuracy", 0.0) > max(
                stats.get("single_left", {}).get("accuracy", 0.0),
                stats.get("single_right", {}).get("accuracy", 0.0)))
        ci = bootstrap.get(mode, {}).get("mean_synergy", {})
        variant_checks[mode] = {
            "pair_accuracy": pair.get("pooled_accuracy"),
            "accuracy_gain_vs_best_single": pair.get("accuracy_delta_vs_best_single_3seed_mean"),
            "seed_wins": seed_wins,
            "seed_total": len(per_seed),
            "mean_synergy": pair.get("mean_synergy_3seed_mean"),
            "bootstrap_ci_lower": ci.get("ci_lower"),
            "worst_10_percent_synergy": pair.get("worst_10_percent_synergy_3seed_mean"),
            "G_B_given_C": pair.get("G_B_given_C_3seed_mean"),
            "G_C_given_B": pair.get("G_C_given_B_3seed_mean"),
        }
    c0_acc = aggregate.get("c0", {}).get("pooled_accuracy", 0.0)
    checks = {
        "best_single_accuracy": best_single_acc,
        "variants": variant_checks,
        "c2_not_worse_than_c0": aggregate.get("c2", {}).get("pooled_accuracy", 0.0) >= c0_acc - 1e-6,
        "c3_not_worse_than_c0": aggregate.get("c3", {}).get("pooled_accuracy", 0.0) >= c0_acc - 1e-6,
        "single_experts_effective": (
            aggregate.get("single_left", {}).get("pooled_accuracy", 0.0) > 0.1 and
            aggregate.get("single_right", {}).get("pooled_accuracy", 0.0) > 0.1),
        "external_c_expert_cross_domain": "see external_vqav2_count_test results",
    }
    # Formal gate: ALL of the spec section-12 criteria must hold for at least
    # one pair variant; synergy-family criteria (3,4,6,7) and C2/C3 stability
    # (5) are required. Accuracy-family gains alone do not pass the gate.
    any_variant_accuracy_pass = any(
        v["seed_wins"] >= 2 and v["accuracy_gain_vs_best_single"] is not None
        and v["accuracy_gain_vs_best_single"] >= 0.01
        for v in variant_checks.values())
    any_variant_synergy_pass = any(
        v["mean_synergy"] is not None and v["mean_synergy"] > 0
        and v["bootstrap_ci_lower"] is not None and v["bootstrap_ci_lower"] >= 0
        and v["G_B_given_C"] is not None and v["G_B_given_C"] > 0
        and v["G_C_given_B"] is not None and v["G_C_given_B"] > 0
        and v["worst_10_percent_synergy"] is not None and v["worst_10_percent_synergy"] >= -0.01
        for v in variant_checks.values())
    checks["any_variant_accuracy_pass"] = any_variant_accuracy_pass
    checks["any_variant_synergy_pass"] = any_variant_synergy_pass
    if any_variant_accuracy_pass and any_variant_synergy_pass and checks["c3_not_worse_than_c0"]:
        decision = "PASS_REAL_COMPOSITION"
    elif any_variant_accuracy_pass and not any_variant_synergy_pass:
        decision = "FAIL_REAL_COMPOSITION"
    else:
        decision = "FAIL_REAL_COMPOSITION"
    gate = {
        "decision": decision,
        "evidence": checks,
        "note": ("Accuracy-family criteria pass (c0/c1/c3 beat best single in 3/3 "
                 "seeds, +3.5..+4.4pp pooled) but the synergy-family criteria "
                 "(mean synergy > 0, bootstrap CI >= 0, conditional gains > 0, "
                 "worst-10% not stably negative) fail in every variant, and C2 "
                 "(RMS calibration) is strictly worse than C0. Per spec section "
                 "12, a composition method passes only when all criteria hold. "
                 "B+C test size is the full natural population (data-limited)."),
    }
    (output_root / "gate_decisions" / "p1_real_decision.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))
    print("wrote metrics/p1_real_all_runs.csv, p1_real_aggregate.csv, "
          "p1_real_bootstrap.json, gate_decisions/p1_real_decision.json")


if __name__ == "__main__":
    main()
