"""Summarize preregistered controlled residual-expert comparisons."""

import argparse
import json
import random
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _indexed(root: Path, seed: int, dataset: str, model: str):
    rows = _read_jsonl(root / "seed{}".format(seed) / dataset / model / "per_sample.jsonl")
    return {str(row["sample_id"]): row for row in rows}


def _mean(values):
    return statistics.fmean(values)


def bootstrap_ci(values, seed: int, draws: int = 5000):
    rng = random.Random(seed)
    size = len(values)
    means = sorted(_mean(values[rng.randrange(size)] for _ in range(size)) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def _summary(rows):
    values = list(rows.values())
    return {
        "mean_nll": _mean(float(row["nll"]) for row in values),
        "accuracy": _mean(float(row["correct"]) for row in values),
        "mean_generation_seconds": _mean(float(row["generation_seconds"]) for row in values),
    }


def _paired(rows_a, rows_b, field_a="nll", field_b="nll"):
    if set(rows_a) != set(rows_b):
        raise ValueError("evaluation sample IDs do not match")
    return [float(rows_a[key][field_a]) - float(rows_b[key][field_b]) for key in sorted(rows_a)]


def _seed_metrics(root: Path, seed: int):
    data = {}
    for dataset, models in {
        "B_only": ("base", "independent_b", "residual_b"),
        "A_plus_B": ("base", "expert_a", "independent_b", "residual_b", "a_independent_b", "a_residual_b", "rank16_ab", "task_ab"),
        "B_plus_C": ("base", "independent_b", "residual_b", "expert_c", "independent_b_c", "residual_b_c", "rank16_ab", "upper_bc"),
    }.items():
        data[dataset] = {model: _indexed(root, seed, dataset, model) for model in models}

    b = data["B_only"]
    b_gain = _paired(b["base"], b["residual_b"])
    b_acc_gain = _summary(b["residual_b"])["accuracy"] - _summary(b["base"])["accuracy"]

    ab = data["A_plus_B"]
    ids = sorted(ab["a_residual_b"])
    synergy_ab = [
        min(float(ab["expert_a"][key]["nll"]), float(ab["residual_b"][key]["nll"]))
        - float(ab["a_residual_b"][key]["nll"])
        for key in ids
    ]
    best_single_ab_acc = max(_summary(ab[name])["accuracy"] for name in ("expert_a", "residual_b"))
    pair_ab_acc = _summary(ab["a_residual_b"])["accuracy"]
    exclusive_ab = _mean(
        float(ab["a_residual_b"][key]["correct"] and not ab["expert_a"][key]["correct"]
              and not ab["residual_b"][key]["correct"] and not ab["rank16_ab"][key]["correct"])
        for key in ids
    )

    bc = data["B_plus_C"]
    ids_bc = sorted(bc["residual_b_c"])
    synergy_bc = [
        min(float(bc["residual_b"][key]["nll"]), float(bc["expert_c"][key]["nll"]))
        - float(bc["residual_b_c"][key]["nll"])
        for key in ids_bc
    ]
    best_single_bc_acc = max(_summary(bc[name])["accuracy"] for name in ("residual_b", "expert_c"))
    pair_bc_acc = _summary(bc["residual_b_c"])["accuracy"]

    result = {
        "seed": seed,
        "B_only": {
            "base": _summary(b["base"]), "independent_b": _summary(b["independent_b"]),
            "residual_b": _summary(b["residual_b"]),
            "residual_nll_improvement_over_base": _mean(b_gain),
            "residual_nll_improvement_95ci": bootstrap_ci(b_gain, seed),
            "residual_accuracy_improvement_over_base": b_acc_gain,
        },
        "A_plus_B": {
            "configurations": {name: _summary(rows) for name, rows in ab.items()},
            "best_single_accuracy": best_single_ab_acc,
            "pair_accuracy": pair_ab_acc,
            "rank16_accuracy": _summary(ab["rank16_ab"])["accuracy"],
            "mean_synergy": _mean(synergy_ab),
            "median_synergy": statistics.median(synergy_ab),
            "positive_synergy_rate": _mean(float(value > 0) for value in synergy_ab),
            "pair_better_than_rank16_rate": _mean(float(ab["a_residual_b"][key]["nll"] < ab["rank16_ab"][key]["nll"]) for key in ids),
            "pair_exclusive_correct_rate": exclusive_ab,
            "mean_nll_improvement_over_rank16": _mean(_paired(ab["rank16_ab"], ab["a_residual_b"])),
            "old_conditional_marginal_nll": _mean(_paired(ab["expert_a"], ab["a_residual_b"])),
        },
        "B_plus_C": {
            "configurations": {name: _summary(rows) for name, rows in bc.items()},
            "best_single_accuracy": best_single_bc_acc,
            "pair_accuracy": pair_bc_acc,
            "rank16_accuracy": _summary(bc["rank16_ab"])["accuracy"],
            "independent_pair_accuracy": _summary(bc["independent_b_c"])["accuracy"],
            "mean_synergy": _mean(synergy_bc),
            "median_synergy": statistics.median(synergy_bc),
            "positive_synergy_rate": _mean(float(value > 0) for value in synergy_bc),
            "pair_better_than_rank16_rate": _mean(float(bc["residual_b_c"][key]["nll"] < bc["rank16_ab"][key]["nll"]) for key in ids_bc),
            "mean_nll_improvement_over_rank16": _mean(_paired(bc["rank16_ab"], bc["residual_b_c"])),
            "mean_nll_improvement_over_independent_pair": _mean(_paired(bc["independent_b_c"], bc["residual_b_c"])),
            "c_conditional_marginal_nll": _mean(_paired(bc["expert_c"], bc["residual_b_c"])),
        },
    }
    abm = result["A_plus_B"]
    bcm = result["B_plus_C"]
    result["conditions"] = {
        "seen_composition": abm["mean_synergy"] > 0 and abm["median_synergy"] > 0
            and abm["pair_accuracy"] - abm["best_single_accuracy"] >= 0.01,
        "b_only_transfer": result["B_only"]["residual_nll_improvement_95ci"][0] > 0
            and result["B_only"]["residual_accuracy_improvement_over_base"] > 0,
        "unseen_composition": bcm["pair_accuracy"] > bcm["best_single_accuracy"]
            and bcm["pair_accuracy"] > bcm["independent_pair_accuracy"]
            and bcm["pair_accuracy"] >= bcm["rank16_accuracy"]
            and bcm["mean_nll_improvement_over_independent_pair"] > 0
            and bcm["mean_nll_improvement_over_rank16"] >= 0,
        "two_composition_contribution": abm["old_conditional_marginal_nll"] > 0
            and bcm["c_conditional_marginal_nll"] > 0,
        "nll_accuracy_direction_consistent": abm["mean_synergy"] > 0
            and abm["pair_accuracy"] > abm["best_single_accuracy"]
            and bcm["mean_synergy"] > 0 and bcm["pair_accuracy"] > bcm["best_single_accuracy"],
    }
    return result


def _aggregate(seed_results):
    paths = {
        "B_only.residual_nll_improvement_over_base": ("B_only", "residual_nll_improvement_over_base"),
        "B_only.residual_accuracy_improvement_over_base": ("B_only", "residual_accuracy_improvement_over_base"),
        "A_plus_B.mean_synergy": ("A_plus_B", "mean_synergy"),
        "A_plus_B.median_synergy": ("A_plus_B", "median_synergy"),
        "A_plus_B.pair_accuracy": ("A_plus_B", "pair_accuracy"),
        "A_plus_B.best_single_accuracy": ("A_plus_B", "best_single_accuracy"),
        "A_plus_B.rank16_accuracy": ("A_plus_B", "rank16_accuracy"),
        "A_plus_B.pair_exclusive_correct_rate": ("A_plus_B", "pair_exclusive_correct_rate"),
        "B_plus_C.mean_synergy": ("B_plus_C", "mean_synergy"),
        "B_plus_C.median_synergy": ("B_plus_C", "median_synergy"),
        "B_plus_C.pair_accuracy": ("B_plus_C", "pair_accuracy"),
        "B_plus_C.best_single_accuracy": ("B_plus_C", "best_single_accuracy"),
        "B_plus_C.independent_pair_accuracy": ("B_plus_C", "independent_pair_accuracy"),
        "B_plus_C.rank16_accuracy": ("B_plus_C", "rank16_accuracy"),
    }
    result = {}
    for name, (section, field) in paths.items():
        values = [float(item[section][field]) for item in seed_results]
        result[name] = {"mean": _mean(values), "std": statistics.pstdev(values), "values": values}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--failure-file", required=True)
    parser.add_argument("--marginal-file", required=True)
    args = parser.parse_args()
    root = Path(args.evaluation_root)
    seeds = [_seed_metrics(root, seed) for seed in SEEDS]
    condition_counts = {
        name: sum(bool(item["conditions"][name]) for item in seeds)
        for name in seeds[0]["conditions"]
    }
    passed = all(count == 3 for count in condition_counts.values())
    if passed:
        decision = "PROCEED_TO_KEY_ROUTER"
    elif sum(condition_counts.values()) == 0:
        decision = "STOP_COMPOSITION"
    else:
        decision = "REDESIGN_EXPERT_FORMATION"
    result = {
        "seeds": seeds,
        "aggregate": _aggregate(seeds),
        "condition_pass_counts_out_of_3": condition_counts,
        "stage_f2_passed": passed,
        "decision": decision,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    failures = []
    for seed in SEEDS:
        for dataset, pair, singles in (
            ("A_plus_B", "a_residual_b", ("expert_a", "residual_b")),
            ("B_plus_C", "residual_b_c", ("residual_b", "expert_c")),
        ):
            pair_rows = _indexed(root, seed, dataset, pair)
            single_rows = [_indexed(root, seed, dataset, name) for name in singles]
            for sample_id, row in pair_rows.items():
                best = min(float(values[sample_id]["nll"]) for values in single_rows)
                synergy = best - float(row["nll"])
                if synergy < 0 or not row["correct"]:
                    failures.append({
                        "seed": seed, "dataset": dataset, "sample_id": sample_id,
                        "pair_prediction": row["prediction"], "target": row["target"],
                        "pair_correct": row["correct"], "synergy": synergy,
                    })
    failures.sort(key=lambda item: (item["synergy"], item["pair_correct"]))
    failure_path = Path(args.failure_file)
    with failure_path.open("w", encoding="utf-8") as handle:
        json.dump(failures[:200], handle, indent=2, sort_keys=True)
        handle.write("\n")
    marginal_path = Path(args.marginal_file)
    with marginal_path.open("w", encoding="utf-8") as handle:
        for seed in SEEDS:
            comparisons = (
                ("B_only", "base", "residual_b", "base_minus_residual"),
                ("A_plus_B", "expert_a", "a_residual_b", "old_minus_old_residual"),
                ("B_plus_C", "expert_c", "residual_b_c", "c_minus_residual_c"),
            )
            for dataset, reference_name, residual_name, metric_name in comparisons:
                reference = _indexed(root, seed, dataset, reference_name)
                residual = _indexed(root, seed, dataset, residual_name)
                for sample_id in sorted(reference):
                    handle.write(json.dumps({
                        "seed": seed, "dataset": dataset, "sample_id": sample_id,
                        "metric": metric_name,
                        "marginal_nll_contribution": float(reference[sample_id]["nll"])
                            - float(residual[sample_id]["nll"]),
                    }, sort_keys=True) + "\n")
    print(json.dumps({"stage_f2_passed": passed, "condition_counts": condition_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
