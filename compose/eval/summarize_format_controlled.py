"""Summarize format-controlled composition evaluation.

Computes, per seed and across seeds:

  - per-configuration metrics (accuracy overall/positive/negative/balanced,
    answer-token NLL, Brier, ECE with 15 pre-registered bins, hard-negative
    stratification, diagnostic free-generation accuracy)
  - per-pair synergy statistics (mean/median/positive rate/negative rate,
    scene-cluster bootstrap 95% CI, worst-10% tail, exclusive-correct rates,
    error-set Jaccard, pair-vs-rank16 per-sample NLL)
  - conditional marginal contributions G_B_given_A / G_A_given_B /
    G_B_given_C / G_C_given_B with bootstrap CIs and stratification
  - 10,000-draw paired bootstrap per seed with the scene as the resampling
    unit and a fixed RNG seed
  - the five preregistered conditions (section 十八) and the final automatic
    decision (section 十九)

All thresholds in this file are pre-registered and must not be adjusted after
results are produced.
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List

SEEDS = (42, 43, 44)
BOOTSTRAP_DRAWS = 10_000
ECE_BINS = 15

CONFIGS = {
    "A_only": ["base", "expert_a", "independent_b", "expert_c", "residual_b"],
    "B_only": ["base", "independent_b", "residual_b", "expert_a", "expert_c"],
    "C_only": ["base", "expert_c", "independent_b", "residual_b", "expert_a"],
    "A_plus_B": [
        "base", "expert_a", "independent_b", "residual_b",
        "a_independent_b", "a_residual_b", "rank16_ab", "task_ab",
    ],
    "B_plus_C": [
        "base", "independent_b", "residual_b", "expert_c",
        "independent_b_c", "residual_b_c", "upper_bc",
    ],
}

# (dataset, pair_selection, (single_a, single_b), rank16_selection)
PAIRS = [
    ("A_plus_B", "a_independent_b", ("expert_a", "independent_b"), "rank16_ab"),
    ("A_plus_B", "a_residual_b", ("expert_a", "residual_b"), "rank16_ab"),
    ("B_plus_C", "independent_b_c", ("independent_b", "expert_c"), "upper_bc"),
    ("B_plus_C", "residual_b_c", ("residual_b", "expert_c"), "upper_bc"),
]

NEGATIVE_TYPES = ("count_negative", "attribute_negative", "relation_negative")


def _read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _indexed(root: Path, seed: int, dataset: str, model: str) -> Dict[str, dict]:
    path = root / "seed{}".format(seed) / dataset / model / "per_sample.jsonl"
    if not path.is_file():
        raise FileNotFoundError("missing evaluation output: {}".format(path))
    rows = _read_jsonl(path)
    indexed = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in indexed:
            raise ValueError("duplicate sample_id in {}".format(path))
        indexed[sample_id] = row
    return indexed


def _mean(values):
    return statistics.fmean(values)


def _config_metrics(rows: Dict[str, dict]) -> dict:
    values = list(rows.values())
    positives = [r for r in values if r["polarity"] == "positive"]
    negatives = [r for r in values if r["polarity"] == "negative"]
    pos_acc = _mean(float(r["correct"]) for r in positives) if positives else 0.0
    neg_acc = _mean(float(r["correct"]) for r in negatives) if negatives else 0.0
    ece_values = [
        (float(r["probability_A"]), 1.0 if r["target"] == "A" else 0.0)
        for r in values
    ]
    hard_negative = {}
    for key in NEGATIVE_TYPES + ("positive",):
        subset = [r for r in values if (r["negative_type"] == key if key != "positive" else r["polarity"] == "positive")]
        hard_negative[key] = {
            "count": len(subset),
            "accuracy": _mean(float(r["correct"]) for r in subset) if subset else 0.0,
            "mean_nll": _mean(float(r["answer_token_nll"]) for r in subset) if subset else 0.0,
        }
    return {
        "samples": len(values),
        "accuracy": _mean(float(r["correct"]) for r in values),
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_accuracy": 0.5 * (pos_acc + neg_acc),
        "mean_answer_token_nll": _mean(float(r["answer_token_nll"]) for r in values),
        "brier": _mean(float(r["brier"]) for r in values),
        "ece_15_bins": _ece(ece_values),
        "free_generation_accuracy": _mean(float(r["free_correct"]) for r in values),
        "hard_negative": hard_negative,
    }


def _ece(values: List[tuple], bins: int = ECE_BINS) -> float:
    bin_count = [0] * bins
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins
    for p_a, y_a in values:
        index = min(bins - 1, int(p_a * bins))
        bin_count[index] += 1
        bin_conf[index] += p_a
        bin_acc[index] += float(y_a)
    total = sum(bin_count)
    if not total:
        return 0.0
    error = 0.0
    for index in range(bins):
        if bin_count[index]:
            error += (bin_count[index] / total) * abs(
                bin_conf[index] / bin_count[index] - bin_acc[index] / bin_count[index]
            )
    return error


def _scene_clusters(rows: Dict[str, dict]) -> Dict[str, List[str]]:
    clusters: Dict[str, List[str]] = {}
    for sample_id, row in rows.items():
        clusters.setdefault(str(row["scene_id"]), []).append(sample_id)
    return clusters


def bootstrap_ci(scene_clusters: Dict[str, List[str]], compute: callable,
                 rng: random.Random, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Scene-cluster bootstrap. compute(samples) -> float; CI from percentiles."""
    scene_ids = list(scene_clusters)
    all_sample_ids = [sid for ids in scene_clusters.values() for sid in ids]
    values = []
    for _ in range(draws):
        chosen = []
        for _ in range(len(scene_ids)):
            chosen.extend(scene_clusters[rng.choice(scene_ids)])
        values.append(compute([scene_clusters[s] for s in scene_ids], chosen) if False else None)
    return values


def _bootstrap_mean_diff(clusters: Dict[str, List[str]], per_sample: Dict[str, float],
                         rng: random.Random, draws: int = BOOTSTRAP_DRAWS) -> List[float]:
    """Vectorized scene-cluster bootstrap of the mean of a per-sample value.

    Scenes are the resampling unit (section 十四); each drawn scene brings all
    of its samples, and the statistic is the sample-weighted mean.
    """
    import numpy as np

    scene_ids = list(clusters)
    sums = np.array(
        [sum(float(per_sample[sid]) for sid in clusters[scene]) for scene in scene_ids],
        dtype=np.float64,
    )
    counts = np.array([len(clusters[scene]) for scene in scene_ids], dtype=np.float64)
    flat = np.array(
        rng.choices(range(len(scene_ids)), k=draws * len(scene_ids)), dtype=np.int64
    ).reshape(draws, len(scene_ids))
    return (sums[flat].sum(axis=1) / counts[flat].sum(axis=1)).tolist()


def _ci(values: List[float]) -> List[float]:
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered))]]


def _pair_stats(data: dict, dataset: str, pair_name: str, singles: tuple,
                rank16_name: str, rng: random.Random) -> dict:
    pair = data[dataset][pair_name]
    single_a = data[dataset][singles[0]]
    single_b = data[dataset][singles[1]]
    rank16 = data[dataset][rank16_name]
    if set(pair) != set(single_a) or set(pair) != set(single_b):
        raise ValueError("sample sets differ for {} {}".format(dataset, pair_name))
    ids = sorted(pair)
    synergy = [
        min(float(single_a[sid]["answer_token_nll"]), float(single_b[sid]["answer_token_nll"]))
        - float(pair[sid]["answer_token_nll"])
        for sid in ids
    ]
    best_single_pred = {}
    for sid in ids:
        nll_a = float(single_a[sid]["answer_token_nll"])
        nll_b = float(single_b[sid]["answer_token_nll"])
        best_single_pred[sid] = single_a[sid] if nll_a <= nll_b else single_b[sid]
    pair_rank16_nll_diff = [
        float(pair[sid]["answer_token_nll"]) - float(rank16[sid]["answer_token_nll"])
        for sid in ids
    ]
    worst10_count = max(1, len(ids) // 10)
    worst10 = sorted(ids, key=lambda sid: synergy[ids.index(sid)])[:worst10_count]
    worst10_synergy = [synergy[ids.index(sid)] for sid in worst10]
    negative_synergies = [value for value in synergy if value < 0]
    worst10_negative_contribution = (
        sum(-value for value in worst10_synergy if value < 0)
        / sum(-value for value in negative_synergies)
        if negative_synergies else 0.0
    )
    clusters = _scene_clusters(pair)
    synergy_per_sample = {
        sid: min(float(single_a[sid]["answer_token_nll"]), float(single_b[sid]["answer_token_nll"]))
        - float(pair[sid]["answer_token_nll"])
        for sid in ids
    }
    synergy_ci = _ci(_bootstrap_mean_diff(clusters, synergy_per_sample, rng))
    pair_rank16_diff_per_sample = {
        sid: float(pair[sid]["answer_token_nll"]) - float(rank16[sid]["answer_token_nll"])
        for sid in ids
    }
    pair_rank16_diff_ci = _ci(_bootstrap_mean_diff(clusters, pair_rank16_diff_per_sample, rng))
    pair_exclusive = _mean(
        float(pair[sid]["correct"]) and not float(single_a[sid]["correct"])
        and not float(single_b[sid]["correct"]) for sid in ids
    )
    pair_correct_best_single_wrong = _mean(
        float(pair[sid]["correct"]) and not float(best_single_pred[sid]["correct"]) for sid in ids
    )
    best_single_correct_pair_wrong = _mean(
        float(best_single_pred[sid]["correct"]) and not float(pair[sid]["correct"]) for sid in ids
    )
    errors_pair = {sid for sid in ids if not float(pair[sid]["correct"])}
    errors_best = {
        sid for sid in ids
        if not (float(single_a[sid]["correct"]) or float(single_b[sid]["correct"]))
    }
    jaccard = (
        len(errors_pair & errors_best) / len(errors_pair | errors_best)
        if errors_pair | errors_best else 0.0
    )
    return {
        "mean_synergy": _mean(synergy),
        "median_synergy": statistics.median(synergy),
        "positive_synergy_rate": _mean(float(value > 0) for value in synergy),
        "negative_synergy_rate": _mean(float(value < 0) for value in synergy),
        "synergy_95ci": synergy_ci,
        "worst10_mean_synergy": _mean(worst10_synergy),
        "worst10_negative_magnitude_contribution": worst10_negative_contribution,
        "pair_exclusive_correct_rate": pair_exclusive,
        "pair_correct_best_single_wrong_rate": pair_correct_best_single_wrong,
        "best_single_correct_pair_wrong_rate": best_single_correct_pair_wrong,
        "error_set_jaccard": jaccard,
        "pair_vs_rank16_mean_nll_diff": _mean(pair_rank16_nll_diff),
        "pair_vs_rank16_median_nll_diff": statistics.median(pair_rank16_nll_diff),
        "pair_vs_rank16_nll_diff_95ci": pair_rank16_diff_ci,
        "pair_better_than_rank16_nll_rate": _mean(float(value < 0) for value in pair_rank16_nll_diff),
        "worst10_sample_ids": worst10,
    }


def _marginal(name: str, reference: Dict[str, dict], pair: Dict[str, dict],
              clusters, rng: random.Random) -> dict:
    ids = sorted(pair)
    values = [
        float(reference[sid]["answer_token_nll"]) - float(pair[sid]["answer_token_nll"])
        for sid in ids
    ]
    by_type = {"positive": [], "negative": {}}
    for sid in ids:
        row = pair[sid]
        if row["polarity"] == "positive":
            by_type["positive"].append(float(reference[sid]["answer_token_nll"]) - float(pair[sid]["answer_token_nll"]))
        else:
            key = str(row["negative_type"] or "other")
            by_type["negative"].setdefault(key, []).append(
                float(reference[sid]["answer_token_nll"]) - float(pair[sid]["answer_token_nll"])
            )
    per_sample = {
        sid: float(reference[sid]["answer_token_nll"]) - float(pair[sid]["answer_token_nll"])
        for sid in ids
    }
    return {
        "name": name,
        "mean": _mean(values),
        "median": statistics.median(values),
        "positive_rate": _mean(float(value > 0) for value in values),
        "95ci": _ci(_bootstrap_mean_diff(clusters, per_sample, rng)),
        "by_polarity": {
            "positive": _mean(by_type["positive"]) if by_type["positive"] else 0.0,
            "negative": {
                key: _mean(values) for key, values in by_type["negative"].items()
            },
        },
    }


def _seed_metrics(root: Path, seed: int) -> dict:
    rng = random.Random(seed)
    data = {}
    for dataset, models in CONFIGS.items():
        data[dataset] = {model: _indexed(root, seed, dataset, model) for model in models}

    configs = {
        dataset: {model: _config_metrics(rows) for model, rows in models.items()}
        for dataset, models in data.items()
    }

    pairs = {}
    marginals = {}
    for dataset, pair_name, singles, rank16_name in PAIRS:
        pair_rows = data[dataset][pair_name]
        clusters = _scene_clusters(pair_rows)
        pairs[pair_name] = _pair_stats(data, dataset, pair_name, singles, rank16_name, rng)
        marginals[pair_name] = {}
        reference = data[dataset][singles[0]]
        other = data[dataset][singles[1]]
        if dataset == "A_plus_B" and pair_name == "a_residual_b":
            marginals[pair_name]["G_B_given_A"] = _marginal(
                "G_B_given_A", data[dataset]["expert_a"], pair_rows, clusters, rng)
            marginals[pair_name]["G_A_given_B"] = _marginal(
                "G_A_given_B", data[dataset]["residual_b"], pair_rows, clusters, rng)
        elif dataset == "A_plus_B" and pair_name == "a_independent_b":
            marginals[pair_name]["G_B_given_A"] = _marginal(
                "G_B_given_A", data[dataset]["expert_a"], pair_rows, clusters, rng)
            marginals[pair_name]["G_A_given_B"] = _marginal(
                "G_A_given_B", data[dataset]["independent_b"], pair_rows, clusters, rng)
        elif dataset == "B_plus_C":
            marginals[pair_name]["G_B_given_C"] = _marginal(
                "G_B_given_C", data[dataset]["expert_c"], pair_rows, clusters, rng)
            marginals[pair_name]["G_C_given_B"] = _marginal(
                "G_C_given_B", data[dataset][singles[0]], pair_rows, clusters, rng)

    # ---- paired bootstrap: differences with scene as resampling unit --------
    ab = data["A_plus_B"]
    bc = data["B_plus_C"]
    b = data["B_only"]
    ab_clusters = _scene_clusters(ab["a_residual_b"])
    bc_clusters = _scene_clusters(bc["residual_b_c"])
    b_clusters = _scene_clusters(b["residual_b"])

    best_single_ab = max(
        ("expert_a", configs["A_plus_B"]["expert_a"]["accuracy"]),
        ("residual_b", configs["A_plus_B"]["residual_b"]["accuracy"]),
        key=lambda item: item[1],
    )[0]
    best_single_bc = max(
        ("independent_b", configs["B_plus_C"]["independent_b"]["accuracy"]),
        ("expert_c", configs["B_plus_C"]["expert_c"]["accuracy"]),
        key=lambda item: item[1],
    )[0]

    def _ci_for_clusters(clusters, per_sample):
        return _ci(_bootstrap_mean_diff(clusters, per_sample, rng))

    ab_ids = sorted(ab["a_residual_b"])
    bc_ids = sorted(bc["residual_b_c"])
    b_ids = sorted(b["residual_b"])
    bootstrap = {
        "A_plus_B": {
            "acc_pair_minus_best_single": _ci_for_clusters(ab_clusters, {
                sid: float(ab["a_residual_b"][sid]["correct"]) - float(ab[best_single_ab][sid]["correct"])
                for sid in ab_ids
            }),
            "nll_best_single_minus_pair": _ci_for_clusters(ab_clusters, {
                sid: float(ab[best_single_ab][sid]["answer_token_nll"]) - float(ab["a_residual_b"][sid]["answer_token_nll"])
                for sid in ab_ids
            }),
            "brier_best_single_minus_pair": _ci_for_clusters(ab_clusters, {
                sid: float(ab[best_single_ab][sid]["brier"]) - float(ab["a_residual_b"][sid]["brier"])
                for sid in ab_ids
            }),
            "acc_pair_minus_rank16": _ci_for_clusters(ab_clusters, {
                sid: float(ab["a_residual_b"][sid]["correct"]) - float(ab["rank16_ab"][sid]["correct"])
                for sid in ab_ids
            }),
            "nll_rank16_minus_pair": _ci_for_clusters(ab_clusters, {
                sid: float(ab["rank16_ab"][sid]["answer_token_nll"]) - float(ab["a_residual_b"][sid]["answer_token_nll"])
                for sid in ab_ids
            }),
        },
        "B_plus_C": {
            "acc_pair_minus_best_single": _ci_for_clusters(bc_clusters, {
                sid: float(bc["residual_b_c"][sid]["correct"]) - float(bc[best_single_bc][sid]["correct"])
                for sid in bc_ids
            }),
            "nll_best_single_minus_pair": _ci_for_clusters(bc_clusters, {
                sid: float(bc[best_single_bc][sid]["answer_token_nll"]) - float(bc["residual_b_c"][sid]["answer_token_nll"])
                for sid in bc_ids
            }),
            "nll_rank16_minus_pair": _ci_for_clusters(bc_clusters, {
                sid: float(bc["upper_bc"][sid]["answer_token_nll"]) - float(bc["residual_b_c"][sid]["answer_token_nll"])
                for sid in bc_ids
            }),
        },
        "B_only": {
            "residual_acc_minus_base": _ci_for_clusters(b_clusters, {
                sid: float(b["residual_b"][sid]["correct"]) - float(b["base"][sid]["correct"])
                for sid in b_ids
            }),
            "residual_nll_minus_base": _ci_for_clusters(b_clusters, {
                sid: float(b["residual_b"][sid]["answer_token_nll"]) - float(b["base"][sid]["answer_token_nll"])
                for sid in b_ids
            }),
        },
    }

    # ---- preregistered conditions (section 十八) -----------------------------
    a_only, b_only, c_only = data["A_only"], data["B_only"], data["C_only"]
    a_only_clusters = _scene_clusters(a_only["expert_a"])
    b_only_clusters = _scene_clusters(b_only["independent_b"])
    c_only_clusters = _scene_clusters(c_only["expert_c"])

    def _nll_improvement_ci(clusters, base_rows, expert_rows):
        per_sample = {
            sid: float(base_rows[sid]["answer_token_nll"]) - float(expert_rows[sid]["answer_token_nll"])
            for sid in sorted(expert_rows)
        }
        return _ci(_bootstrap_mean_diff(clusters, per_sample, rng))

    single_validity = {
        "expert_a_A_only": {
            "accuracy_above_base": configs["A_only"]["expert_a"]["accuracy"] > configs["A_only"]["base"]["accuracy"],
            "nll_improvement_mean": _mean(float(a_only["base"][sid]["answer_token_nll"]) - float(a_only["expert_a"][sid]["answer_token_nll"]) for sid in a_only["expert_a"]),
            "nll_improvement_95ci": _nll_improvement_ci(a_only_clusters, a_only["base"], a_only["expert_a"]),
            "hard_negative_accuracy": configs["A_only"]["expert_a"]["hard_negative"]["attribute_negative"]["accuracy"],
        },
        "independent_b_B_only": {
            "accuracy_above_base": configs["B_only"]["independent_b"]["accuracy"] > configs["B_only"]["base"]["accuracy"],
            "nll_improvement_mean": _mean(float(b_only["base"][sid]["answer_token_nll"]) - float(b_only["independent_b"][sid]["answer_token_nll"]) for sid in b_only["independent_b"]),
            "nll_improvement_95ci": _nll_improvement_ci(b_only_clusters, b_only["base"], b_only["independent_b"]),
            "hard_negative_accuracy": configs["B_only"]["independent_b"]["hard_negative"]["count_negative"]["accuracy"],
        },
        "expert_c_C_only": {
            "accuracy_above_base": configs["C_only"]["expert_c"]["accuracy"] > configs["C_only"]["base"]["accuracy"],
            "nll_improvement_mean": _mean(float(c_only["base"][sid]["answer_token_nll"]) - float(c_only["expert_c"][sid]["answer_token_nll"]) for sid in c_only["expert_c"]),
            "nll_improvement_95ci": _nll_improvement_ci(c_only_clusters, c_only["base"], c_only["expert_c"]),
            "hard_negative_accuracy": configs["C_only"]["expert_c"]["hard_negative"]["relation_negative"]["accuracy"],
        },
        "residual_b_B_only": {
            "accuracy_above_base": configs["B_only"]["residual_b"]["accuracy"] > configs["B_only"]["base"]["accuracy"],
            "nll_improvement_mean": _mean(float(b_only["base"][sid]["answer_token_nll"]) - float(b_only["residual_b"][sid]["answer_token_nll"]) for sid in b_only["residual_b"]),
            "nll_improvement_95ci": _nll_improvement_ci(b_only_clusters, b_only["base"], b_only["residual_b"]),
        },
    }
    condition1 = all(
        item["accuracy_above_base"]
        and item["nll_improvement_mean"] > 0
        and item["nll_improvement_95ci"][0] > 0
        and (item.get("hard_negative_accuracy", 0.5) > 0.5 if "hard_negative_accuracy" in item else True)
        for item in single_validity.values()
    )

    ab_pair = pairs["a_residual_b"]
    ab_marginals = marginals["a_residual_b"]
    ab_best_single_acc = max(
        configs["A_plus_B"]["expert_a"]["accuracy"],
        configs["A_plus_B"]["residual_b"]["accuracy"],
    )
    ab_rank16_gap = configs["A_plus_B"]["a_residual_b"]["accuracy"] - configs["A_plus_B"]["rank16_ab"]["accuracy"]
    direction_ab = (
        configs["A_plus_B"]["a_residual_b"]["accuracy"] > ab_best_single_acc
        and ab_pair["mean_synergy"] > 0
    )
    condition2 = {
        "accuracy_gain_vs_best_single_pp": (configs["A_plus_B"]["a_residual_b"]["accuracy"] - ab_best_single_acc) * 100.0,
        "passed": (
            configs["A_plus_B"]["a_residual_b"]["accuracy"] - ab_best_single_acc >= 0.01
            and ab_pair["mean_synergy"] > 0
            and ab_pair["median_synergy"] > 0
            and ab_pair["synergy_95ci"][0] > 0
            and ab_marginals["G_B_given_A"]["mean"] > 0
            and ab_marginals["G_A_given_B"]["mean"] > 0
            and direction_ab
            and ab_rank16_gap >= -0.01
        ),
        "strong_pass_over_rank16": configs["A_plus_B"]["a_residual_b"]["accuracy"] >= configs["A_plus_B"]["rank16_ab"]["accuracy"],
        "rank16_accuracy_gap_pp": ab_rank16_gap * 100.0,
        "accuracy_vs_best_single": ab_best_single_acc,
        "pair_accuracy": configs["A_plus_B"]["a_residual_b"]["accuracy"],
    }

    bc_ind = pairs["independent_b_c"]
    bc_res = pairs["residual_b_c"]
    bc_ind_marginals = marginals["independent_b_c"]
    bc_res_marginals = marginals["residual_b_c"]
    bc_best_single_acc = max(
        configs["B_plus_C"]["independent_b"]["accuracy"],
        configs["B_plus_C"]["expert_c"]["accuracy"],
    )
    condition3 = {
        "accuracy_gain_vs_best_single_pp": (configs["B_plus_C"]["independent_b_c"]["accuracy"] - bc_best_single_acc) * 100.0,
        "passed": (
            configs["B_plus_C"]["independent_b_c"]["accuracy"] > bc_best_single_acc
            and configs["B_plus_C"]["independent_b_c"]["accuracy"] - bc_best_single_acc >= 0.01
            and bc_ind["mean_synergy"] > 0
            and bc_ind["median_synergy"] > 0
            and bc_ind["synergy_95ci"][0] > 0
            and bc_ind_marginals["G_B_given_C"]["mean"] > 0
            and bc_ind_marginals["G_C_given_B"]["mean"] > 0
            and configs["B_plus_C"]["independent_b_c"]["accuracy"] > bc_best_single_acc
            and bc_ind["mean_synergy"] > 0
        ),
        "accuracy_vs_best_single": bc_best_single_acc,
        "pair_accuracy": configs["B_plus_C"]["independent_b_c"]["accuracy"],
    }
    bc_res_best = max(
        configs["B_plus_C"]["residual_b"]["accuracy"],
        configs["B_plus_C"]["expert_c"]["accuracy"],
    )
    condition4 = {
        "accuracy_gain_vs_best_single_pp": (configs["B_plus_C"]["residual_b_c"]["accuracy"] - bc_res_best) * 100.0,
        "passed": (
            configs["B_plus_C"]["residual_b_c"]["accuracy"] > bc_res_best
            and configs["B_plus_C"]["residual_b_c"]["accuracy"] >= configs["B_plus_C"]["independent_b_c"]["accuracy"]
            and bc_res["mean_synergy"] > 0
            and bc_res["median_synergy"] > 0
            and bc_res_marginals["G_B_given_C"]["mean"] > 0
            and bc_res_marginals["G_C_given_B"]["mean"] > 0
            and configs["B_plus_C"]["residual_b_c"]["accuracy"] > bc_res_best
            and bc_res["mean_synergy"] > 0
        ),
        "accuracy_vs_best_single": bc_res_best,
        "accuracy_vs_independent_pair": configs["B_plus_C"]["independent_b_c"]["accuracy"],
        "pair_accuracy": configs["B_plus_C"]["residual_b_c"]["accuracy"],
    }
    condition5 = {
        "worst10_mean_synergy": ab_pair["worst10_mean_synergy"],
        "worst10_mean_synergy_floor_ok": ab_pair["worst10_mean_synergy"] > -0.5
            and bc_res["worst10_mean_synergy"] > -0.5,
        "pair_correct_vs_best_single_wrong_ok": (
            ab_pair["pair_correct_best_single_wrong_rate"] >= ab_pair["best_single_correct_pair_wrong_rate"]
        ),
        "brier_not_worse": (
            configs["A_plus_B"]["a_residual_b"]["brier"] <= configs["A_plus_B"][best_single_ab]["brier"] + 0.01
            and configs["B_plus_C"]["residual_b_c"]["brier"] <= configs["B_plus_C"][best_single_bc]["brier"] + 0.01
        ),
        "ece_not_worse": (
            configs["A_plus_B"]["a_residual_b"]["ece_15_bins"] <= configs["A_plus_B"][best_single_ab]["ece_15_bins"] + 0.02
            and configs["B_plus_C"]["residual_b_c"]["ece_15_bins"] <= configs["B_plus_C"][best_single_bc]["ece_15_bins"] + 0.02
        ),
        "direction_consistent": direction_ab
            and configs["B_plus_C"]["residual_b_c"]["accuracy"] > bc_res_best
            and bc_res["mean_synergy"] > 0,
        "passed": None,  # filled at aggregate level (cross-seed repetition)
    }
    condition5["passed"] = (
        condition5["worst10_mean_synergy_floor_ok"]
        and condition5["pair_correct_vs_best_single_wrong_ok"]
        and condition5["brier_not_worse"]
        and condition5["ece_not_worse"]
        and condition5["direction_consistent"]
    )

    return {
        "seed": seed,
        "configs": configs,
        "pairs": pairs,
        "marginals": marginals,
        "bootstrap": bootstrap,
        "best_single_ab": best_single_ab,
        "best_single_bc": best_single_bc,
        "single_validity": single_validity,
        "conditions": {
            "single_function_validity": condition1,
            "seen_composition": condition2,
            "independent_unseen_composition": condition3,
            "residual_unseen_transfer": condition4,
            "tail_stability": condition5,
        },
    }


def _cross_seed_worst10_repetition(seed_results: List[dict]) -> float:
    """Fraction of worst-10 synergy samples shared by >=2 seeds, per pair."""
    rates = {}
    for dataset, pair_name, _singles, _rank16 in PAIRS:
        sets = [set(item["pairs"][pair_name]["worst10_sample_ids"]) for item in seed_results]
        shared = set().union(*sets)
        repeated = {sid for sid in shared if sum(sid in s for s in sets) >= 2}
        union = set().union(*sets)
        rates[pair_name] = len(repeated) / len(union) if union else 0.0
    return rates


def _cross_seed(seed_results: List[dict]) -> dict:
    aggregate = {}
    def agg(name, getter, fmt="value"):
        values = [getter(result) for result in seed_results]
        aggregate[name] = {
            "mean": _mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    for dataset, models in CONFIGS.items():
        for model in models:
            for metric in ("accuracy", "positive_accuracy", "negative_accuracy",
                           "balanced_accuracy", "mean_answer_token_nll", "brier",
                           "ece_15_bins", "free_generation_accuracy"):
                agg("{}.{}.{}".format(dataset, model, metric),
                    lambda r, d=dataset, m=model, k=metric: r["configs"][d][m][k])
            for hard in ("count_negative", "attribute_negative", "relation_negative", "positive"):
                agg("{}.{}.hard_negative.{}.accuracy".format(dataset, model, hard),
                    lambda r, d=dataset, m=model, h=hard: r["configs"][d][m]["hard_negative"][h]["accuracy"])
    for dataset, pair_name, _singles, _rank16 in PAIRS:
        for metric in ("mean_synergy", "median_synergy", "positive_synergy_rate",
                       "worst10_mean_synergy", "pair_exclusive_correct_rate",
                       "pair_correct_best_single_wrong_rate", "best_single_correct_pair_wrong_rate",
                       "error_set_jaccard", "pair_vs_rank16_mean_nll_diff"):
            agg("{}.{}".format(pair_name, metric),
                lambda r, p=pair_name, k=metric: r["pairs"][p][k])
    agg("A_plus_B.a_residual_b.rank16_gap_pp",
        lambda r: r["conditions"]["seen_composition"]["rank16_accuracy_gap_pp"])
    def _condition_passed(result, name):
        value = result["conditions"][name]
        return bool(value["passed"]) if isinstance(value, dict) else bool(value)

    condition_counts = {
        name: sum(_condition_passed(result, name) for result in seed_results)
        for name in seed_results[0]["conditions"]
    }
    return {
        "aggregate": aggregate,
        "condition_pass_counts_out_of_3": condition_counts,
        "cross_seed_worst10_repetition_rate": _cross_seed_worst10_repetition(seed_results),
    }


def _decide(condition_counts: dict, aggregate: dict, repetition_rates: dict) -> dict:
    """Section 十九 automatic decision."""
    c1 = condition_counts["single_function_validity"]
    c2 = condition_counts["seen_composition"]
    c3 = condition_counts["independent_unseen_composition"]
    c4 = condition_counts["residual_unseen_transfer"]
    c5 = condition_counts["tail_stability"]
    rep_ok = (
        repetition_rates.get("a_residual_b", 0.0) < 0.30
        and repetition_rates.get("residual_b_c", 0.0) < 0.30
    )
    evidence = {
        "single_function_validity_pass_seeds": c1,
        "seen_composition_pass_seeds": c2,
        "independent_unseen_pass_seeds": c3,
        "residual_unseen_pass_seeds": c4,
        "tail_stability_pass_seeds": c5,
        "worst10_cross_seed_repetition_ok": rep_ok,
        "worst10_repetition_rates": repetition_rates,
    }
    if c1 < 2:
        return {
            "decision": "INVALID_EXPERIMENT",
            "reason": "single-function validity failed in more than one seed: experts were not reliably learned",
            "evidence": evidence,
        }
    ab_gap = aggregate["A_plus_B.a_residual_b.rank16_gap_pp"]["mean"]
    seen_and_rank16_close = (
        c2 >= 2
        and ab_gap >= -1.0
    )
    if (
        c2 >= 2 and c3 >= 2 and c4 >= 2 and c5 >= 2
        and seen_and_rank16_close and rep_ok
    ):
        return {
            "decision": "FORMAT_CONTROLLED_COMPOSITION_PASSED",
            "reason": ("seen + independent unseen + residual unseen composition all passed in >=2/3 seeds "
                       "with tail stability and rank16 proximity; accuracy/NLL/Brier direction consistent"),
            "evidence": evidence,
        }
    if c3 >= 2 and c4 < 2:
        return {
            "decision": "DIRECT_COMPOSITION_POSSIBLE_RESIDUAL_FAILED",
            "reason": "independent B+C composition passed but residual B|A+C failed",
            "evidence": evidence,
        }
    if c2 >= 2 and c4 < 2:
        return {
            "decision": "RESIDUAL_CONTEXT_BOUND",
            "reason": "A+ResidualB passed on seen A+B but ResidualB+C failed on unseen B+C",
            "evidence": evidence,
        }
    if c2 >= 2 or c3 >= 2 or c4 >= 2:
        return {
            "decision": "LOCAL_COMPOSITION_ONLY",
            "reason": "pair gains are local: not all four composition conditions hold at >=2/3 seeds",
            "evidence": evidence,
        }
    return {
        "decision": "STOP_COMPOSITION_CONFIRMED",
        "reason": "no composition condition passed at >=2/3 seeds under the unified answer format",
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-dir", required=True)
    parser.add_argument("--final-output", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    bootstrap_dir = Path(args.bootstrap_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in seeds:
        result = _seed_metrics(root, seed)
        seed_results.append(result)
        (output_dir / "seed{}.json".format(seed)).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (bootstrap_dir / "seed{}_bootstrap.json".format(seed)).write_text(
            json.dumps(result["bootstrap"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    cross = _cross_seed(seed_results)
    decision = _decide(
        cross["condition_pass_counts_out_of_3"],
        cross["aggregate"],
        cross["cross_seed_worst10_repetition_rate"],
    )
    final = {
        "seeds": [{"seed": item["seed"],
                   "conditions": item["conditions"],
                   "single_validity": item["single_validity"],
                   "pairs": item["pairs"],
                   "marginals": item["marginals"]} for item in seed_results],
        **cross,
        "decision": decision,
    }
    final_path = Path(args.final_output)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision["decision"],
        "condition_pass_counts": cross["condition_pass_counts_out_of_3"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
