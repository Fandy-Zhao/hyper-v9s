"""Oracle-set quality, synergy and collapse diagnostics."""

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def _rate(values):
    values = list(values)
    return statistics.fmean(float(value) for value in values) if values else 0.0


def _bootstrap_ci(values, seed=42, iterations=1000):
    values = list(map(float, values))
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(iterations))
    return [means[int(.025 * iterations)], means[min(iterations - 1, int(.975 * iterations))]]


def summarize_oracles(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Oracle metrics require records")
    modes = {row["composition_mode"] for row in rows}
    scopes = {row["temporal_scope"] for row in rows}
    configs = {row["config_hash"] for row in rows}
    if len(modes) != 1 or len(scopes) != 1 or len(configs) != 1:
        raise ValueError("Oracle metrics cannot mix mode, temporal scope, or config")
    selected_sizes = [int(row["selected_cardinality"]) for row in rows]
    pair_rows = [pair for row in rows for pair in row["pairs"]]
    synergies = [float(pair["raw_gain_over_best_single"]) for pair in pair_rows]
    selected_pairs = [row for row in rows if int(row["selected_cardinality"]) == 2]
    best_single_ids = Counter(tuple(row["best_single_ids"] or []) for row in rows)
    selected_experts = Counter(value for row in rows for value in row["selected_expert_ids"])
    cooccurrence = Counter(tuple(row["selected_expert_ids"]) for row in selected_pairs)
    selected_candidates = []
    for row in rows:
        candidates = [row["empty"], *row["singles"], *row["pairs"]]
        selected_ids = tuple(row["selected_expert_ids"])
        matches = [candidate for candidate in candidates if tuple(candidate["expert_ids"]) == selected_ids]
        if len(matches) != 1:
            raise ValueError("selected Oracle set is missing or duplicated in candidates")
        selected_candidates.append(matches[0])
    transitions = Counter()
    for row in rows:
        if not row.get("best_pair") or not row.get("best_single"):
            continue
        single = bool(row["best_single"].get("exact_teacher_forced"))
        pair = bool(row["best_pair"].get("exact_teacher_forced"))
        transitions[(single, pair)] += 1
    single_errors = {row["sample_id"] for row in rows if row.get("best_single") and not row["best_single"].get("exact_teacher_forced")}
    pair_errors = {row["sample_id"] for row in rows if row.get("best_pair") and not row["best_pair"].get("exact_teacher_forced")}
    union = single_errors | pair_errors
    collapse_threshold = 0.95
    top_selected = selected_experts.most_common(1)[0][1] / len(rows) if selected_experts else 0.0
    task_sets = Counter(tuple(row["selected_expert_ids"]) for row in rows)
    selected_accuracy = _rate(bool(value.get("exact_teacher_forced")) for value in selected_candidates)
    best_single_values = [bool(row["best_single"].get("exact_teacher_forced")) for row in rows if row.get("best_single")]
    best_pair_values = [bool(row["best_pair"].get("exact_teacher_forced")) for row in rows if row.get("best_pair")]
    best_single_accuracy = _rate(best_single_values) if best_single_values else None
    best_pair_accuracy = _rate(best_pair_values) if best_pair_values else None
    pair_selected_accuracy_delta = _rate(
        bool(selected_candidates[index].get("exact_teacher_forced"))
        - bool(row["best_single"].get("exact_teacher_forced"))
        for index, row in enumerate(rows)
        if row["selected_cardinality"] == 2 and row.get("best_single")
    ) if selected_pairs else None
    single_by_expert = defaultdict(list)
    best_single_counts = Counter()
    for row in rows:
        if row.get("best_single_ids"):
            best_single_counts[int(row["best_single_ids"][0])] += 1
        for single in row["singles"]:
            single_by_expert[int(single["expert_ids"][0])].append((single, row["empty"]))
    single_metrics = {}
    for expert_id, values in sorted(single_by_expert.items()):
        single_metrics[str(expert_id)] = {
            "support": len(values),
            "best_single_count": best_single_counts[expert_id],
            "exact_accuracy": _rate(bool(single.get("exact_teacher_forced")) for single, _ in values),
            "mean_nll_improvement_over_empty": statistics.fmean(
                float(empty["mean_nll"]) - float(single["mean_nll"]) for single, empty in values
            ),
        }
    row_synergies = [
        (float(row["best_pair"]["raw_gain_over_best_single"]), str(row["sample_id"]))
        for row in rows if row.get("best_pair")
    ]
    worst_count = max(1, math.ceil(.1 * len(row_synergies))) if row_synergies else 0
    result = {
        "samples": len(rows), "composition_mode": next(iter(modes)), "temporal_scope": next(iter(scopes)),
        "config_hash": next(iter(configs)),
        "EmptyOracleRate": _rate(value == 0 for value in selected_sizes),
        "SingleOracleRate": _rate(value == 1 for value in selected_sizes),
        "PairOracleRate": _rate(value == 2 for value in selected_sizes),
        "PairEvaluatedRate": _rate(bool(row["pairs"]) for row in rows),
        "PairBetterThanBestSingleRate": _rate(any(float(pair["raw_gain_over_best_single"]) > 0 for pair in row["pairs"]) for row in rows),
        "PairPassedThresholdRate": _rate(any(bool(pair["valid_pair"]) for pair in row["pairs"]) for row in rows),
        "PairSelectedRate": _rate(value == 2 for value in selected_sizes),
        "mean_pair_synergy": statistics.fmean(synergies) if synergies else 0.0,
        "median_pair_synergy": statistics.median(synergies) if synergies else 0.0,
        "positive_pair_synergy_rate": _rate(value > 0 for value in synergies),
        "harmful_pair_rate": _rate(value < 0 for value in synergies),
        "worst_10_percent_pair_synergy": statistics.fmean(sorted(synergies)[:max(1, math.ceil(.1 * len(synergies)))]) if synergies else 0.0,
        "pair_gain_ci95": _bootstrap_ci(synergies),
        "average_selected_cardinality": statistics.fmean(selected_sizes),
        "average_evaluated_singles": statistics.fmean(len(row["singles"]) for row in rows),
        "average_evaluated_pairs": statistics.fmean(len(row["pairs"]) for row in rows),
        "selected_set_mean_nll": statistics.fmean(float(row["selected_mean_nll"]) for row in rows),
        "selected_set_exact_accuracy": selected_accuracy,
        "best_single_exact_accuracy": best_single_accuracy,
        "best_pair_exact_accuracy": best_pair_accuracy,
        "selected_vs_best_single_accuracy_delta": selected_accuracy - best_single_accuracy if best_single_accuracy is not None else None,
        "pair_selected_accuracy_delta_vs_best_single": pair_selected_accuracy_delta,
        "current_hyper_route_exact_accuracy": None,
        "selected_vs_current_hyper_route_accuracy_delta": None,
        "empty_mean_nll": statistics.fmean(float(row["empty"]["mean_nll"]) for row in rows),
        "best_single_mean_nll": statistics.fmean(float(row["best_single_nll"]) for row in rows if row["best_single_nll"] is not None) if any(row["best_single_nll"] is not None for row in rows) else None,
        "best_pair_mean_nll": statistics.fmean(float(row["best_pair_nll"]) for row in rows if row["best_pair_nll"] is not None) if any(row["best_pair_nll"] is not None for row in rows) else None,
        "best_single_distribution": {str(list(key)): value for key, value in sorted(best_single_ids.items())},
        "selected_expert_frequency": {str(key): value / len(rows) for key, value in sorted(selected_experts.items())},
        "single_expert_metrics": single_metrics,
        "pair_expert_cooccurrence": {str(list(key)): value for key, value in sorted(cooccurrence.items())},
        "accuracy_transitions": {"single_correct_pair_correct": transitions[(True, True)], "single_wrong_pair_correct": transitions[(False, True)], "single_correct_pair_wrong": transitions[(True, False)], "both_wrong": transitions[(False, False)]},
        "PairExclusiveCorrectRate": transitions[(False, True)] / len(rows),
        "BestSingleCorrectPairWrongRate": transitions[(True, False)] / len(rows),
        "error_set_jaccard": len(single_errors & pair_errors) / len(union) if union else 1.0,
        "worst_tail_sample_ids": [sample_id for _, sample_id in sorted(row_synergies)[:worst_count]],
    }
    result["collapse_checks"] = {
        "all_empty": result["EmptyOracleRate"] >= collapse_threshold,
        "all_single": result["SingleOracleRate"] >= collapse_threshold,
        "all_pair": result["PairOracleRate"] >= collapse_threshold,
        "one_expert": top_selected >= collapse_threshold,
        "task_id_fixed_set": task_sets.most_common(1)[0][1] / len(rows) >= collapse_threshold,
        "duplicate_pair": any(len({tuple(pair["expert_ids"]) for pair in row["pairs"]}) != len(row["pairs"]) for row in rows),
    }
    return result
