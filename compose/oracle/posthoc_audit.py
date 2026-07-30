import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

from .cache import read_jsonl


CONFIG_NAMES = ("base", "single0", "single1", "pair_l2", "pair_direct_sum", "rank16")
ORACLE_A_NAMES = CONFIG_NAMES[:5]
ORACLE_B_NAMES = ("rank16", "pair_l2", "pair_direct_sum")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_rows(path: str, id_field: str) -> Dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        sample_id = str(row[id_field])
        if sample_id in rows:
            raise ValueError("duplicate sample ID {} in {}".format(sample_id, path))
        rows[sample_id] = row
    return rows


def _annotations(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise ValueError("annotation file must contain a JSON list")
    rows = {}
    for row in values:
        sample_id = str(row["question_id"])
        if sample_id in rows:
            raise ValueError("duplicate annotation sample ID {}".format(sample_id))
        rows[sample_id] = row
    return rows


def _correct(answer: object, prediction: object) -> bool:
    return str(answer).strip().casefold() == str(prediction).strip().casefold()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict:
    positive = [value for value in values if value > 0.0]
    negative = [value for value in values if value < 0.0]
    negative_magnitudes = sorted((-value for value in negative), reverse=True)
    total_negative_magnitude = sum(negative_magnitudes)

    def tail_share(fraction: float) -> float:
        if not negative_magnitudes or total_negative_magnitude == 0.0:
            return 0.0
        count = max(1, int(math.ceil(len(negative_magnitudes) * fraction)))
        return sum(negative_magnitudes[:count]) / total_negative_magnitude

    widespread_small = (
        len(negative) / len(values) >= 0.5 and tail_share(0.10) < 0.5
    )
    result = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "positive_rate": len(positive) / len(values),
        "negative_rate": len(negative) / len(values),
        "zero_rate": (len(values) - len(positive) - len(negative)) / len(values),
        "positive_subset_mean": statistics.fmean(positive) if positive else None,
        "negative_subset_mean": statistics.fmean(negative) if negative else None,
        "maximum_positive_gain": max(values),
        "maximum_negative_gain": min(values),
        "negative_magnitude_share_worst_1pct": tail_share(0.01),
        "negative_magnitude_share_worst_5pct": tail_share(0.05),
        "negative_magnitude_share_worst_10pct": tail_share(0.10),
        "negative_mean_characterization": (
            "widespread_small_negative_gains" if widespread_small
            else "tail_concentrated_or_not_widespread"
        ),
    }
    for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        result["p{}".format(percentile)] = _percentile(values, percentile)
    return result


def _mean_or_none(values: Iterable[float]):
    values = list(values)
    return statistics.fmean(values) if values else None


def _subset_metrics(rows: Sequence[dict]) -> dict:
    return {
        "samples": len(rows),
        "mean_synergy": _mean_or_none(row["synergy"] for row in rows),
        "mean_pair_minus_rank16_nll": _mean_or_none(
            row["pair_best_nll"] - row["rank16_nll"] for row in rows
        ),
    }


def _selection_summary(rows: Sequence[dict], prefix: str, names: Sequence[str]) -> dict:
    counts = Counter(row[prefix + "_selection"] for row in rows)
    return {name: counts[name] / len(rows) for name in names}


def _aggregate(rows: Sequence[dict]) -> dict:
    count = len(rows)
    oracle_a_correct = sum(row["oracle_a_correct"] for row in rows)
    oracle_b_correct = sum(row["oracle_b_correct"] for row in rows)
    best_single_correct = sum(row["best_single_correct"] for row in rows)
    rank16_correct = sum(row["rank16_correct"] for row in rows)
    pair_correct = sum(row["pair_best_correct"] for row in rows)
    rank16_errors = {row["sample_id"] for row in rows if not row["rank16_correct"]}
    pair_errors = {row["sample_id"] for row in rows if not row["pair_best_correct"]}
    union = rank16_errors | pair_errors
    exclusive_correct = sum(
        row["pair_best_correct"]
        and not row["rank16_correct"]
        and not row["best_single_correct"]
        for row in rows
    )
    rank16_correct_pair_wrong = sum(
        row["rank16_correct"] and not row["pair_best_correct"] for row in rows
    )
    rank16_wrong_pair_correct = sum(
        not row["rank16_correct"] and row["pair_best_correct"] for row in rows
    )
    best_single_mean_nll = statistics.fmean(row["best_single_nll"] for row in rows)
    rank16_mean_nll = statistics.fmean(row["rank16_nll"] for row in rows)
    oracle_a_mean_nll = statistics.fmean(row["oracle_a_nll"] for row in rows)
    oracle_b_mean_nll = statistics.fmean(row["oracle_b_nll"] for row in rows)
    result = {
        "samples": count,
        "best_single": {
            "mean_nll": best_single_mean_nll,
            "generation_accuracy": best_single_correct / count,
        },
        "rank16": {
            "mean_nll": rank16_mean_nll,
            "generation_accuracy": rank16_correct / count,
        },
        "best_pair": {
            "mean_nll": statistics.fmean(row["pair_best_nll"] for row in rows),
            "generation_accuracy": pair_correct / count,
        },
        "oracle_a": {
            "mean_nll": oracle_a_mean_nll,
            "generation_accuracy": oracle_a_correct / count,
            "accuracy_gain_over_best_single_percentage_points":
                100.0 * (oracle_a_correct - best_single_correct) / count,
            "mean_nll_improvement_over_best_single": best_single_mean_nll - oracle_a_mean_nll,
            "selection_rate": _selection_summary(rows, "oracle_a", ORACLE_A_NAMES),
        },
        "oracle_b": {
            "mean_nll": oracle_b_mean_nll,
            "generation_accuracy": oracle_b_correct / count,
            "accuracy_gain_over_rank16_percentage_points":
                100.0 * (oracle_b_correct - rank16_correct) / count,
            "mean_nll_improvement_over_rank16": rank16_mean_nll - oracle_b_mean_nll,
            "selection_rate": _selection_summary(rows, "oracle_b", ORACLE_B_NAMES),
        },
        "pair_exclusive_nll_rate": sum(row["pair_exclusive_nll"] for row in rows) / count,
        "pair_exclusive_correct_rate": exclusive_correct / count,
        "pair_exclusive_correct_count": exclusive_correct,
        "rank16_correct_pair_wrong": {
            "count": rank16_correct_pair_wrong,
            "rate": rank16_correct_pair_wrong / count,
        },
        "rank16_wrong_pair_correct": {
            "count": rank16_wrong_pair_correct,
            "rate": rank16_wrong_pair_correct / count,
        },
        "pair_rank16_error_jaccard": len(rank16_errors & pair_errors) / len(union) if union else 1.0,
        "subsets": {
            "pair_positive_synergy": _subset_metrics([row for row in rows if row["synergy"] > 0]),
            "pair_negative_synergy": _subset_metrics([row for row in rows if row["synergy"] < 0]),
            "pair_better_than_rank16": _subset_metrics([row for row in rows if row["pair_best_nll"] < row["rank16_nll"]]),
            "pair_not_better_than_rank16": _subset_metrics([row for row in rows if row["pair_best_nll"] >= row["rank16_nll"]]),
        },
    }
    return result


def audit(
    oracle_cache: str,
    rank16_scores: str,
    direct_sum_scores: str,
    annotation_file: str,
    prediction_paths: Mapping[str, str],
) -> tuple:
    oracle = _unique_rows(oracle_cache, "sample_id")
    rank16 = _unique_rows(rank16_scores, "sample_id")
    direct = _unique_rows(direct_sum_scores, "sample_id")
    annotations = _annotations(annotation_file)
    predictions = {
        name: _unique_rows(path, "question_id") for name, path in prediction_paths.items()
    }
    expected_ids = set(oracle)
    sources = {"rank16": set(rank16), "direct_sum": set(direct), "annotations": set(annotations)}
    sources.update({"prediction_{}".format(name): set(values) for name, values in predictions.items()})
    for name, sample_ids in sources.items():
        if sample_ids != expected_ids:
            raise ValueError(
                "{} sample IDs mismatch; missing={}, unexpected={}".format(
                    name, sorted(expected_ids - sample_ids)[:8], sorted(sample_ids - expected_ids)[:8]
                )
            )

    rows = []
    for sample_id, source in oracle.items():
        if source["candidate_expert_ids"] != [[], [0], [1], [0, 1]]:
            raise ValueError("unexpected candidate ordering for {}".format(sample_id))
        task_id = str(source["task_id"])
        if str(rank16[sample_id]["task_id"]) != task_id or str(direct[sample_id]["task_id"]) != task_id:
            raise ValueError("task ID mismatch for {}".format(sample_id))
        nll = {
            "base": float(source["set_nll"][0]),
            "single0": float(source["set_nll"][1]),
            "single1": float(source["set_nll"][2]),
            "pair_l2": float(source["set_nll"][3]),
            "pair_direct_sum": float(direct[sample_id]["nll"]),
            "rank16": float(rank16[sample_id]["nll"]),
        }
        answer = str(annotations[sample_id]["answer"])
        prediction = {name: str(predictions[name][sample_id]["text"]) for name in CONFIG_NAMES}
        correct = {name: _correct(answer, value) for name, value in prediction.items()}
        best_single_name = min(("single0", "single1"), key=lambda name: (nll[name], name))
        pair_best_name = min(("pair_l2", "pair_direct_sum"), key=lambda name: (nll[name], name))
        oracle_a_name = min(ORACLE_A_NAMES, key=lambda name: (nll[name], name))
        oracle_b_name = min(ORACLE_B_NAMES, key=lambda name: (nll[name], name))
        row = {
            "sample_id": sample_id,
            "task_id": task_id,
            "ground_truth": answer,
            **{"{}_nll".format(name): nll[name] for name in CONFIG_NAMES},
            **{"{}_prediction".format(name): prediction[name] for name in CONFIG_NAMES},
            **{"{}_correct".format(name): correct[name] for name in CONFIG_NAMES},
            "best_single_selection": best_single_name,
            "best_single_nll": nll[best_single_name],
            "best_single_correct": correct[best_single_name],
            "pair_best_selection": pair_best_name,
            "pair_best_nll": nll[pair_best_name],
            "pair_best_correct": correct[pair_best_name],
            "synergy": nll[best_single_name] - nll[pair_best_name],
            "pair_minus_rank16_nll": nll[pair_best_name] - nll["rank16"],
            "pair_exclusive_nll": nll[pair_best_name] < nll[best_single_name] and nll[pair_best_name] < nll["rank16"],
            "oracle_a_selection": oracle_a_name,
            "oracle_a_nll": nll[oracle_a_name],
            "oracle_a_correct": correct[oracle_a_name],
            "oracle_b_selection": oracle_b_name,
            "oracle_b_nll": nll[oracle_b_name],
            "oracle_b_correct": correct[oracle_b_name],
        }
        rows.append(row)

    overall = _aggregate(rows)
    by_task_rows = defaultdict(list)
    for row in rows:
        by_task_rows[row["task_id"]].append(row)
    distributions = {"overall": _distribution([row["synergy"] for row in rows])}
    distributions.update({
        task_id: _distribution([row["synergy"] for row in task_rows])
        for task_id, task_rows in sorted(by_task_rows.items())
    })
    stop_values = {
        "oracle_b_accuracy_gain_percentage_points": overall["oracle_b"]["accuracy_gain_over_rank16_percentage_points"],
        "oracle_b_mean_nll_improvement": overall["oracle_b"]["mean_nll_improvement_over_rank16"],
        "pair_exclusive_correct_rate": overall["pair_exclusive_correct_rate"],
    }
    stop_triggered = (
        stop_values["oracle_b_accuracy_gain_percentage_points"] < 0.5
        and stop_values["oracle_b_mean_nll_improvement"] < 0.005
        and stop_values["pair_exclusive_correct_rate"] < 0.02
    )
    inputs = [oracle_cache, rank16_scores, direct_sum_scores, annotation_file]
    inputs.extend(prediction_paths[name] for name in CONFIG_NAMES)
    result = {
        "schema_version": 1,
        "definitions": {
            "best_single": "per-sample minimum-NLL selection over single0 and single1",
            "best_pair": "per-sample minimum-NLL selection over pair_l2 and pair_direct_sum",
            "oracle_generation": "generation prediction belonging to the per-sample minimum-NLL selection",
            "synergy": "best_single_nll - best_pair_nll",
            "standard_deviation": "population standard deviation",
            "percentiles": "linear interpolation at (n-1)*p/100",
        },
        "input_sha256": {path: _sha256(path) for path in inputs},
        "overall": overall,
        "by_task": {task_id: _aggregate(task_rows) for task_id, task_rows in sorted(by_task_rows.items())},
        "synergy_distribution": distributions,
        "stage_f0_stop_condition": {
            **stop_values,
            "thresholds": {
                "oracle_b_accuracy_gain_percentage_points_lt": 0.5,
                "oracle_b_mean_nll_improvement_lt": 0.005,
                "pair_exclusive_correct_rate_lt": 0.02,
            },
            "triggered": stop_triggered,
            "decision": (
                "当前 pair 对 rank16 只有极弱局部互补，不值得实现 rank16/pair Router。"
                if stop_triggered else
                "当前 pair 对 rank16 存在超过预注册停止门槛的局部互补；仅保留为后续实验基线，不实现 Router。"
            ),
        },
    }
    return result, rows


def _write_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return "{:.4f}%".format(100.0 * value)


def _write_report(path: str, result: dict) -> None:
    overall = result["overall"]
    distribution = result["synergy_distribution"]["overall"]
    stop = result["stage_f0_stop_condition"]
    lines = [
        "# Stage F0 Post-hoc Oracle Audit",
        "",
        "## Scope and definitions",
        "",
        "This is an offline audit of the existing 6,000-sample caches. No model was trained or evaluated again. Best single and best pair are selected per sample by teacher-forced NLL; Oracle generation accuracy uses the prediction attached to that NLL-selected configuration.",
        "",
        "## Oracle A",
        "",
        "- Mean NLL: `{:.8f}`".format(overall["oracle_a"]["mean_nll"]),
        "- Generation accuracy: `{}`".format(_pct(overall["oracle_a"]["generation_accuracy"])),
        "- Accuracy gain over best single: `{:+.4f}` percentage points".format(overall["oracle_a"]["accuracy_gain_over_best_single_percentage_points"]),
        "- Mean NLL improvement over best single: `{:+.8f}`".format(overall["oracle_a"]["mean_nll_improvement_over_best_single"]),
        "- Selection rates: `{}`".format(json.dumps(overall["oracle_a"]["selection_rate"], sort_keys=True)),
        "",
        "## Oracle B",
        "",
        "- Mean NLL: `{:.8f}`".format(overall["oracle_b"]["mean_nll"]),
        "- Generation accuracy: `{}`".format(_pct(overall["oracle_b"]["generation_accuracy"])),
        "- Accuracy gain over rank16: `{:+.4f}` percentage points".format(overall["oracle_b"]["accuracy_gain_over_rank16_percentage_points"]),
        "- Mean NLL improvement over rank16: `{:+.8f}`".format(overall["oracle_b"]["mean_nll_improvement_over_rank16"]),
        "- Selection rates: `{}`".format(json.dumps(overall["oracle_b"]["selection_rate"], sort_keys=True)),
        "",
        "## Pair-exclusive value",
        "",
        "- PairExclusiveNLLRate: `{}`".format(_pct(overall["pair_exclusive_nll_rate"])),
        "- PairExclusiveCorrectRate: `{}` (`{}` samples)".format(_pct(overall["pair_exclusive_correct_rate"]), overall["pair_exclusive_correct_count"]),
        "- Rank16CorrectPairWrong: `{}` (`{}` samples)".format(_pct(overall["rank16_correct_pair_wrong"]["rate"]), overall["rank16_correct_pair_wrong"]["count"]),
        "- Rank16WrongPairCorrect: `{}` (`{}` samples)".format(_pct(overall["rank16_wrong_pair_correct"]["rate"]), overall["rank16_wrong_pair_correct"]["count"]),
        "- Pair/rank16 error-set Jaccard: `{:.8f}`".format(overall["pair_rank16_error_jaccard"]),
        "",
        "## Synergy distribution",
        "",
        "- Mean / median / std: `{:.8f}` / `{:.8f}` / `{:.8f}`".format(distribution["mean"], distribution["median"], distribution["std"]),
        "- P1/P5/P10/P25/P50/P75/P90/P95/P99: `{}`".format(", ".join("{:.8f}".format(distribution["p{}".format(p)]) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99))),
        "- Positive / negative subset mean: `{}` / `{}`".format(distribution["positive_subset_mean"], distribution["negative_subset_mean"]),
        "- Maximum positive / negative gain: `{:.8f}` / `{:.8f}`".format(distribution["maximum_positive_gain"], distribution["maximum_negative_gain"]),
        "- Negative mean characterization: `{}`; worst 10% of negative samples account for `{}` of negative magnitude.".format(distribution["negative_mean_characterization"], _pct(distribution["negative_magnitude_share_worst_10pct"])),
        "",
        "| Scope | Mean | Median | Std | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 | Positive mean | Negative mean | Max positive | Max negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, values in result["synergy_distribution"].items():
        lines.append(
            "| {} | {} |".format(
                scope,
                " | ".join("{:.8f}".format(values[field]) for field in (
                    "mean", "median", "std", "p1", "p5", "p10", "p25", "p50",
                    "p75", "p90", "p95", "p99", "positive_subset_mean",
                    "negative_subset_mean", "maximum_positive_gain", "maximum_negative_gain",
                )),
            )
        )
    lines.extend([
        "",
        "## Requested subset comparisons",
        "",
        "| Scope | Subset | Samples | Mean synergy | Mean pair minus rank16 NLL |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for scope in ("overall", "UCIT/ImageNet-R", "UCIT/IconQA"):
        metrics = result["overall"] if scope == "overall" else result["by_task"][scope]
        for subset, values in metrics["subsets"].items():
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    scope, subset, values["samples"], values["mean_synergy"],
                    values["mean_pair_minus_rank16_nll"],
                )
            )
    lines.extend([
        "",
        "All source hashes are preserved in `posthoc_oracle.json`; every sample, NLL, prediction, answer, and derived selection is preserved in `per_sample_audit.csv`.",
        "",
        "## Pre-registered decision",
        "",
        "- Stop condition triggered: `{}`".format(str(stop["triggered"]).lower()),
        "- Oracle B accuracy gain: `{:+.4f}` pp (threshold `< 0.5`)".format(stop["oracle_b_accuracy_gain_percentage_points"]),
        "- Oracle B mean NLL improvement: `{:+.8f}` (threshold `< 0.005`)".format(stop["oracle_b_mean_nll_improvement"]),
        "- PairExclusiveCorrectRate: `{}` (threshold `< 2%`)".format(_pct(stop["pair_exclusive_correct_rate"])),
        "- Decision: {}".format(stop["decision"]),
        "",
        "Regardless of this gate, no rank16/pair Router or Set Router is implemented.",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-cache", required=True)
    parser.add_argument("--rank16-scores", required=True)
    parser.add_argument("--direct-sum-scores", required=True)
    parser.add_argument("--annotation-file", required=True)
    for name in CONFIG_NAMES:
        parser.add_argument("--{}-predictions".format(name.replace("_", "-")), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-file", required=True)
    args = parser.parse_args()
    predictions = {
        name: getattr(args, "{}_predictions".format(name)) for name in CONFIG_NAMES
    }
    result, rows = audit(
        args.oracle_cache,
        args.rank16_scores,
        args.direct_sum_scores,
        args.annotation_file,
        predictions,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    _write_csv(args.output_csv, rows)
    _write_report(args.report_file, result)
    print(json.dumps(result["stage_f0_stop_condition"], sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
