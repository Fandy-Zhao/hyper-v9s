import argparse
import json
import os
import statistics
from collections import defaultdict

from .cache import read_jsonl


def _by_id(path, value_field):
    result = {}
    for row in read_jsonl(path):
        sample_id = str(row["sample_id"] if "sample_id" in row else row["question_id"])
        if sample_id in result:
            raise ValueError("duplicate sample ID {} in {}".format(sample_id, path))
        result[sample_id] = row[value_field]
    return result


def _predictions(path):
    result = {}
    for row in read_jsonl(path):
        sample_id = str(row["question_id"])
        result[sample_id] = str(row["text"])
    return result


def _annotations(path):
    with open(path, encoding="utf-8") as handle:
        return {str(row["question_id"]): str(row["answer"]) for row in json.load(handle)}


def _correct(answer, prediction):
    return str(answer).strip().casefold() == str(prediction).strip().casefold()


def _checkpoint_parameters(path):
    with open(os.path.join(path, "compose_experts.json"), encoding="utf-8") as handle:
        return int(json.load(handle)["metrics"]["adapter_parameter_count"])


def _metrics(rows, rank16_nll, annotations, predictions):
    synergies = [float(row["synergy"]) for row in rows]
    pair_minus_rank16 = []
    best_single_correct = 0
    best_pair_correct = 0
    rank16_correct = 0
    direct_sum_correct = 0
    base_correct = 0
    single0_correct = 0
    single1_correct = 0
    active = []
    for row in rows:
        sample_id = str(row["sample_id"])
        pair_minus_rank16.append(float(row["best_pair_loss"]) - float(rank16_nll[sample_id]))
        best_single_ids = row["candidate_expert_ids"][int(row["best_single_index"])]
        if len(best_single_ids) != 1:
            raise AssertionError("best single index is not a singleton")
        single_name = "single{}".format(best_single_ids[0])
        answer = annotations[sample_id]
        base_correct += _correct(answer, predictions["base"][sample_id])
        single0_correct += _correct(answer, predictions["single0"][sample_id])
        single1_correct += _correct(answer, predictions["single1"][sample_id])
        best_single_correct += _correct(answer, predictions[single_name][sample_id])
        best_pair_correct += _correct(answer, predictions["pair_l2"][sample_id])
        rank16_correct += _correct(answer, predictions["rank16"][sample_id])
        if "pair_direct_sum" in predictions:
            direct_sum_correct += _correct(answer, predictions["pair_direct_sum"][sample_id])
        active.append(len(row["candidate_expert_ids"][int(row["best_overall_index"])]))
    count = len(rows)
    result = {
        "samples": count,
        "pair_oracle_rate": sum(bool(row["pair_oracle"]) for row in rows) / count,
        "mean_synergy": statistics.fmean(synergies),
        "median_synergy": statistics.median(synergies),
        "positive_synergy_rate": sum(value > 0 for value in synergies) / count,
        "pair_better_than_rank16_rate": sum(value < 0 for value in pair_minus_rank16) / count,
        "mean_pair_minus_rank16": statistics.fmean(pair_minus_rank16),
        "best_single_accuracy": best_single_correct / count,
        "best_pair_accuracy": best_pair_correct / count,
        "rank16_accuracy": rank16_correct / count,
        "base_accuracy": base_correct / count,
        "single0_accuracy": single0_correct / count,
        "single1_accuracy": single1_correct / count,
        "average_active_experts": statistics.fmean(active),
        "synergy_threshold_pair_acceptance": {
            str(threshold): sum(
                float(row["synergy"]) / max(abs(float(row["best_single_loss"])), 1e-12)
                > threshold
                for row in rows
            )
            / count
            for threshold in (0.0, 0.01, 0.02, 0.03, 0.05)
        },
    }
    if "pair_direct_sum" in predictions:
        result["pair_direct_sum_accuracy"] = direct_sum_correct / count
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-cache", required=True)
    parser.add_argument("--rank16-scores", required=True)
    parser.add_argument("--direct-sum-scores")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--single0-predictions", required=True)
    parser.add_argument("--single1-predictions", required=True)
    parser.add_argument("--pair-predictions", required=True)
    parser.add_argument("--rank16-predictions", required=True)
    parser.add_argument("--direct-sum-predictions")
    parser.add_argument("--pool-checkpoint", required=True)
    parser.add_argument("--rank16-checkpoint", required=True)
    parser.add_argument("--run-summary", action="append", default=[])
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.oracle_cache))
    rank16_nll = _by_id(args.rank16_scores, "nll")
    annotations = _annotations(args.annotation_file)
    predictions = {
        "base": _predictions(args.base_predictions),
        "single0": _predictions(args.single0_predictions),
        "single1": _predictions(args.single1_predictions),
        "pair_l2": _predictions(args.pair_predictions),
        "rank16": _predictions(args.rank16_predictions),
    }
    if args.direct_sum_predictions:
        predictions["pair_direct_sum"] = _predictions(args.direct_sum_predictions)
    expected_ids = {str(row["sample_id"]) for row in rows}
    sources = {"rank16_nll": set(rank16_nll), "annotations": set(annotations)}
    sources.update({name: set(values) for name, values in predictions.items()})
    for name, sample_ids in sources.items():
        if sample_ids != expected_ids:
            raise ValueError(
                "{} sample IDs mismatch; missing={}, unexpected={}".format(
                    name, sorted(expected_ids - sample_ids)[:8], sorted(sample_ids - expected_ids)[:8]
                )
            )
    result = {
        "overall": _metrics(rows, rank16_nll, annotations, predictions),
        "parameters": {
            "rank8_pair": _checkpoint_parameters(args.pool_checkpoint),
            "rank16_single": _checkpoint_parameters(args.rank16_checkpoint),
        },
    }
    inference = {}
    for specification in args.run_summary:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path:
            raise ValueError("run summaries must use NAME=PATH")
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        samples = int(summary["samples"])
        duration = float(summary["duration_seconds"])
        inference[name] = {
            "samples": samples,
            "duration_seconds": duration,
            "latency_seconds_per_sample": duration / samples,
            "peak_memory_bytes": int(summary["peak_memory_bytes"]),
        }
    result["inference"] = inference
    by_task = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    result["by_task"] = {
        task_id: _metrics(task_rows, rank16_nll, annotations, predictions)
        for task_id, task_rows in sorted(by_task.items())
    }
    if args.direct_sum_scores:
        direct = _by_id(args.direct_sum_scores, "nll")
        if set(direct) != expected_ids:
            raise ValueError("direct-sum score sample IDs mismatch")
        result["overall"]["mean_direct_sum_minus_l2"] = statistics.fmean(
            float(direct[str(row["sample_id"])]) - float(row["best_pair_loss"])
            for row in rows
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
