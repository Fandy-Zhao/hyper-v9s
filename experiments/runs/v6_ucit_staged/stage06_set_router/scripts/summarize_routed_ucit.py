#!/usr/bin/env python3
"""Summarize the complete routed UCIT matrix and answer-free routing diagnostics."""

import argparse
import json
import re
from pathlib import Path
from statistics import fmean


TASKS = (("ImageNet-R", "Accuracy"), ("ArxivQA", "Accuracy"), ("VizWiz", "Average"),
         ("IconQA", "Accuracy"), ("CLEVR", "Accuracy"), ("Flickr30k", "Average"))


def score(path, metric):
    match = re.search(rf"^{metric}\s*:\s*([0-9.]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise ValueError(f"cannot parse {metric} from {path}")
    return float(match.group(1))


def continual_metrics(matrix):
    return {
        "MAA": fmean(fmean(row[:index + 1]) for index, row in enumerate(matrix)),
        "MFN": fmean(matrix[-1]),
        "MFT": fmean(matrix[index][index] for index in range(6)),
        "BWT": fmean(matrix[-1][index] - matrix[index][index] for index in range(5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = [[None] * 6 for _ in range(6)]
    per_stage, total_counts = {}, [0, 0, 0]
    route_seconds, route_samples = 0.0, 0
    model_seconds, model_samples, reused, peak = 0.0, 0, 0, 0
    for stage in range(1, 7):
        stage_counts = [0, 0, 0]
        for seen in range(1, stage + 1):
            dataset, metric = TASKS[seen - 1]
            cell = args.eval_root / dataset / f"routed-task{stage}"
            matrix[stage - 1][seen - 1] = score(cell / "Result.text", metric)
            route = json.loads((args.route_root / f"task{stage}_{dataset}.json").read_text(encoding="utf-8"))
            if route.get("oracle_used") is not False or route.get("answer_features_used") is not False or route.get("task_id_lookup_used") is not False:
                raise ValueError("route leakage audit failed")
            counts = [sum(item["cardinality"] == size for item in route["routes"]) for size in range(3)]
            stage_counts = [left + right for left, right in zip(stage_counts, counts)]
            total_counts = [left + right for left, right in zip(total_counts, counts)]
            route_samples += len(route["routes"])
            route_seconds += float(route["route_seconds_per_sample"]) * len(route["routes"])
            for metrics_path in cell.glob("*_metrics.json"):
                value = json.loads(metrics_path.read_text(encoding="utf-8"))
                samples = int(value["samples"])
                model_samples += samples
                model_seconds += float(value["model_seconds_per_sample"]) * samples
                reused += int(value["reused_frozen_stage01_answers"])
                peak = max(peak, int(value["peak_memory_bytes"]))
        per_stage[str(stage)] = {
            "counts": {str(size): stage_counts[size] for size in range(3)},
            "rates": {str(size): stage_counts[size] / sum(stage_counts) for size in range(3)},
            "average_active_experts": (stage_counts[1] + 2 * stage_counts[2]) / sum(stage_counts),
        }
    metrics = continual_metrics(matrix)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["metrics"]
    output = {
        "status": "COMPLETE",
        "task_order": [item[0] for item in TASKS],
        "accuracy_matrix": matrix,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "delta_vs_hyper_baseline": {key: metrics[key] - float(baseline[key]) for key in metrics},
        "routing": {
            "counts": {str(size): total_counts[size] for size in range(3)},
            "rates": {str(size): total_counts[size] / sum(total_counts) for size in range(3)},
            "average_active_experts": (total_counts[1] + 2 * total_counts[2]) / sum(total_counts),
            "seconds_per_sample": route_seconds / route_samples,
            "per_stage": per_stage,
            "predicted_single_only_equals_predicted_set": total_counts[2] == 0,
        },
        "generation": {
            "seconds_per_sample_including_reuse": model_seconds / model_samples,
            "peak_memory_bytes": peak,
            "reused_frozen_stage01_answers": reused,
            "samples": model_samples,
        },
        "oracle_used": False,
        "answer_features_used": False,
        "task_id_lookup_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SUMMARIZED", "metrics": metrics, "routing": output["routing"]}, sort_keys=True))


if __name__ == "__main__":
    main()
