#!/usr/bin/env python3
"""作用：提供数据集或评测结果格式转换脚本，服务训练、评测或提交流程。"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

TASKS = [
    {"task_id": 1, "dataset": "ImageNet-R", "metric": "Accuracy"},
    {"task_id": 2, "dataset": "ArxivQA", "metric": "Accuracy"},
    {"task_id": 3, "dataset": "VizWiz", "metric": "Average"},
    {"task_id": 4, "dataset": "IconQA", "metric": "Accuracy"},
    {"task_id": 5, "dataset": "CLEVR-Math", "metric": "Accuracy"},
    {"task_id": 6, "dataset": "Flickr30k", "metric": "Average"},
]

METRIC_DEFINITIONS = {
    "MAA": "Mean Average Accuracy: mean over stages i of the average performance on tasks 1..i after training Task i.",
    "MFN": "Mean Final performance across Tasks: mean of the final checkpoint row R[N][j] for tasks j=1..N.",
    "MFT": "Mean performance on Newly learned tasks: mean diagonal performance R[i][i], measured immediately after each task is learned.",
    "BWT": "Backward Transfer: mean over old tasks j=1..N-1 of R[N][j] - R[j][j]. Negative values indicate forgetting.",
}


def parse_args():
    """作用：执行 parse_args 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    parser = argparse.ArgumentParser(
        description="Build continual-learning metrics from UCIT Result.text files."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args()


def read_score(path, metric_name):
    """作用：执行 read_score 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    pattern = re.compile(
        rf"^\s*{re.escape(metric_name)}\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*$",
        re.IGNORECASE,
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return round(float(match.group(1)), 6)
    raise ValueError(f"Metric {metric_name!r} not found in {path}")


def rounded_mean(values):
    """作用：执行 rounded_mean 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    values = list(values)
    return round(fmean(values), 6) if values else None


def main():
    """作用：作为脚本入口，串联参数解析、数据准备和核心处理流程。"""
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_file = (
        args.output_file or result_root / "continual_metrics.json"
    ).expanduser().resolve()
    tasks = TASKS[: args.num_tasks]
    matrix = []
    matrix_by_stage = {}
    stage_averages = []

    for model_task in range(1, args.num_tasks + 1):
        row = [None] * args.num_tasks
        named_row = {}
        for eval_task in range(1, model_task + 1):
            task = tasks[eval_task - 1]
            result_file = (
                result_root
                / task["dataset"]
                / f"hyper-task{model_task}"
                / "Result.text"
            )
            score = read_score(result_file, task["metric"])
            row[eval_task - 1] = score
            named_row[task["dataset"]] = score
        matrix.append(row)
        matrix_by_stage[f"Task{model_task}"] = named_row
        stage_averages.append(rounded_mean(row[:model_task]))

    final_row = matrix[-1]
    diagonal = [matrix[index][index] for index in range(args.num_tasks)]
    final_performance = {
        task["dataset"]: final_row[index] for index, task in enumerate(tasks)
    }

    if args.num_tasks > 1:
        backward_transfer_by_task = {
            tasks[index]["dataset"]: round(final_row[index] - diagonal[index], 6)
            for index in range(args.num_tasks - 1)
        }
        bwt = rounded_mean(backward_transfer_by_task.values())
    else:
        backward_transfer_by_task = {}
        bwt = 0.0

    metrics = {
        "MAA": rounded_mean(stage_averages),
        "MFN": rounded_mean(final_row),
        "MFT": rounded_mean(diagonal),
        "BWT": bwt,
    }

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root),
        "num_tasks": args.num_tasks,
        "score_unit": "percentage_points",
        "task_order": tasks,
        "performance_matrix": matrix,
        "performance_matrix_by_stage": matrix_by_stage,
        "stage_average_performance": {
            f"Task{index + 1}": value
            for index, value in enumerate(stage_averages)
        },
        "new_task_performance": {
            task["dataset"]: diagonal[index]
            for index, task in enumerate(tasks)
        },
        "final_model": f"Task{args.num_tasks}",
        "final_performance_by_task": final_performance,
        "backward_transfer_by_task": backward_transfer_by_task,
        "metrics": metrics,
        "metric_definitions": METRIC_DEFINITIONS,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_file": str(output_file), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
