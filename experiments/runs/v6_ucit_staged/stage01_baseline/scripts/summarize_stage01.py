#!/usr/bin/env python3
"""Stage 01: summarize continual metrics from evaluation Result.text files.

Reuses the exact formulas of scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py:
  MAA = mean over stages i of mean(performance on tasks 1..i after training Task i)
  MFN = mean of final row R[N][j]
  MFT = mean of diagonal R[i][i]
  BWT = mean over old tasks j of (R[N][j] - R[j][j])
Includes an independent hand-computed verification on a small matrix.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
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

# CLEVR results live under directory "CLEVR"
DIR_BY_TASK = {1: "ImageNet-R", 2: "ArxivQA", 3: "VizWiz", 4: "IconQA", 5: "CLEVR", 6: "Flickr30k"}


def read_score(path: Path, metric: str):
    text = path.read_text()
    m = re.search(rf"^{metric}\s*:\s*([0-9.]+)", text, re.MULTILINE)
    if not m:
        m = re.search(rf"^[A-Za-z ]*{metric}\s*:\s*([0-9.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def build_matrix(eval_root: Path, stage_prefix: str = "hyper-task", num_tasks: int = 6) -> tuple[list, list[list]]:
    """Returns (score_keys, matrix[i][j]) where i is the finished stage (1-based),
    j the evaluated task (1-based). Reads <eval_root>/<Dataset>/<stage>/Result.text."""
    matrix = [[None] * num_tasks for _ in range(num_tasks)]
    scores = [[None] * num_tasks for _ in range(num_tasks)]
    for i in range(1, num_tasks + 1):
        for j in range(1, i + 1):
            p = eval_root / DIR_BY_TASK[j] / f"{stage_prefix}{i}" / "Result.text"
            if not p.exists():
                print(f"missing: {p}")
                continue
            metric = TASKS[j - 1]["metric"]
            s = read_score(p, metric)
            if s is None:
                print(f"unparseable: {p}")
                continue
            matrix[i - 1][j - 1] = s
            scores[i - 1][j - 1] = (str(p), metric)
    return scores, matrix


def metrics_from_matrix(matrix):
    n = len(matrix)
    diag = [matrix[i][i] for i in range(n) if matrix[i][i] is not None]
    mft = fmean(diag) if diag else None
    finals = [matrix[n - 1][j] for j in range(n) if matrix[n - 1][j] is not None]
    mfn = fmean(finals) if finals else None
    stage_means = []
    for i in range(n):
        row = [matrix[i][j] for j in range(i + 1) if matrix[i][j] is not None]
        if row:
            stage_means.append(fmean(row))
    maa = fmean(stage_means) if stage_means else None
    bwt_terms = []
    for j in range(n - 1):
        if matrix[n - 1][j] is not None and matrix[j][j] is not None:
            bwt_terms.append(matrix[n - 1][j] - matrix[j][j])
    bwt = fmean(bwt_terms) if bwt_terms else None
    return {"MAA": maa, "MFN": mfn, "MFT": mft, "BWT": bwt}


def verify_formula():
    """Independent hand-computed check on a fixed 3x3 matrix."""
    M = [[90.0, None, None], [80.0, 85.0, None], [70.0, 75.0, 88.0]]
    m = metrics_from_matrix(M)
    assert abs(m["MAA"] - (90 + (80 + 85) / 2 + (70 + 75 + 88) / 3) / 3) < 1e-9
    assert abs(m["MFN"] - (70 + 75 + 88) / 3) < 1e-9
    assert abs(m["MFT"] - (90 + 85 + 88) / 3) < 1e-9
    assert abs(m["BWT"] - ((70 - 90) + (75 - 85)) / 2) < 1e-9
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--stage-prefix", default="hyper-task")
    ap.add_argument("--num-tasks", type=int, default=6)
    args = ap.parse_args()

    verified = verify_formula()
    print("formula verification on hand matrix:", json.dumps(verified))

    scores, matrix = build_matrix(args.eval_root, args.stage_prefix, args.num_tasks)
    metrics = metrics_from_matrix(matrix)
    out = {
        "name": args.name,
        "task_order": [t["dataset"] for t in TASKS],
        "accuracy_matrix": matrix,
        "metrics": metrics,
        "formula_verified": verified,
        "metric_definitions": {
            "MAA": "mean over stages i of mean(seen-task performance after task i)",
            "MFN": "mean of final row",
            "MFT": "mean diagonal",
            "BWT": "mean over old tasks (final - at-learn)",
        },
    }
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    print(json.dumps({"matrix": matrix, "metrics": metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
