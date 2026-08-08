#!/usr/bin/env python3
"""Recompute correctness for single-answer eval pools.

The official VQA accuracy is designed for 10-answer records: a single match
scores 1/3, which never passes the >=2/3 correctness threshold. For the
external single-answer pools (OCR-VQA, VQAv2 counting), correctness is
recomputed as normalized exact match (score 1.0 iff the prediction equals the
official answer after normalization) and the mode summaries are rewritten.
The 10-answer B+C pools are untouched (official metric preserved).
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import sys
sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA/compose/data/real_p1")
from official_metric import normalize  # noqa: E402


def recompute_row(row, mode):
    metrics = row["modes"][mode]
    answers = row.get("answers", [row.get("gold", "")])
    if len(answers) > 1:
        return metrics  # keep official VQA accuracy for 10-answer pools
    score = 1.0 if normalize(metrics["prediction"]) == normalize(answers[0]) else 0.0
    correct = score == 1.0
    metrics["vqa_score"] = score
    metrics["correct"] = correct
    metrics["brier"] = (metrics["confidence"] - float(correct)) ** 2
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--pools",
                        default="external_ocrvqa_Bonly_test,external_vqav2_count_test")
    args = parser.parse_args()
    root = Path(args.predictions_root)
    for pool in args.pools.split(","):
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir():
                continue
            sample_file = seed_dir / pool / "per_sample.jsonl"
            if not sample_file.exists():
                continue
            rows = [json.loads(line) for line in sample_file.read_text().splitlines()]
            for row in rows:
                for mode in row["modes"]:
                    recompute_row(row, mode)
            sample_file.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8")
            # rewrite summary.json modes for this pool
            summary_path = seed_dir / pool / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            for mode in summary["modes"]:
                entries = [row["modes"][mode] for row in rows]
                accuracy = statistics.fmean(float(e["correct"]) for e in entries)
                summary["modes"][mode]["accuracy"] = accuracy
                summary["modes"][mode]["mean_vqa_score"] = statistics.fmean(
                    float(e["vqa_score"]) for e in entries)
                summary["modes"][mode]["mean_em"] = statistics.fmean(
                    1.0 if normalize(e["prediction"]) == normalize(rows[i]["gold"]) else 0.0
                    for i, e in enumerate(entries))
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
            print("recomputed {} seed {}".format(pool, seed_dir.name))


if __name__ == "__main__":
    main()
