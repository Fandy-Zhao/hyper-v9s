#!/usr/bin/env python3
"""Supplement external-eval metrics for OCR-VQA / VQAv2.

The official VQA-accuracy is a 10-answer metric; the external pools carry a
single official answer (OCR-VQA official answers include title prefixes and
name formats; VQAv2 counting answers are single). Strict normalized matching
therefore under-reports semantically correct readings. This script recomputes
per-sample secondary metrics and writes them into the per-sample files:

  - strict_em: normalized exact match vs the official answer
  - token_overlap: Jaccard-like semantic match (>=2/3 shared tokens)
  - numeric_relaxed: relaxed numeric match (official_metric.relaxed_numeric_match)

The official metric remains the primary one; these are auxiliary diagnostics
for the domain-shift discussion (spec 4.2).
"""

import argparse
import json
import statistics
from pathlib import Path

import sys
sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA/compose/data/real_p1")
from official_metric import normalize, relaxed_numeric_match  # noqa: E402


def token_overlap_match(prediction, answer, min_ratio=2.0 / 3.0):
    pred = set(normalize(prediction).split())
    gold = set(normalize(answer).split())
    if not gold:
        return normalize(prediction) == normalize(answer)
    if not pred:
        return False
    shared = len(pred & gold)
    # must cover at least 2/3 of the gold tokens and not contradict
    return shared >= min_ratio * len(gold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--pools", default="external_ocrvqa_Bonly_test,external_vqav2_count_test")
    args = parser.parse_args()
    root = Path(args.predictions_root)
    for pool in args.pools.split(","):
        summaries = {}
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir():
                continue
            sample_file = seed_dir / pool / "per_sample.jsonl"
            if not sample_file.exists():
                continue
            rows = [json.loads(line) for line in sample_file.read_text().splitlines()]
            for row in rows:
                for mode, metrics in row["modes"].items():
                    gold = row["gold"]
                    metrics["strict_em"] = 1.0 if normalize(metrics["prediction"]) == normalize(gold) else 0.0
                    metrics["token_overlap"] = 1.0 if token_overlap_match(metrics["prediction"], gold) else 0.0
                    metrics["numeric_relaxed"] = relaxed_numeric_match(metrics["prediction"], [gold])
            sample_file.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8")
            for mode in rows[0]["modes"]:
                summaries.setdefault(mode, []).append(statistics.fmean(
                    row["modes"][mode]["token_overlap"] for row in rows))
        if summaries:
            print("{}: token-overlap accuracy per mode (mean over seeds):".format(pool))
            for mode, values in sorted(summaries.items()):
                print("  {:>12s}: {:.4f}".format(mode, statistics.fmean(values)))
        else:
            print("{}: no prediction files".format(pool))


if __name__ == "__main__":
    main()
