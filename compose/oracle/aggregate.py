import argparse
import json
import os
from typing import Dict, List

from .cache import read_jsonl
from .metrics import summarize_oracle_records


def aggregate_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    summary = summarize_oracle_records(records)
    set_count = len(records[0]["set_nll"])
    if any(len(row["set_nll"]) != set_count for row in records):
        raise ValueError("oracle cache rows have inconsistent candidate counts")
    active_counts = [
        len(row["candidate_expert_ids"][int(row["best_overall_index"])])
        for row in records
    ]
    relative_synergy = [
        float(row["synergy"]) / max(abs(float(row["best_single_loss"])), 1e-12)
        for row in records
    ]
    summary.update(
        {
            "mean_nll_by_set": [
                sum(float(row["set_nll"][index]) for row in records) / len(records)
                for index in range(set_count)
            ],
            "average_active_experts": sum(active_counts) / len(active_counts),
            "synergy_threshold_pair_acceptance": {
                str(threshold): sum(value > threshold for value in relative_synergy)
                / len(relative_synergy)
                for threshold in (0.0, 0.01, 0.02, 0.03, 0.05)
            },
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    records = list(read_jsonl(args.input_file))
    result = aggregate_records(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
