"""Verify evaluation completeness for the format-controlled experiment.

Checks per seed: every (dataset, model) evaluation exists, has the expected
sample count, matches the eval-file sample ids exactly, and every row carries
the full preregistered per-sample field set.
"""

import argparse
import json
from pathlib import Path

SEEDS = (42, 43, 44)
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
REQUIRED_FIELDS = [
    "sample_id", "scene_id", "task", "split", "seed", "selection_name",
    "expert_ids", "gates", "normalization", "target", "prediction", "correct",
    "logit_A", "logit_B", "probability_A", "probability_B",
    "answer_token_nll", "brier", "polarity", "negative_type",
    "required_functions", "queried_shape", "queried_count", "queried_relation",
    "true_count", "latency", "peak_memory", "answer_token_position",
    "supervised_token_ids", "free_prediction", "free_correct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    root = Path(args.evaluation_root)
    data_root = Path(args.data_root)
    errors = []
    checked = 0
    for seed in (int(value) for value in args.seeds.split(",")):
        for dataset, models in CONFIGS.items():
            eval_path = data_root / "instructions" / dataset / "test_eval.json"
            expected_ids = [str(row["question_id"]) for row in json.loads(eval_path.read_text(encoding="utf-8"))]
            for model in models:
                checked += 1
                output = root / "seed{}".format(seed) / dataset / model
                per_sample = output / "per_sample.jsonl"
                summary = output / "summary.json"
                if not per_sample.is_file() or not summary.is_file():
                    errors.append("missing {} seed{}".format(output, seed))
                    continue
                with per_sample.open(encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
                actual_ids = [str(row["sample_id"]) for row in rows]
                if actual_ids != expected_ids:
                    errors.append(
                        "sample id mismatch {} seed{}: {} vs {}".format(
                            output, seed, len(actual_ids), len(expected_ids))
                    )
                missing_fields = [
                    field for field in REQUIRED_FIELDS
                    if any(field not in row for row in rows)
                ]
                if missing_fields:
                    errors.append("missing fields {} in {}".format(missing_fields, output))
                if any(row["supervised_token_ids"][0] not in (319, 350) for row in rows):
                    errors.append("non-single-token answer in {}".format(output))
    print(json.dumps({
        "checked": checked,
        "error_count": len(errors),
        "errors": errors[:50],
        "passed": not errors,
    }, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
