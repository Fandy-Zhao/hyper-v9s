"""Stage 00: compare reproduction evaluations against the recorded formal ones.

For every (seed, dataset, model) reproduction eval, compares summary metrics
(accuracy %, answer-token NLL, Brier, ECE) against the recorded formal
summary, and per-sample logits/NLL/predictions against the recorded
per-sample rows.

Tolerance for the pre-registered gate: accuracy within 0.2 percentage
points; per-sample predictions identical; per-sample |delta logit| reported
(not gated).

Usage:
  python -m compose.eval.dual_lora_stage00_repro_compare \
      --repro-root experiments/runs/dual_lora_stage00/reproduction \
      --recorded-root experiments/runs/format_controlled_composition_v1/evaluation \
      --output artifacts/dual_lora_stage00/reproduction.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Tuple

REPRO_CONFIGS = {
    42: {"A_only": ["expert_a"], "B_only": ["independent_b", "residual_b"], "C_only": ["expert_c"],
         "A_plus_B": ["a_residual_b", "a_independent_b", "rank16_ab"],
         "B_plus_C": ["independent_b_c", "residual_b_c", "upper_bc"]},
    43: {"A_only": ["expert_a"], "B_only": ["independent_b", "residual_b"], "C_only": ["expert_c"],
         "A_plus_B": ["a_residual_b", "a_independent_b", "rank16_ab"],
         "B_plus_C": ["independent_b_c", "residual_b_c", "upper_bc"]},
    44: {"A_only": ["expert_a"], "B_only": ["independent_b", "residual_b"], "C_only": ["expert_c"],
         "A_plus_B": ["a_residual_b", "a_independent_b", "rank16_ab"],
         "B_plus_C": ["independent_b_c", "residual_b_c", "upper_bc"]},
}

SUMMARY_KEYS = ("accuracy_percent", "mean_answer_token_nll", "brier", "ece_15_bins")


def read_summary(path: Path) -> Dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_per_sample(path: Path) -> Dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def compare_per_sample(repro: Dict[str, dict], recorded: Dict[str, dict]) -> Tuple[float, int, int]:
    common = sorted(set(repro) & set(recorded))
    worst_logit = 0.0
    worst_nll = 0.0
    pred_mismatch = 0
    for sid in common:
        r, o = repro[sid], recorded[sid]
        worst_logit = max(worst_logit, abs(float(r["logit_A"]) - float(o["logit_A"])),
                          abs(float(r["logit_B"]) - float(o["logit_B"])))
        worst_nll = max(worst_nll, abs(float(r["answer_token_nll"]) - float(o["answer_token_nll"])))
        if (float(r["logit_A"]) > float(r["logit_B"])) != (float(o["logit_A"]) > float(o["logit_B"])):
            pred_mismatch += 1
    return worst_logit, worst_nll, pred_mismatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-root", required=True)
    parser.add_argument("--recorded-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    failures = []
    pending = 0
    for seed, datasets in REPRO_CONFIGS.items():
        for dataset, models in datasets.items():
            for model in models:
                repro_dir = Path(args.repro_root) / "seed{}".format(seed) / dataset / model
                recorded_dir = Path(args.recorded_root) / "seed{}".format(seed) / dataset / model
                if not (repro_dir / "summary.json").is_file():
                    pending += 1
                    continue
                repro_summary = read_summary(repro_dir / "summary.json")
                recorded_summary = read_summary(recorded_dir / "summary.json")
                delta = {key: float(repro_summary[key]) - float(recorded_summary[key]) for key in SUMMARY_KEYS}
                worst_logit, worst_nll, pred_mismatch = compare_per_sample(
                    read_per_sample(repro_dir / "per_sample.jsonl"),
                    read_per_sample(recorded_dir / "per_sample.jsonl"),
                )
                acc_ok = abs(delta["accuracy_percent"]) <= 0.2
                nll_ok = abs(delta["mean_answer_token_nll"]) <= 0.01
                pred_ok = pred_mismatch == 0
                ok = acc_ok and nll_ok and pred_ok
                rows.append({
                    "seed": seed, "dataset": dataset, "model": model,
                    "recorded_accuracy_percent": recorded_summary["accuracy_percent"],
                    "repro_accuracy_percent": repro_summary["accuracy_percent"],
                    "delta_accuracy_pp": delta["accuracy_percent"],
                    "recorded_nll": recorded_summary["mean_answer_token_nll"],
                    "repro_nll": repro_summary["mean_answer_token_nll"],
                    "delta_nll": delta["mean_answer_token_nll"],
                    "delta_brier": delta["brier"], "delta_ece": delta["ece_15_bins"],
                    "per_sample_max_abs_logit_diff": worst_logit,
                    "per_sample_max_abs_nll_diff": worst_nll,
                    "prediction_mismatches": pred_mismatch,
                    "samples": len(read_per_sample(repro_dir / "per_sample.jsonl")),
                    "pass_0_2pp_and_identical_predictions": ok,
                })
                if not ok:
                    failures.append("seed{} {}/{} acc_delta={:.4f}pp nll_delta={:.6f} pred_mismatch={}".format(
                        seed, dataset, model, delta["accuracy_percent"], delta["mean_answer_token_nll"], pred_mismatch))

    passed = not failures and pending == 0
    if rows:
        with open(args.output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("compared {} evals; pending {}; failures: {}".format(len(rows), pending, len(failures)))
    for failure in failures:
        print("  FAIL", failure)
    print("ALL_PASSED" if passed else "REPRODUCTION_FAILED_OR_INCOMPLETE")


if __name__ == "__main__":
    main()
