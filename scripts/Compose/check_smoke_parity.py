"""Smoke parity checks (spec §15/§17/§19): single-GPU vs 4-GPU.

Usage:

    python -m scripts.Compose.check_smoke_parity \
      --features-a <single/features.json> --features-b <merged-4gpu/features.json> \
      --rms-a <single/rms_calibration.json> --rms-b <4gpu/rms_calibration.json> \
      --answers-a <single answers.jsonl> --answers-b <4gpu answers.jsonl>

Every provided pair is compared; missing pairs are skipped.

- features (spec §15): per-sample visual/text/query vectors, max abs diff
  <= 1e-5 across the whole intersection of sample ids.
- rms (spec §17): kappa map per (layer, expert), max abs diff <= 1e-5.
- answers (spec §19): line-for-line byte equality (deterministic
  generation: same snapshot, config, seed, prompt, processor).

Exit code 0 = all provided pairs PASS; nonzero on any violation.
"""

import argparse
import json
import sys

FEATURE_TOLERANCE = 1e-5
RMS_TOLERANCE = 1e-5


def _max_abs_diff(values_a, values_b):
    if len(values_a) != len(values_b):
        raise ValueError("length mismatch: {} vs {}".format(len(values_a), len(values_b)))
    return max(
        (abs(float(value_a) - float(value_b)) for value_a, value_b in zip(values_a, values_b)),
        default=0.0,
    )


def compare_features(path_a, path_b, tolerance):
    with open(path_a, "r", encoding="utf-8") as handle:
        payload_a = json.load(handle)
    with open(path_b, "r", encoding="utf-8") as handle:
        payload_b = json.load(handle)
    records_a = payload_a["records"]
    records_b = payload_b["records"]
    common = sorted(set(records_a) & set(records_b))
    missing_a = sorted(set(records_b) - set(records_a))
    missing_b = sorted(set(records_a) - set(records_b))
    if missing_a or missing_b:
        raise ValueError(
            "sample id mismatch: only-a={} only-b={}".format(len(missing_b), len(missing_a))
        )
    worst = 0.0
    for sample_id in common:
        for field in ("visual_feature", "text_feature", "query"):
            diff = _max_abs_diff(records_a[sample_id][field], records_b[sample_id][field])
            worst = max(worst, diff)
            if diff > tolerance:
                raise ValueError(
                    "{} sample {} field {} diff {} > {}".format(
                        path_b, sample_id, field, diff, tolerance
                    )
                )
    return len(common), worst


def compare_rms(path_a, path_b, tolerance):
    with open(path_a, "r", encoding="utf-8") as handle:
        map_a = json.load(handle)
    with open(path_b, "r", encoding="utf-8") as handle:
        map_b = json.load(handle)
    worst = 0.0
    layers = sorted(set(map_a) | set(map_b))
    for layer in layers:
        layer_a = map_a.get(layer, {})
        layer_b = map_b.get(layer, {})
        for expert_id in sorted(set(layer_a) | set(layer_b)):
            value_a = layer_a.get(str(expert_id))
            value_b = layer_b.get(str(expert_id))
            if value_a is None or value_b is None:
                raise ValueError(
                    "{} expert {} missing in one calibration (layer {})".format(
                        path_b, expert_id, layer
                    )
                )
            diff = abs(float(value_a) - float(value_b))
            worst = max(worst, diff)
            if diff > tolerance:
                raise ValueError(
                    "layer {} expert {} kappa diff {} > {}".format(
                        layer, expert_id, diff, tolerance
                    )
                )
    return len(layers), worst


def compare_answers(path_a, path_b):
    with open(path_a, "r", encoding="utf-8") as handle:
        lines_a = handle.readlines()
    with open(path_b, "r", encoding="utf-8") as handle:
        lines_b = handle.readlines()
    if len(lines_a) != len(lines_b):
        raise ValueError("answer line count mismatch: {} vs {}".format(len(lines_a), len(lines_b)))
    for index, (line_a, line_b) in enumerate(zip(lines_a, lines_b)):
        if line_a != line_b:
            raise ValueError("answer line {} differs:\n  A: {}\n  B: {}".format(index, line_a.strip(), line_b.strip()))
    return len(lines_a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-a")
    parser.add_argument("--features-b")
    parser.add_argument("--rms-a")
    parser.add_argument("--rms-b")
    parser.add_argument("--answers-a")
    parser.add_argument("--answers-b")
    args = parser.parse_args()

    checks = []
    if args.features_a and args.features_b:
        count, worst = compare_features(args.features_a, args.features_b, FEATURE_TOLERANCE)
        checks.append("FEATURES({} samples) max_abs_diff={:.3e} <= 1e-5".format(count, worst))
    if args.rms_a and args.rms_b:
        layers, worst = compare_rms(args.rms_a, args.rms_b, RMS_TOLERANCE)
        checks.append("RMS({} layers) max_abs_diff={:.3e} <= 1e-5".format(layers, worst))
    if args.answers_a and args.answers_b:
        lines = compare_answers(args.answers_a, args.answers_b)
        checks.append("ANSWERS({} lines) byte-identical".format(lines))
    if not checks:
        raise SystemExit("no comparison pairs provided")

    for check in checks:
        print("PASS {}".format(check))
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
    sys.exit(0)
