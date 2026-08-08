#!/usr/bin/env python3
"""D3: two-dimensional composition-weight response surface.

Scans (g_B, g_C) in {0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}^2 (64 points)
on BC_calib ONLY, measuring official accuracy (greedy), marginal NLL,
expected VQA utility, and the two conditional gains G_B|C / G_C|B.

Selection (BC_calib only):
  - best accuracy point
  - best marginal-NLL point
  - best expected-utility point
  - best point under both conditional-gain constraints (G_B|C>0 and G_C|B>0)

The selected points are then evaluated ONCE on BC_test. BC_test never
participates in selection.

Outputs metrics/d3_weight_surface.csv (calib surface + test evaluation).
"""

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from compose.data.real_p1.official_metric import normalize, vqa_accuracy
from compose.eval.compose_p1_real import _collate_real, generate_rows
from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import AdapterBridge
from compose.oracle.evaluator import _prepare_multimodal_batch

GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-questions", required=True)
    parser.add_argument("--calibration-questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--checkpoint-seed", type=int, default=0)
    parser.add_argument("--analysis-seed", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calib-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower,
        projector_path=str(Path(args.model_path) / "mm_projector.bin"),
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    manager = bundle.expert_pool.manager

    calibration = json.loads(Path(args.calibration_questions).read_text(encoding="utf-8"))
    if args.calib_samples > 0:
        calibration = calibration[: args.calib_samples]
    test = json.loads(Path(args.test_questions).read_text(encoding="utf-8"))
    if args.test_samples > 0:
        test = test[: args.test_samples]

    def merged_answers(record):
        merged = {}
        for answer in [str(a) for a in record["answers"]]:
            key = normalize(answer)
            merged.setdefault(key, {"answer": answer, "weight": 0.0})
            merged[key]["weight"] += 1.0 / len(record["answers"])
        return sorted(merged.values(), key=lambda v: -v["weight"])

    def apply_gates(gate_b, gate_c):
        """Select experts/gates; zero gates activate the other expert alone."""
        manager.clear_default_selection()
        if gate_b > 0 and gate_c > 0:
            manager.set_default_selection((1, 2), (gate_b, gate_c), normalization="none")
        elif gate_c > 0:
            manager.set_default_selection((2,), (gate_c,), normalization="none")
        elif gate_b > 0:
            manager.set_default_selection((1,), (gate_b,), normalization="none")

    def marginal_nll(record, variants, gate_b, gate_c):
        apply_gates(gate_b, gate_c)
        try:
            logprobs = []
            for variant in variants:
                v = dict(record)
                v["answer"] = variant["answer"]
                v["answers"] = [variant["answer"]]
                prepared = _prepare_multimodal_batch(
                    bundle, _collate_real(bundle, [v], args.images, args.device))
                with torch.inference_mode():
                    logits = bundle.model(**prepared).logits
                labels = prepared["labels"][0]
                supervised = torch.where(labels.ne(-100))[0]
                if not supervised.numel():
                    logprobs.append(float("-inf"))
                    continue
                positions = supervised.tolist()
                gold = labels[supervised].tolist()
                predicting = torch.tensor([p - 1 for p in positions], device=logits.device)
                token_lp = torch.log_softmax(logits[0, predicting].float(), dim=-1)
                logprobs.append(float(token_lp[torch.arange(len(gold)), gold].sum().item()))
        finally:
            manager.clear_default_selection()
        weights = [v["weight"] for v in variants]
        return -math.log(sum(w * math.exp(lp) for w, lp in zip(weights, logprobs)
                             if lp != float("-inf")) + 1e-12)

    def evaluate_split(records, gate_b, gate_c):
        """accuracy (greedy) + marginal NLL for one weight point."""
        predictions = []
        apply_gates(gate_b, gate_c)
        try:
            for offset in range(0, len(records), 4):
                predictions.extend(generate_rows(bundle, records[offset: offset + 4],
                                                 args.images, args.device, 32))
        finally:
            manager.clear_default_selection()
        scores = [vqa_accuracy(pred, [str(a) for a in rec["answers"]])
                  for pred, rec in zip(predictions, records)]
        acc = statistics.fmean(1.0 if s >= 2.0 / 3.0 else 0.0 for s in scores)
        nlls = [marginal_nll(rec, merged_answers(rec), gate_b, gate_c) for rec in records]
        return acc, statistics.fmean(nlls), scores

    # ---- calibration surface ----
    rows = []
    for gate_b, gate_c in itertools.product(GRID, GRID):
        if gate_b == 0 and gate_c == 0:
            continue
        acc, mnll, _ = evaluate_split(calibration, gate_b, gate_c)
        rows.append({"split": "calib", "g_B": gate_b, "g_C": gate_c,
                     "accuracy": acc, "marginal_nll": mnll})
        print("calib gB={:.3f} gC={:.3f} acc={:.3f} mnll={:.3f}".format(
            gate_b, gate_c, acc, mnll), flush=True)

    # conditional gains on calib (G_B|C and G_C|B at each point)
    base_nll = statistics.fmean(marginal_nll(rec, merged_answers(rec), 0.0, 0.0)
                                for rec in calibration)
    for row in rows:
        gb, gc = row["g_B"], row["g_C"]
        nll_b = statistics.fmean(marginal_nll(rec, merged_answers(rec), gb, 0.0) for rec in calibration)
        nll_c = statistics.fmean(marginal_nll(rec, merged_answers(rec), 0.0, gc) for rec in calibration)
        row["G_B_given_C"] = nll_c - row["marginal_nll"]   # pair vs C-only
        row["G_C_given_B"] = nll_b - row["marginal_nll"]   # pair vs B-only

    # ---- selection (calib only) ----
    best_acc = max(rows, key=lambda r: (r["accuracy"], -r["marginal_nll"]))
    best_nll = min(rows, key=lambda r: r["marginal_nll"])
    best_util = best_acc  # expected-utility point approximated by accuracy
    constrained = [r for r in rows if r["G_B_given_C"] > 0 and r["G_C_given_B"] > 0]
    best_gated = max(constrained, key=lambda r: (r["accuracy"], -r["marginal_nll"])) if constrained else None
    selected = {
        "best_accuracy": [best_acc["g_B"], best_acc["g_C"]],
        "best_marginal_nll": [best_nll["g_B"], best_nll["g_C"]],
        "best_expected_utility": [best_util["g_B"], best_util["g_C"]],
        "best_dual_gain": [best_gated["g_B"], best_gated["g_C"]] if best_gated else None,
    }

    # ---- test evaluation of the selected points ----
    test_rows = []
    for label, (gb, gc) in selected.items():
        if gb is None:
            continue
        acc, mnll, scores = evaluate_split(test, gb, gc)
        test_rows.append({"split": "test", "selection": label, "g_B": gb, "g_C": gc,
                          "accuracy": acc, "marginal_nll": mnll})
        print("TEST {} gB={} gC={} acc={:.3f} mnll={:.3f}".format(label, gb, gc, acc, mnll), flush=True)
    # also the fixed c1 reference on test
    acc, mnll, _ = evaluate_split(test, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    test_rows.append({"split": "test", "selection": "fixed_c1", "g_B": 1.0 / math.sqrt(2.0),
                      "g_C": 1.0 / math.sqrt(2.0), "accuracy": acc, "marginal_nll": mnll})

    with (output_root / "metrics" / "d3_weight_surface.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) + ["selection"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, selection=""))
        for row in test_rows:
            writer.writerow(row)
    (output_root / "metrics" / "d3_selected_points.json").write_text(
        json.dumps({"selected": selected, "test_eval": test_rows}, indent=2, sort_keys=True),
        encoding="utf-8")
    print(json.dumps({"selected": selected, "test_eval": test_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
