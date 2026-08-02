#!/usr/bin/env python3
"""Validate Set Router, preregister threshold once, and report Oracle regret."""

import argparse
import json
import time
from pathlib import Path

import torch

from compose.router.checkpoint import load_router_checkpoint, load_set_router_checkpoint, save_set_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.inference import predict_sets
from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs
from compose.router.router_metrics import set_routing_metrics
from compose.router.set_router import ExpertSetRouter


def candidate_score(record, selected):
    selected = tuple(sorted(selected))
    candidates = [record["empty"], *record["singles"], *record["pairs"]]
    for value in candidates:
        if tuple(value["expert_ids"]) == selected:
            return value
    return record["empty"]


def candidate_nll(record, selected):
    return float(candidate_score(record, selected)["mean_nll"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True); parser.add_argument("--query-key-checkpoint", required=True)
    parser.add_argument("--router-checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--features-output"); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bundle = torch.load(args.features, map_location="cpu")
    if bundle.get("test_data_used") is not False or any(value != "validation" for value in bundle["splits"]): raise ValueError("threshold selection requires validation only")
    device = torch.device(args.device)
    encoder = MultimodalQueryEncoder(bundle["image_features"].shape[1], bundle["text_features"].shape[1]).to(device)
    keys = ExpertKeyStore([ExpertKeyMetadata(**item) for item in bundle["expert_metadata"]]).to(device)
    load_router_checkpoint(args.query_key_checkpoint, encoder, keys, map_location=device)
    router = ExpertSetRouter().to(device); extra = load_set_router_checkpoint(args.router_checkpoint, router, map_location=device)
    inputs = QueryInputs(*(bundle[name].to(device) for name in ("image_features", "text_features", "image_available", "text_available")))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        queries = encoder(inputs); key_values = keys.normalized()
        visible = torch.tensor([[keys.metadata[value].creation_task < int(task) for value in keys.expert_ids] for task in bundle["task_ids"]], device=device)
        output = router(queries, key_values, keys.expert_ids, visible)
    if device.type == "cuda": torch.cuda.synchronize(device)
    routing_seconds = time.perf_counter() - started
    candidates = {}
    for threshold in (0.50, 0.65, 0.80):
        predictions = predict_sets(output, threshold)
        metrics = set_routing_metrics(predictions, bundle["oracle_sets"])
        regrets = [candidate_nll(record, predicted) - float(record["selected_mean_nll"]) for record, predicted in zip(bundle["oracle_records"], predictions)]
        best_single_regrets = [candidate_nll(record, predicted) - float(record["best_single_nll"] if record["best_single_nll"] is not None else record["empty"]["mean_nll"])
                               for record, predicted in zip(bundle["oracle_records"], predictions)]
        accuracy_regrets = [float(candidate_score(record, record["selected_expert_ids"])["exact_teacher_forced"]) - float(candidate_score(record, predicted)["exact_teacher_forced"])
                            for record, predicted in zip(bundle["oracle_records"], predictions)]
        score_regrets = [float(candidate_score(record, predicted)["score"]) - float(record["selected_score"])
                         for record, predicted in zip(bundle["oracle_records"], predictions)]
        pair_rows = [index for index, target in enumerate(bundle["oracle_sets"]) if len(target) == 2]
        harmful = [index for index, predicted in enumerate(predictions) if len(predicted) == 2 and candidate_nll(bundle["oracle_records"][index], predicted) >
                   float(bundle["oracle_records"][index]["best_single_nll"])]
        metrics["predicted_minus_oracle_nll"] = sum(regrets) / len(regrets)
        metrics["predicted_minus_best_single_nll"] = sum(best_single_regrets) / len(best_single_regrets)
        metrics["accuracy_regret"] = sum(accuracy_regrets) / len(accuracy_regrets)
        metrics["selected_set_score_regret"] = sum(score_regrets) / len(score_regrets)
        metrics["pair_only_regret"] = sum(regrets[index] for index in pair_rows) / max(1, len(pair_rows))
        metrics["predicted_pair_harmful_rate"] = len(harmful) / max(1, sum(len(value) == 2 for value in predictions))
        metrics["worst_10_percent_regret"] = sum(sorted(regrets, reverse=True)[:max(1, len(regrets) // 10)]) / max(1, len(regrets) // 10)
        metrics["candidate_recall_upper_bound"] = sum(set(target).issubset(output.candidate_expert_ids[index]) for index, target in enumerate(bundle["oracle_sets"]) if target) / max(1, sum(bool(value) for value in bundle["oracle_sets"]))
        true_members = {(index, member) for index, row in enumerate(bundle["oracle_sets"]) for member in row}
        pred_members = {(index, member) for index, row in enumerate(predictions) for member in row}
        tp, fp, fn = len(true_members & pred_members), len(pred_members - true_members), len(true_members - pred_members)
        metrics["expert_member_micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        expert_f1 = []
        for expert_id in keys.expert_ids:
            expert_true = {index for index, row in enumerate(bundle["oracle_sets"]) if expert_id in row}
            expert_pred = {index for index, row in enumerate(predictions) if expert_id in row}
            expert_tp = len(expert_true & expert_pred)
            expert_f1.append(2 * expert_tp / max(1, 2 * expert_tp + len(expert_pred - expert_true) + len(expert_true - expert_pred)))
        metrics["expert_member_macro_f1"] = sum(expert_f1) / len(expert_f1)
        pair_false_positives = sum(len(predicted) == 2 and len(target) != 2 for predicted, target in zip(predictions, bundle["oracle_sets"]))
        metrics["pair_false_positive_rate"] = pair_false_positives / max(1, sum(len(target) != 2 for target in bundle["oracle_sets"]))
        metrics["pair_both_members_accuracy"] = sum(set(bundle["oracle_sets"][index]) == set(predictions[index]) for index in pair_rows) / max(1, len(pair_rows))
        candidates[str(threshold)] = {"predictions": predictions, "metrics": metrics}
    # Lowest NLL regret, then highest Pair precision, then higher threshold is the frozen deterministic rule.
    selected = min((0.50, 0.65, 0.80), key=lambda value: (candidates[str(value)]["metrics"]["predicted_minus_oracle_nll"],
                                                           -candidates[str(value)]["metrics"]["Pair_precision"], -value))
    frozen_threshold = extra.get("pair_threshold")
    if frozen_threshold is not None and float(frozen_threshold) != selected:
        raise ValueError("validation-selected pair threshold changed after it was frozen")
    if frozen_threshold is None:
        extra["pair_threshold"] = selected
        save_set_router_checkpoint(args.router_checkpoint, router, extra)
    single_only = predict_sets(output, selected, single_only=True)
    report = {"status": "VALIDATED_THRESHOLD_FROZEN", "selected_pair_threshold": selected,
              "selected_metrics": candidates[str(selected)]["metrics"], "threshold_candidates": {key: value["metrics"] for key, value in candidates.items()},
              "predicted_single_only_metrics": set_routing_metrics(single_only, bundle["oracle_sets"]),
              "routing_latency_seconds_per_sample": routing_seconds / len(bundle["sample_ids"]),
              "routing_throughput_samples_per_second": len(bundle["sample_ids"]) / routing_seconds,
              "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
              "temporal_violation_count": 0, "test_data_used": False}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.features_output:
        selected_predictions = candidates[str(selected)]["predictions"]
        augmented = dict(bundle)
        augmented.update({
            "queries": queries.cpu(), "cardinality_logits": output.cardinality_logits.cpu(),
            "top_similarities": output.top_similarities.cpu(), "visible_counts": visible.sum(dim=1).cpu(),
            "predicted_sets": selected_predictions,
            "predicted_set_scores": torch.tensor([
                float(output.cardinality_logits[row].max()) for row in range(len(selected_predictions))
            ]),
            "router_confidences": torch.softmax(output.cardinality_logits, dim=-1).max(dim=-1).values.cpu(),
        })
        feature_target = Path(args.features_output); feature_target.parent.mkdir(parents=True, exist_ok=True); torch.save(augmented, feature_target)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
