#!/usr/bin/env python3
"""Export answer-free Router features for Sufficiency Head training/evaluation."""

import argparse
import json
from pathlib import Path

import torch

from compose.router.checkpoint import load_router_checkpoint, load_set_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.inference import predict_sets
from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs
from compose.router.set_router import ExpertSetRouter


def selected_scores(output, predictions):
    values = []
    for row, selected in enumerate(predictions):
        cardinality = len(selected)
        value = output.cardinality_logits[row, cardinality]
        if cardinality == 1:
            index = output.candidate_expert_ids[row].index(selected[0])
            value = value + output.single_scores[row][index]
        elif cardinality == 2:
            index = output.pair_ids[row].index(tuple(selected))
            value = value + output.pair_scores[row][index]
        values.append(value)
    return torch.stack(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--query-key-checkpoint", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bundle = torch.load(args.features, map_location="cpu")
    splits = set(bundle["splits"])
    if bundle.get("test_data_used") is not False or len(splits) != 1 or next(iter(splits)) not in ("train", "validation"):
        raise ValueError("Router feature export accepts one non-test train/validation split")
    device = torch.device(args.device)
    encoder = MultimodalQueryEncoder(bundle["image_features"].shape[1], bundle["text_features"].shape[1]).to(device)
    keys = ExpertKeyStore([ExpertKeyMetadata(**item) for item in bundle["expert_metadata"]]).to(device)
    load_router_checkpoint(args.query_key_checkpoint, encoder, keys, map_location=device)
    router = ExpertSetRouter().to(device)
    extra = load_set_router_checkpoint(args.router_checkpoint, router, map_location=device)
    encoder.eval(); keys.eval(); router.eval()
    inputs = QueryInputs(*(bundle[name].to(device) for name in
                           ("image_features", "text_features", "image_available", "text_available")))
    with torch.inference_mode():
        queries = encoder(inputs)
        visible = torch.tensor([[keys.metadata[expert_id].creation_task < int(task_id) for expert_id in keys.expert_ids]
                                for task_id in bundle["task_ids"]], dtype=torch.bool, device=device)
        routed = router(queries, keys.normalized(), keys.expert_ids, visible)
        predictions = predict_sets(routed, float(extra["pair_threshold"]))
        scores = selected_scores(routed, predictions)
        cardinality_probabilities = torch.softmax(routed.cardinality_logits, dim=-1)
        confidences = torch.stack([cardinality_probabilities[row, len(selected)]
                                   for row, selected in enumerate(predictions)])
    violations = sum(any(keys.metadata[value].creation_task >= int(task_id) for value in selected)
                     for task_id, selected in zip(bundle["task_ids"], predictions))
    if violations:
        raise ValueError("Router feature export contains future experts")
    augmented = dict(bundle)
    augmented.update({
        "queries": queries.cpu(), "cardinality_logits": routed.cardinality_logits.cpu(),
        "top_similarities": routed.top_similarities.cpu(), "visible_counts": visible.sum(dim=1).cpu(),
        "predicted_sets": predictions, "predicted_set_scores": scores.cpu(),
        "router_confidences": confidences.cpu(), "pair_threshold": float(extra["pair_threshold"]),
        "router_feature_source": "query_cardinality_similarity_predicted_set_only",
        "answer_features_used": False, "task_id_lookup_used": False,
        "temporal_violation_count": 0,
    })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, target)
    print(json.dumps({"status": "EXPORTED", "split": next(iter(splits)), "samples": len(predictions),
                      "temporal_violation_count": 0, "answer_features_used": False,
                      "task_id_lookup_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
