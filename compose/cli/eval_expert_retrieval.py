#!/usr/bin/env python3
"""Evaluate Stage 05 retrieval without exposing answer-side fields to Query."""

import argparse
import json
from pathlib import Path

import torch

from compose.router.checkpoint import load_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.metrics import retrieval_metrics
from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs
from compose.router.retrieval import retrieve_experts
from compose.router.validation import validate_query_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bundle = torch.load(args.features, map_location="cpu")
    if bundle.get("test_data_used") is not False or any("test" in str(value).lower() for value in bundle["splits"]):
        raise ValueError("Stage 05 evaluation accepts validation only, never test")
    fields = {key: bundle[key] for key in ("image_features", "text_features", "image_available", "text_available")}
    validate_query_payload(fields)
    device = torch.device(args.device)
    encoder = MultimodalQueryEncoder(bundle["image_features"].shape[1], bundle["text_features"].shape[1]).to(device)
    keys = ExpertKeyStore([ExpertKeyMetadata(**item) for item in bundle["expert_metadata"]]).to(device)
    extra = load_router_checkpoint(args.checkpoint, encoder, keys, map_location=device)
    inputs = QueryInputs(*(fields[name].to(device) for name in ("image_features", "text_features", "image_available", "text_available")))
    predictions, margins, violations = [], [], 0
    encoder.eval()
    with torch.inference_mode():
        queries = encoder(inputs)
        all_keys = keys.normalized()
        for row, task_id in enumerate(bundle["task_ids"]):
            visible_ids = set(keys.visible_expert_ids(int(task_id), historical_only=True))
            visible = torch.tensor([value in visible_ids for value in keys.expert_ids], device=device)
            result = retrieve_experts(queries[row:row + 1], all_keys, keys.expert_ids, visible, top_k=min(8, max(1, len(visible_ids))))
            sims = result.similarities[0].tolist()
            ids = result.expert_ids[0].tolist()
            predictions.append(ids if sims and sims[0] >= float(extra["config"]["empty_margin"]) else [])
            margins.append(sims[0] - sims[1] if len(sims) > 1 else (sims[0] if sims else 0.0))
            violations += sum(keys.metadata[value].creation_task >= int(task_id) for value in predictions[-1])
    metrics = retrieval_metrics(predictions, bundle["oracle_sets"])
    per_task_top1 = {}
    for task_id in sorted(set(map(int, bundle["task_ids"]))):
        values = [row[0] for row, value in zip(predictions, bundle["task_ids"]) if int(value) == task_id and row]
        per_task_top1[str(task_id)] = {str(expert_id): values.count(expert_id) for expert_id in sorted(set(values))}
    metrics.update({
        "average_similarity_margin": sum(margins) / len(margins) if margins else 0.0,
        "temporal_mask_violations": violations,
        "key_utilization": len(set(value for row in predictions for value in row)) / max(1, len(keys.expert_ids)),
        "one_key_collapse": len(set(value for row in predictions for value in row)) <= 1 and len(keys.expert_ids) > 1,
        "task_ID_collapse_diagnostic": per_task_top1,
        "config_hash": extra["config_hash"], "checkpoint": args.checkpoint,
    })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
