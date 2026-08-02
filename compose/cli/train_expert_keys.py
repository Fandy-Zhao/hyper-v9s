#!/usr/bin/env python3
"""Train answer-free Query/Keys from frozen feature bundles and Direct Oracle labels."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from compose.router.anchor_memory import AnchorMemory, AnchorRecord
from compose.router.checkpoint import save_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.losses import RetrievalLossConfig, multi_positive_retrieval_loss, retrieval_objective
from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs
from compose.router.validation import validate_query_payload, validate_temporal_targets


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu")
    if bundle.get("schema_version") != 1 or bundle.get("test_data_used") is not False:
        raise ValueError("feature bundle schema/test audit failed")
    feature_fields = {key: bundle[key] for key in ("image_features", "text_features", "image_available", "text_available")}
    validate_query_payload(feature_fields)
    if any(split != "train" for split in bundle["splits"]):
        raise ValueError("training bundle contains non-train samples")
    return bundle, feature_fields


def masks(bundle, expert_ids, device):
    positives = torch.zeros(len(bundle["sample_ids"]), len(expert_ids), dtype=torch.bool, device=device)
    visible = torch.zeros_like(positives)
    index = {value: offset for offset, value in enumerate(expert_ids)}
    creation = {int(item["expert_id"]): int(item["creation_task"]) for item in bundle["expert_metadata"]}
    validate_temporal_targets(bundle["oracle_sets"], creation, bundle["task_ids"])
    for row, (task_id, targets) in enumerate(zip(bundle["task_ids"], bundle["oracle_sets"])):
        for expert_id in expert_ids:
            visible[row, index[expert_id]] = creation[expert_id] < int(task_id)
        for expert_id in targets:
            positives[row, index[int(expert_id)]] = True
    if torch.any(positives & ~visible):
        raise ValueError("Oracle positives violate historical-only visibility")
    return positives, visible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("offline_diagnostic", "continual_anchor"), default="continual_anchor")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bundle, fields = load_bundle(args.features)
    device = torch.device(args.device)
    metadata = [ExpertKeyMetadata(**item) for item in bundle["expert_metadata"]]
    encoder = MultimodalQueryEncoder(bundle["image_features"].shape[1], bundle["text_features"].shape[1]).to(device)
    keys = ExpertKeyStore(metadata, seed=args.seed).to(device)
    inputs = QueryInputs(*(fields[name].to(device) for name in ("image_features", "text_features", "image_available", "text_available")))
    positive_mask, visible_mask = masks(bundle, keys.expert_ids, device)
    config = RetrievalLossConfig()
    frozen_config = {
        "mode": args.mode, "query_dim": 128, "seed": args.seed, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "temperature": config.temperature,
        "empty_margin": config.empty_margin, "lambda_empty": config.lambda_empty,
        "lambda_diversity": config.lambda_diversity, "lambda_anchor": config.lambda_anchor,
        "oracle_source": "historical_only_direct", "test_data_used": False,
    }
    config_hash = stable_hash(frozen_config)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(keys.parameters()), lr=args.learning_rate)
    with torch.no_grad():
        initial_queries = encoder(inputs)
    initialization = {}
    for expert_id in keys.expert_ids:
        creation_task = keys.metadata[expert_id].creation_task
        rows = [index for index, selected_set in enumerate(bundle["post_task_oracle_sets"])
                if int(bundle["task_ids"][index]) == creation_task and expert_id in selected_set]
        initialized = keys.initialize_from_queries(expert_id, initial_queries[rows] if rows else initial_queries[:0], minimum_positives=2)
        initialization[str(expert_id)] = {"method": "post_task_direct_mean" if initialized else "seed42_random_fallback", "positive_samples": len(rows)}
    task_order = sorted(set(map(int, bundle["task_ids"]))) if args.mode == "continual_anchor" else [None]
    anchors = AnchorMemory(32)
    history = []
    for current_task in task_order:
        selected = torch.ones(len(bundle["sample_ids"]), dtype=torch.bool, device=device)
        if current_task is not None:
            selected = torch.tensor([int(value) == current_task for value in bundle["task_ids"]], device=device)
        replay = anchors.records(int(current_task) - 1) if current_task is not None else ()
        for epoch in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            query = encoder(inputs)
            similarities = query @ keys.normalized().T
            anchor_loss = similarities.sum() * 0.0
            if replay:
                anchor_queries = torch.tensor([row.query for row in replay], device=device, dtype=similarities.dtype)
                anchor_positive = torch.zeros(len(replay), len(keys.expert_ids), dtype=torch.bool, device=device)
                key_index = {value: index for index, value in enumerate(keys.expert_ids)}
                for row, record in enumerate(replay): anchor_positive[row, key_index[record.expert_id]] = True
                anchor_visible = torch.tensor([[keys.metadata[value].creation_task < int(current_task) for value in keys.expert_ids]
                                               for _ in replay], device=device)
                anchor_loss = multi_positive_retrieval_loss(anchor_queries @ keys.normalized().T, anchor_positive, anchor_visible, config.temperature)
            total, parts = retrieval_objective(similarities[selected], positive_mask[selected], visible_mask[selected],
                                                keys.normalized(), config, anchor_loss=anchor_loss)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite Query/Key loss")
            total.backward()
            optimizer.step()
        history.append({"task_id": current_task, "loss": float(total.detach()), **{name: float(value.detach()) for name, value in parts.items()}})
        if current_task is not None:
            with torch.no_grad():
                frozen_queries = encoder(inputs).cpu()
            for row in torch.where(selected)[0].tolist():
                for expert_id in bundle["oracle_sets"][row]:
                    vector = frozen_queries[row]
                    anchors.add(AnchorRecord(
                        str(bundle["sample_ids"][row]), int(expert_id), int(current_task), "train",
                        hashlib.sha256(vector.numpy().tobytes()).hexdigest(), tuple(map(float, vector.tolist())),
                    ), current_task)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_router_checkpoint(output, encoder, keys, optimizer, {
        "config": frozen_config, "config_hash": config_hash, "history": history,
        "anchors": anchors.state_dict(), "feature_manifest_hash": bundle["manifest_hash"],
        "oracle_cache_hash": bundle["oracle_cache_hash"], "key_initialization": initialization,
    })
    print(json.dumps({"status": "TRAINED", "checkpoint": str(output), "config_hash": config_hash, "history": history}, sort_keys=True))


if __name__ == "__main__":
    main()
