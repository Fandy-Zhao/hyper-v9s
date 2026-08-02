#!/usr/bin/env python3
"""Train the answer-free 0/1/2 Set Router on historical-only Direct labels."""

import argparse
import hashlib
import json

import torch

from compose.router.checkpoint import load_router_checkpoint, save_set_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs
from compose.router.set_losses import SetLossConfig, set_router_loss
from compose.router.set_router import ExpertSetRouter


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def continual_anchor_indices(bundle, current_task, max_per_expert=32, seed=42):
    """Return current rows plus deterministic train-only historical member anchors."""
    current = [index for index, task in enumerate(bundle["task_ids"]) if int(task) == int(current_task)]
    historical = [index for index, task in enumerate(bundle["task_ids"]) if int(task) < int(current_task)]
    expert_ids = [int(item["expert_id"]) for item in bundle["expert_metadata"]
                  if int(item["creation_task"]) < int(current_task) and not item.get("archived", False)]
    selected_anchors = set()
    member_counts = {expert_id: 0 for expert_id in expert_ids}
    for expert_id in expert_ids:
        candidates = [index for index in historical if expert_id in set(map(int, bundle["oracle_sets"][index]))]
        candidates.sort(key=lambda index: hashlib.sha256(
            f"{seed}:{expert_id}:{bundle['sample_ids'][index]}".encode()).hexdigest())
        for index in candidates:
            if member_counts[expert_id] >= int(max_per_expert):
                break
            if index in selected_anchors:
                continue
            members = set(map(int, bundle["oracle_sets"][index])) & set(expert_ids)
            if any(member_counts[member] >= int(max_per_expert) for member in members):
                continue
            selected_anchors.add(index)
            for member in members:
                member_counts[member] += 1
    anchors = sorted(selected_anchors)
    anchors_by_expert = {str(expert_id): [index for index in anchors
                                          if expert_id in set(map(int, bundle["oracle_sets"][index]))]
                         for expert_id in expert_ids}
    return current + anchors, anchors_by_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--query-key-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("offline_diagnostic", "continual_anchor"), default="continual_anchor")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    bundle = torch.load(args.features, map_location="cpu")
    if bundle.get("test_data_used") is not False or any(value != "train" for value in bundle["splits"]):
        raise ValueError("Set Router training requires train-only bundle")
    device = torch.device(args.device)
    encoder = MultimodalQueryEncoder(bundle["image_features"].shape[1], bundle["text_features"].shape[1]).to(device)
    keys = ExpertKeyStore([ExpertKeyMetadata(**item) for item in bundle["expert_metadata"]]).to(device)
    key_extra = load_router_checkpoint(args.query_key_checkpoint, encoder, keys, map_location=device)
    encoder.eval(); keys.eval(); encoder.requires_grad_(False); keys.requires_grad_(False)
    inputs = QueryInputs(*(bundle[name].to(device) for name in ("image_features", "text_features", "image_available", "text_available")))
    with torch.inference_mode(): queries = encoder(inputs); key_values = keys.normalized()
    visible = torch.tensor([[keys.metadata[value].creation_task < int(task) for value in keys.expert_ids] for task in bundle["task_ids"]], device=device)
    router = ExpertSetRouter(top_m=8).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate)
    loss_config = SetLossConfig()
    frozen = {"mode": args.mode, "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "lambda_member": 1.0, "lambda_pair": 1.0, "lambda_retrieval": 0.5, "lambda_empty": 1.0,
              "pair_threshold_candidates": [0.50, 0.65, 0.80], "composition_mode": "direct_sum",
              "anchor_max_per_expert": 32, "anchor_selection": "seeded_sha256_member_stratified_train_only",
              "oracle_source": "historical_only_direct", "test_data_used": False}
    config_hash = stable_hash(frozen)
    tasks = sorted(set(bundle["task_ids"])) if args.mode == "continual_anchor" else [None]
    history = []
    for current_task in tasks:
        if current_task is None:
            indices = list(range(len(bundle["sample_ids"])))
            anchors_by_expert = {}
        else:
            indices, anchors_by_expert = continual_anchor_indices(bundle, current_task, 32, args.seed)
        selected = torch.tensor(indices, dtype=torch.long, device=device)
        for _ in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            output = router(queries[selected], key_values, keys.expert_ids, visible[selected])
            targets = [bundle["oracle_sets"][index] for index in indices]
            total, parts = set_router_loss(output, targets, keys.expert_ids, loss_config)
            if not torch.isfinite(total): raise FloatingPointError("non-finite Set Router loss")
            total.backward(); optimizer.step()
        anchor_indices = {value for rows in anchors_by_expert.values() for value in rows}
        history.append({"task_id": current_task, "loss": float(total.detach()),
                        "current_samples": len(indices) - len(anchor_indices),
                        "anchor_samples": len(anchor_indices),
                        "anchor_ids_by_expert": {expert: [bundle["sample_ids"][index] for index in rows]
                                                 for expert, rows in anchors_by_expert.items()},
                        **{name: float(value.detach()) for name, value in parts.items()}})
    save_set_router_checkpoint(args.output, router, {"config": frozen, "config_hash": config_hash, "history": history,
                                                     "query_key_config_hash": key_extra["config_hash"], "pair_threshold": None})
    print(json.dumps({"status": "TRAINED", "config_hash": config_hash, "history": history}, sort_keys=True))


if __name__ == "__main__": main()
