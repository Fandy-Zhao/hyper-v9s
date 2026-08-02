#!/usr/bin/env python3
"""Train on historical Direct teacher labels and freeze a validation threshold."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from compose.expansion.checkpoint import save_sufficiency_checkpoint
from compose.expansion.metrics import sufficiency_metrics
from compose.expansion.sufficiency import teacher_sufficiency
from compose.expansion.sufficiency_head import SufficiencyHead


def temporal_prefix(bundle, max_task_id):
    if max_task_id is None:
        return bundle
    indices = [index for index, value in enumerate(bundle["task_ids"]) if int(value) <= int(max_task_id)]
    if not indices:
        raise ValueError("temporal prefix contains no samples")
    total = len(bundle["task_ids"])
    result = {}
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    for key, value in bundle.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == total:
            result[key] = value.index_select(0, tensor_indices)
        elif isinstance(value, (list, tuple)) and len(value) == total:
            result[key] = [value[index] for index in indices]
        else:
            result[key] = value
    return result


def tensors(bundle, device):
    query = bundle["queries"].to(device); cardinality = bundle["cardinality_logits"].to(device); top = bundle["top_similarities"].to(device)
    counts = bundle["visible_counts"].to(device).long()
    positions = torch.arange(top.shape[1], device=device).unsqueeze(0)
    mask = positions < counts.unsqueeze(1)
    masked = top.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked, dim=-1)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs = torch.nan_to_num(probs)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    first = torch.where(counts > 0, top[:, 0], torch.zeros_like(top[:, 0]))
    second = torch.where(counts > 1, top[:, 1], torch.zeros_like(top[:, 0]))
    margin = first - second
    return query, cardinality, top, margin, entropy, bundle["predicted_set_scores"].to(device), bundle["visible_counts"].to(device)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--train", required=True); parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--metrics-output"); parser.add_argument("--max-task-id", type=int); parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); device = torch.device(args.device); torch.manual_seed(42)
    if device.type == "cuda": torch.cuda.manual_seed_all(42)
    train, validation = torch.load(args.train, map_location="cpu"), torch.load(args.validation, map_location="cpu")
    train, validation = temporal_prefix(train, args.max_task_id), temporal_prefix(validation, args.max_task_id)
    if train.get("test_data_used") is not False or validation.get("test_data_used") is not False: raise ValueError("test leakage")
    if any(value != "train" for value in train["splits"]) or any(value != "validation" for value in validation["splits"]): raise ValueError("split leakage")
    train_labels = torch.tensor([teacher_sufficiency(row).old_expert_sufficient for row in train["oracle_records"]], device=device, dtype=torch.float32)
    val_labels = [teacher_sufficiency(row).old_expert_sufficient for row in validation["oracle_records"]]
    model = SufficiencyHead().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_inputs = tensors(train, device)
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True); logits = model(*train_inputs); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite sufficiency loss")
        loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode(): probabilities = torch.sigmoid(model(*tensors(validation, device))).cpu().tolist()
    candidates = {str(value): sufficiency_metrics(val_labels, probabilities, value) for value in (0.3, 0.5, 0.7)}
    threshold = min((0.3, 0.5, 0.7), key=lambda value: (candidates[str(value)]["false_sufficient_rate"], -candidates[str(value)]["f1"], value))
    frozen_config = {"seed": 42, "epochs": args.epochs, "learning_rate": 0.001, "threshold_candidates": [0.3, 0.5, 0.7],
                     "teacher_gain_threshold": 0.02, "max_task_id": args.max_task_id, "test_data_used": False}
    config_hash = hashlib.sha256(json.dumps(frozen_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    per_task = {}
    for task_name in sorted(set(validation["task_names"])):
        indices = [index for index, value in enumerate(validation["task_names"]) if value == task_name]
        labels = [val_labels[index] for index in indices]; scores = [probabilities[index] for index in indices]
        per_task[task_name] = {**sufficiency_metrics(labels, scores, threshold),
                               "teacher_sufficient_rate": sum(labels) / len(labels),
                               "mean_predicted_probability": sum(scores) / len(scores), "samples": len(indices)}
    extra = {"threshold": threshold, "threshold_candidates": candidates, "metrics": candidates[str(threshold)], "per_task_calibration": per_task,
             "teacher_insufficient_rate": 1.0 - sum(val_labels) / len(val_labels), "max_task_id": args.max_task_id,
             "predicted_insufficient_rate": sum(value < threshold for value in probabilities) / len(probabilities), "test_data_used": False,
             "config": frozen_config, "config_hash": config_hash,
             "teacher_definition": "nonempty_and_selected_score_better_than_empty_and_raw_nll_gain_ge_0.02"}
    save_sufficiency_checkpoint(args.output, model, optimizer, extra)
    if args.metrics_output:
        metrics_target = Path(args.metrics_output); metrics_target.parent.mkdir(parents=True, exist_ok=True)
        metrics_target.write_text(json.dumps({"status": "VALIDATED_THRESHOLD_FROZEN", **extra}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALIDATED_THRESHOLD_FROZEN", **extra}, sort_keys=True))


if __name__ == "__main__": main()
