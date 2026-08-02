#!/usr/bin/env python3
"""Build separate teacher and predicted train-only Residual Buffers."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from compose.expansion.buffer_sampler import stratified_residual_sample
from compose.expansion.checkpoint import load_sufficiency_checkpoint
from compose.expansion.residual_buffer import ResidualBuffer, ResidualRecord
from compose.expansion.sufficiency import teacher_sufficiency
from compose.expansion.sufficiency_head import SufficiencyHead
from compose.expansion.validation import validate_buffer_record


def atomic_json(path, payload):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def resume_or_write(buffer, path):
    target = Path(path)
    if not target.exists():
        buffer.write_shard(target, 0)
        return buffer, False
    restored, rank = ResidualBuffer.load_shard(target)
    def canonical(rows):
        values = []
        for row in rows:
            value = asdict(row); value.pop("creation_timestamp"); value.pop("source_git_commit")
            values.append(value)
        return values
    if rank != 0 or restored.mode != buffer.mode or canonical(restored.records) != canonical(buffer.records):
        raise ValueError("existing Residual Buffer shard does not match deterministic reconstruction")
    return restored, True


def answer_type(record):
    tokens = int(record.get("empty", {}).get("token_count", 0))
    if tokens <= 1: return "single_token"
    if tokens <= 4: return "short_phrase"
    return "long_answer"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--features", required=True); parser.add_argument("--sufficiency-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    bundle = torch.load(args.features, map_location="cpu")
    if bundle.get("test_data_used") is not False or any(value != "train" for value in bundle["splits"]): raise ValueError("Residual Buffer input must be train-only")
    device = torch.device(args.device); model = SufficiencyHead().to(device); extra = load_sufficiency_checkpoint(args.sufficiency_checkpoint, model, map_location=device)
    query = bundle["queries"].to(device); card = bundle["cardinality_logits"].to(device); top = bundle["top_similarities"].to(device)
    counts = bundle["visible_counts"].to(device).long(); positions = torch.arange(top.shape[1], device=device).unsqueeze(0); mask = positions < counts.unsqueeze(1)
    masked = top.masked_fill(~mask, float("-inf")); probs = torch.nan_to_num(torch.softmax(masked, dim=-1)); probs = torch.where(mask, probs, torch.zeros_like(probs))
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    first = torch.where(counts > 0, top[:, 0], torch.zeros_like(top[:, 0])); second = torch.where(counts > 1, top[:, 1], torch.zeros_like(top[:, 0])); margin = first - second
    with torch.inference_mode(): predicted = torch.sigmoid(model(query, card, top, margin, entropy, bundle["predicted_set_scores"].to(device), bundle["visible_counts"].to(device))).cpu().tolist()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(); now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    teacher_buffer, predicted_buffer = ResidualBuffer("teacher_buffer"), ResidualBuffer("predicted_buffer")
    creation_tasks = {int(item["expert_id"]): int(item["creation_task"]) for item in bundle["expert_metadata"]}
    all_teacher, all_predicted = [], []
    for index, record in enumerate(bundle["oracle_records"]):
        label = teacher_sufficiency(record); task_id = int(bundle["task_ids"][index]); task_name = bundle["task_names"][index]
        row = ResidualRecord(
            bundle["sample_ids"][index], task_id, task_name, "train", "feature_bundle:" + bundle["sample_ids"][index], bundle["manifest_hash"],
            "sha256:" + bundle["image_hashes"][index], "sha256:" + bundle["prompt_hashes"][index], "oracle-cache-reference:" + bundle["sample_ids"][index],
            hashlib.sha256(bundle["queries"][index].numpy().tobytes()).hexdigest(), tuple(bundle["predicted_sets"][index]), label.teacher_set,
            label.old_expert_sufficient, float(predicted[index]), float(record["empty"]["mean_nll"]), float(record["selected_mean_nll"]),
            label.residual_gain, float(bundle["router_confidences"][index]), now, commit, extra.get("config_hash", "stage07_fixed"),
            answer_type(record), "prompt_bucket_" + bundle["prompt_hashes"][index][:2],
        )
        validate_buffer_record(row, creation_tasks)
        if not label.old_expert_sufficient: all_teacher.append(row)
        if predicted[index] < float(extra["threshold"]): all_predicted.append(row)
    retained_ids, discarded_ids = {}, {}
    for task_id in sorted(set(bundle["task_ids"])):
        teacher_rows = [row for row in all_teacher if row.task_id == task_id]; predicted_rows = [row for row in all_predicted if row.task_id == task_id]
        for mode, rows, target in (("teacher_buffer", teacher_rows, teacher_buffer), ("predicted_buffer", predicted_rows, predicted_buffer)):
            retained, discarded = stratified_residual_sample(rows, 512, 42 + task_id); retained_ids[f"{mode}:{task_id}"] = [row.sample_id for row in retained]; discarded_ids[f"{mode}:{task_id}"] = [row.sample_id for row in discarded]
            for row in retained: target.add(row)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    teacher_buffer, teacher_resumed = resume_or_write(teacher_buffer, output / "teacher_buffer.json")
    predicted_buffer, predicted_resumed = resume_or_write(predicted_buffer, output / "predicted_buffer.json")
    atomic_json(output / "sampling_manifest.json", {"retained_ids": retained_ids, "discarded_ids": discarded_ids, "seed": 42})
    teacher_ids = {row.sample_id for row in teacher_buffer.records}; predicted_ids = {row.sample_id for row in predicted_buffer.records}
    task_metrics = {}
    for task_id in sorted(set(bundle["task_ids"])):
        task_rows = [index for index, value in enumerate(bundle["task_ids"]) if int(value) == int(task_id)]
        teacher_candidates = sum(not teacher_sufficiency(bundle["oracle_records"][index]).old_expert_sufficient for index in task_rows)
        predicted_candidates = sum(predicted[index] < float(extra["threshold"]) for index in task_rows)
        gains = [teacher_sufficiency(bundle["oracle_records"][index]).residual_gain for index in task_rows]
        task_metrics[str(task_id)] = {"task_name": bundle["task_names"][task_rows[0]], "samples": len(task_rows),
                                      "teacher_insufficient_rate": teacher_candidates / len(task_rows),
                                      "predicted_insufficient_rate": predicted_candidates / len(task_rows),
                                      "mean_residual_gain": sum(gains) / len(gains)}
    by_cardinality = {}
    for cardinality in range(3):
        indices = [index for index, value in enumerate(bundle["predicted_sets"]) if len(value) == cardinality]
        by_cardinality[str(cardinality)] = {"samples": len(indices),
            "teacher_insufficient_rate": sum(not teacher_sufficiency(bundle["oracle_records"][index]).old_expert_sufficient for index in indices) / max(1, len(indices)),
            "predicted_insufficient_rate": sum(predicted[index] < float(extra["threshold"]) for index in indices) / max(1, len(indices))}
    route_relation = {}
    for mismatch in (False, True):
        indices = [index for index, (predicted_set, teacher_set) in enumerate(zip(bundle["predicted_sets"], bundle["oracle_sets"]))
                   if (set(predicted_set) != set(teacher_set)) == mismatch]
        route_relation[str(mismatch).lower()] = {"samples": len(indices),
            "mean_residual_gain": sum(teacher_sufficiency(bundle["oracle_records"][index]).residual_gain for index in indices) / max(1, len(indices)),
            "teacher_insufficient_rate": sum(not teacher_sufficiency(bundle["oracle_records"][index]).old_expert_sufficient for index in indices) / max(1, len(indices))}
    def distribution(rows, field):
        values = {}
        for row in rows: values[getattr(row, field)] = values.get(getattr(row, field), 0) + 1
        return dict(sorted(values.items()))
    report = {"teacher_buffer_size": len(teacher_buffer.records), "predicted_buffer_size": len(predicted_buffer.records),
              "teacher_predicted_jaccard": len(teacher_ids & predicted_ids) / max(1, len(teacher_ids | predicted_ids)),
              "per_task": task_metrics, "insufficiency_by_predicted_cardinality": by_cardinality,
              "routing_error_residual_relation": route_relation,
              "buffer_distribution": {"teacher_answer_type": distribution(teacher_buffer.records, "answer_type"),
                                      "predicted_answer_type": distribution(predicted_buffer.records, "answer_type"),
                                      "teacher_question_subtype": distribution(teacher_buffer.records, "question_subtype"),
                                      "predicted_question_subtype": distribution(predicted_buffer.records, "question_subtype")},
              "future_violation_count": 0, "test_data_used": False,
              "teacher_and_predicted_buffers_separate": True,
              "resume": {"teacher_shard_resumed": teacher_resumed, "predicted_shard_resumed": predicted_resumed}}
    atomic_json(output / "metrics.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
