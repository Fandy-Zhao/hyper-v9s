#!/usr/bin/env python3
"""Merge per-task frozen features and bind expert-key metadata to Oracle checkpoint hashes."""

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/features")
CACHE_ROOT = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/oracle_direct")
TASKS = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def combine(paths, split, metadata):
    parts = [torch.load(path, map_location="cpu") for path in paths]
    for part in parts:
        if part["split"] != split or part["test_data_used"] is not False:
            raise ValueError("feature split/test audit failed")
    sample_keys = [(part["task_id"], sample) for part in parts for sample in part["sample_ids"]]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("duplicate task/sample IDs in feature bundle")
    result = {
        "schema_version": 1, "test_data_used": False, "feature_source": "frozen_clip_image_and_question_only",
        "sample_ids": ["{}:{}".format(part["task_name"], sample) for part in parts for sample in part["sample_ids"]],
        "task_ids": [part["task_id"] for part in parts for _ in part["sample_ids"]],
        "task_names": [part["task_name"] for part in parts for _ in part["sample_ids"]],
        "splits": [split for part in parts for _ in part["sample_ids"]],
        "image_features": torch.cat([part["image_features"] for part in parts]),
        "text_features": torch.cat([part["text_features"] for part in parts]),
        "image_available": torch.cat([part["image_available"] for part in parts]),
        "text_available": torch.cat([part["text_available"] for part in parts]),
        "oracle_sets": [tuple(value) for part in parts for value in part["oracle_sets"]],
        "post_task_oracle_sets": [tuple(value) for part in parts for value in part["post_task_oracle_sets"]],
        "oracle_records": [value for part in parts for value in part["oracle_records"]],
        "prompt_hashes": [value for part in parts for value in part["prompt_hashes"]],
        "image_hashes": [value for part in parts for value in part["image_hashes"]],
        "expert_metadata": metadata,
    }
    manifests = [part["dataset_manifest_hash"] for part in parts]
    caches = [part["oracle_cache_hash"] for part in parts]
    result["manifest_hash"] = stable_hash(manifests)
    result["oracle_cache_hash"] = stable_hash(caches)
    return result


def main():
    metadata = []
    for task_id, task_name in enumerate(TASKS):
        cache = json.loads((CACHE_ROOT / "train_missing/post_task_diagnostic" / task_name / "direct.json").read_text(encoding="utf-8"))
        checkpoint_hash = cache["provenance"]["expert_checkpoint_hashes"][str(task_id)]
        metadata.append({"expert_id": task_id, "creation_task": task_id, "checkpoint_sha256": checkpoint_hash,
                         "key_version": 1, "archived": False, "initialization": "post_task_direct_mean_or_seed42_fallback"})
    train_paths = []
    validation_paths = []
    for task_name in TASKS:
        train_paths.extend((FEATURE_ROOT / "partial/train" / (task_name + "_reused.pt"), FEATURE_ROOT / "partial/train" / (task_name + "_missing.pt")))
        validation_paths.append(FEATURE_ROOT / "partial/validation" / (task_name + ".pt"))
    train = combine(train_paths, "train", metadata)
    validation = combine(validation_paths, "validation", metadata)
    if len(train["sample_ids"]) != 768 or len(validation["sample_ids"]) != 192:
        raise ValueError("formal Stage 05 scale mismatch")
    target = FEATURE_ROOT / "bundles"; target.mkdir(parents=True, exist_ok=True)
    torch.save(train, target / "train.pt"); torch.save(validation, target / "validation.pt")
    manifest = {"train_samples": 768, "validation_samples": 192, "test_data_used": False,
                "train_manifest_hash": train["manifest_hash"], "validation_manifest_hash": validation["manifest_hash"],
                "train_oracle_cache_hash": train["oracle_cache_hash"], "validation_oracle_cache_hash": validation["oracle_cache_hash"]}
    (ROOT / "manifests/feature_bundles.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
