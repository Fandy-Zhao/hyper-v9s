#!/usr/bin/env python3
"""Create preregistered, train-only Stage 04 sample subsets."""

import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from compose.data.records import answer_text


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/dataset/zhaozhuofan/v6_ucit_stage04")
MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
TASKS = [
    (0, "ImageNet-R", "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/train.json"),
    (1, "ArxivQA", "/data/dataset/zhaozhuofan/UCIT/instructions/ArxivQA/train_4w.json"),
    (2, "VizWiz", "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/train.json"),
    (3, "IconQA", "/data/dataset/zhaozhuofan/UCIT/instructions/IconQA/train.json"),
    (4, "CLEVR", "/data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/train_4w.json"),
    (5, "Flickr30k", "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/train_brief_4w.json"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_id(row, fallback):
    return str(row.get("question_id", row.get("id", fallback)))


def stratified(rows, count, tokenizer, seed, excluded=()):
    excluded = set(excluded)
    candidates = []
    for index, row in enumerate(rows):
        identifier = record_id(row, index)
        if identifier in excluded:
            continue
        answer = answer_text(row)
        length = len(tokenizer.encode(answer, add_special_tokens=False))
        if length <= 0:
            continue
        candidates.append((length, identifier, row))
    candidates.sort(key=lambda value: (value[0], value[1]))
    strata = []
    for quartile in range(4):
        start = quartile * len(candidates) // 4
        end = (quartile + 1) * len(candidates) // 4
        values = candidates[start:end]
        random.Random(seed + quartile).shuffle(values)
        strata.append(values)
    selected = []
    while len(selected) < count and any(strata):
        for values in strata:
            if values and len(selected) < count:
                selected.append(values.pop())
    if len(selected) != count:
        raise ValueError("not enough valid train samples: wanted {}, got {}".format(count, len(selected)))
    return [value[2] for value in selected], [value[1] for value in selected]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256(path)


tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
manifest = {"sampling_seed": 42, "test_data_used": False, "stratification": "answer_token_length_quartile_round_robin", "tasks": {}}
for task_id, task_name, source in TASKS:
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    full, full_ids = stratified(rows, 32, tokenizer, 42 + task_id * 100)
    full_path = DATA_ROOT / "full_seed42" / (task_name + ".json")
    mini_path = DATA_ROOT / "mini2" / (task_name + ".json")
    task_manifest = {
        "task_id": task_id, "task_name": task_name, "source": source,
        "source_sha256": sha256(source), "source_sample_count": len(rows),
        "full_subset": {"path": str(full_path), "sample_count": 32, "sample_ids": full_ids,
                        "sha256": write(full_path, full)},
    }
    if task_id < 2:
        task_manifest["mini2"] = {"path": str(mini_path), "sample_count": 32,
                                  "sample_ids": full_ids, "sha256": write(mini_path, full)}
    if task_id == 0:
        smoke_train, train_ids = stratified(rows, 64, tokenizer, 1042)
        smoke_val, val_ids = stratified(rows, 32, tokenizer, 2042, excluded=train_ids)
        train_path = DATA_ROOT / "smoke" / "ImageNet-R_train.json"
        val_path = DATA_ROOT / "smoke" / "ImageNet-R_validation.json"
        task_manifest["smoke"] = {
            "train": {"path": str(train_path), "sample_count": 64, "sample_ids": train_ids, "sha256": write(train_path, smoke_train)},
            "validation": {"path": str(val_path), "sample_count": 32, "sample_ids": val_ids, "sha256": write(val_path, smoke_val)},
            "disjoint": not bool(set(train_ids) & set(val_ids)),
        }
    manifest["tasks"][task_name] = task_manifest
controlled = {
    "A_plus_B": "experiments/data/controlled_format_v1_training/instructions/A_plus_B/train_eval.json",
    "B_plus_C": "experiments/data/controlled_format_v1_training/instructions/B_plus_C/train_eval.json",
}
manifest["controlled"] = {}
for name, source in controlled.items():
    rows = json.loads(Path(source).read_text(encoding="utf-8"))[:16]
    target = DATA_ROOT / "controlled" / (name + ".json")
    manifest["controlled"][name] = {"source": source, "source_sha256": sha256(source),
                                     "path": str(target), "sample_count": len(rows),
                                     "sample_ids": [record_id(row, index) for index, row in enumerate(rows)],
                                     "sha256": write(target, rows), "split": "train_calibration"}
manifest_path = ROOT / "manifests" / "sample_subsets.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "PREPARED", "tasks": len(TASKS), "test_data_used": False}, sort_keys=True))
