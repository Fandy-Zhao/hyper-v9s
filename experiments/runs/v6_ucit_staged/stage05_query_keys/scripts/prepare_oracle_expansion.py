#!/usr/bin/env python3
"""Freeze disjoint Stage 05 Direct-Oracle subsets while reusing Stage 04's 32 rows/task."""

import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from compose.data.records import answer_text


ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST = ROOT.parent / "stage04_oracle_teacher/manifests/sample_subsets.json"
DATA_ROOT = Path("/data/dataset/zhaozhuofan/v6_ucit_stage05")
MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record_id(row, fallback):
    return str(row.get("question_id", row.get("id", fallback)))


def stratified(rows, count, tokenizer, seed, excluded):
    candidates = []
    for index, row in enumerate(rows):
        identifier = record_id(row, index)
        if identifier in excluded:
            continue
        length = len(tokenizer.encode(answer_text(row), add_special_tokens=False))
        if length:
            candidates.append((length, identifier, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    strata = []
    for quartile in range(4):
        part = candidates[quartile * len(candidates) // 4:(quartile + 1) * len(candidates) // 4]
        random.Random(seed + quartile).shuffle(part)
        strata.append(part)
    selected = []
    while len(selected) < count and any(strata):
        for part in strata:
            if part and len(selected) < count:
                selected.append(part.pop())
    if len(selected) != count:
        raise ValueError("insufficient valid samples")
    return [item[2] for item in selected], [item[1] for item in selected]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return digest(path)


def main():
    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    manifest = {
        "schema_version": 1, "seed": 42, "stratification": "answer_token_length_quartile_round_robin",
        "train_per_task": 128, "validation_per_task": 32, "reused_per_task": 32,
        "oracle_source": "direct_only", "test_data_used": False, "tasks": {},
    }
    for task_name, previous in sorted(old["tasks"].items(), key=lambda item: item[1]["task_id"]):
        source = Path(previous["source"])
        if digest(source) != previous["source_sha256"]:
            raise ValueError("source hash changed for {}".format(task_name))
        rows = json.loads(source.read_text(encoding="utf-8"))
        reused_ids = list(map(str, previous["full_subset"]["sample_ids"]))
        missing_rows, missing_ids = stratified(rows, 96, tokenizer, 4200 + previous["task_id"] * 100, set(reused_ids))
        validation_rows, validation_ids = stratified(rows, 32, tokenizer, 8400 + previous["task_id"] * 100, set(reused_ids) | set(missing_ids))
        missing_path = DATA_ROOT / "train_missing" / (task_name + ".json")
        validation_path = DATA_ROOT / "validation" / (task_name + ".json")
        if set(reused_ids) & set(missing_ids) or (set(reused_ids) | set(missing_ids)) & set(validation_ids):
            raise AssertionError("train/validation overlap")
        manifest["tasks"][task_name] = {
            "task_id": previous["task_id"], "source": str(source), "source_sha256": previous["source_sha256"],
            "reused_stage04": previous["full_subset"],
            "train_missing": {"path": str(missing_path), "sample_count": 96, "sample_ids": missing_ids, "sha256": write(missing_path, missing_rows)},
            "validation": {"path": str(validation_path), "sample_count": 32, "sample_ids": validation_ids, "sha256": write(validation_path, validation_rows)},
            "disjoint": True,
        }
    target = ROOT / "manifests/oracle_expansion.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED", "train": 128 * 6, "validation": 32 * 6, "test_data_used": False}))


if __name__ == "__main__":
    main()
