#!/usr/bin/env python3
"""Stage 01: build deterministic mini2 data indices (ImageNet-R, ArxivQA).

Creates small JSON instruction files only -- no images copied, no original
data modified. The same mini2 indices are used by BOTH baseline configs
(original_batch and gb24_matched).

Splits per task:
  train: 512 (sampled with fixed seed from the task's official train file)
  val:   128 (sampled from remaining official train entries, fixed seed)
  test:  128 (sampled with fixed seed from the official test_3000 file)
Sample ids are recorded in a manifest so the split is fully reproducible.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
UCIT = Path("/data/dataset/zhaozhuofan/UCIT/instructions")
OUT = REPO / "experiments/runs/v6_ucit_staged/stage01_baseline/data_indices"
SEED = 42
N_TRAIN = 512
N_VAL = 128
N_TEST = 128

TASKS = [
    {"name": "ImageNet-R", "train_file": "ImageNet-R/train.json", "test_file": "ImageNet-R/test_3000.json"},
    {"name": "ArxivQA", "train_file": "ArxivQA/train_4w.json", "test_file": "ArxivQA/test_3000.json"},
]


def sample_ids(entries, n, rng, tag, id_field="id"):
    pool = [e[id_field] for e in entries]
    assert len(pool) >= n, f"{tag}: need {n} ids, have {len(pool)}"
    ids = rng.sample(pool, n)
    by_id = {e[id_field]: e for e in entries}
    return [by_id[i] for i in ids], ids


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": SEED, "tasks": {}}
    for task in TASKS:
        name = task["name"]
        train_path = UCIT / task["train_file"]
        test_path = UCIT / task["test_file"]
        train_all = json.loads(train_path.read_text(encoding="utf-8"))
        test_all = json.loads(test_path.read_text(encoding="utf-8"))

        rng = random.Random(SEED)
        train_entries, train_ids = sample_ids(train_all, N_TRAIN, rng, f"{name}/train")
        remaining = [e for e in train_all if e["id"] not in set(train_ids)]
        val_entries, val_ids = sample_ids(remaining, N_VAL, rng, f"{name}/val")
        test_entries, test_ids = sample_ids(test_all, N_TEST, rng, f"{name}/test", id_field="question_id")

        # ensure non-overlap between train/val ids
        assert not set(train_ids) & set(val_ids), f"{name}: train/val overlap"

        task_dir = OUT / name
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, entries in (("train", train_entries), ("val", val_entries), ("test", test_entries)):
            target = task_dir / f"{split}.json"
            target.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest["tasks"][name] = {
            "train_source": str(train_path),
            "train_source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "train_count": len(train_entries),
            "train_ids": train_ids,
            "val_count": len(val_entries),
            "val_ids": val_ids,
            "test_source": str(test_path),
            "test_source_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            "test_count": len(test_entries),
            "test_ids": test_ids,
        }
        print(f"{name}: train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}")
    (OUT / "mini2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote mini2 manifest to {OUT / 'mini2_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
