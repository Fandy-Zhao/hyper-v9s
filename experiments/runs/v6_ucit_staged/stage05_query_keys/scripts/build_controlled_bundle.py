#!/usr/bin/env python3
"""Create deterministic answer-free controlled Query/Key bundles for L0 smoke."""

import hashlib
from pathlib import Path

import torch


def build(split, per_task):
    generator = torch.Generator().manual_seed(42 if split == "train" else 43)
    sample_ids, task_ids, oracle_sets, post_sets = [], [], [], []
    image, text = [], []
    for task_id in range(3):
        for index in range(per_task):
            sample_ids.append(f"controlled:{split}:{task_id}:{index}"); task_ids.append(task_id)
            if task_id == 0: target = ()
            elif task_id == 1: target = (0,) if index % 3 else ()
            else: target = (0, 1) if index % 2 else (1,)
            oracle_sets.append(target); post_sets.append(tuple(sorted(set(target) | {task_id})))
            base = torch.randn(8, generator=generator) * 0.05; base[task_id] += 1.0
            image.append(base); text.append(base + torch.randn(8, generator=generator) * 0.02)
    return {"schema_version": 1, "test_data_used": False, "sample_ids": sample_ids, "task_ids": task_ids,
            "task_names": [f"T{value}" for value in task_ids], "splits": [split] * len(sample_ids),
            "image_features": torch.stack(image), "text_features": torch.stack(text), "image_available": torch.ones(len(sample_ids)),
            "text_available": torch.ones(len(sample_ids)), "oracle_sets": oracle_sets, "post_task_oracle_sets": post_sets,
            "expert_metadata": [{"expert_id": value, "creation_task": value, "checkpoint_sha256": f"h{value}", "key_version": 1,
                                 "archived": False, "initialization": "controlled"} for value in range(3)],
            "manifest_hash": hashlib.sha256((split + "manifest").encode()).hexdigest(),
            "oracle_cache_hash": hashlib.sha256((split + "oracle").encode()).hexdigest()}


target = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/controlled"); target.mkdir(parents=True, exist_ok=True)
torch.save(build("train", 8), target / "train.pt"); torch.save(build("validation", 4), target / "validation.pt")
print("CONTROLLED_BUNDLES_READY")
