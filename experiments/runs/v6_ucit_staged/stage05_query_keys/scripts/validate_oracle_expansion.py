#!/usr/bin/env python3
"""Audit Stage 05 Direct cache scale, split, chronology, finiteness, and disjoint IDs."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/oracle_direct")
OLD = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/cache/full_seed42")
TASKS = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]


def load(path, count, split, scope, task_id):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["sample_count"] != count or len(value["records"]) != count: raise ValueError("cache count mismatch: {}".format(path))
    provenance = value["provenance"]
    if provenance["split"] != split or provenance["composition_mode"] != "direct_sum": raise ValueError("cache split/mode mismatch")
    visible = sorted(map(int, provenance["expert_checkpoint_hashes"]))
    expected = list(range(task_id if scope == "historical_only" else task_id + 1))
    if visible != expected: raise ValueError("temporal expert boundary mismatch: {}".format(path))
    for row in value["records"]:
        if row["temporal_scope"] != scope or row["split"] != split: raise ValueError("record scope/split mismatch")
        numbers = [row["empty"]["mean_nll"], row["selected_mean_nll"], row["selected_score"]]
        if not all(math.isfinite(float(number)) for number in numbers): raise ValueError("non-finite Oracle record")
    return value


def main():
    manifest = json.loads((ROOT / "manifests/oracle_expansion.json").read_text(encoding="utf-8"))
    checks = []
    for task_id, task_name in enumerate(TASKS):
        info = manifest["tasks"][task_name]
        train_ids = set(info["reused_stage04"]["sample_ids"]) | set(info["train_missing"]["sample_ids"])
        validation_ids = set(info["validation"]["sample_ids"])
        if len(train_ids) != 128 or len(validation_ids) != 32 or train_ids & validation_ids: raise ValueError("sample scale/disjoint audit failed")
        for scope in ("historical_only", "post_task_diagnostic"):
            load(OLD / scope / task_name / "direct.json", 32, "train", scope, task_id)
            load(NEW / "train_missing" / scope / task_name / "direct.json", 96, "train", scope, task_id)
            load(NEW / "validation" / scope / task_name / "direct.json", 32, "validation", scope, task_id)
        checks.append({"task_id": task_id, "task_name": task_name, "train": 128, "validation": 32, "disjoint": True})
    result = {"status": "PASSED", "tasks": checks, "train_total": 768, "validation_total": 192,
              "historical_post_separate": True, "direct_only": True, "temporal_violations": 0, "test_data_used": False}
    target = ROOT / "validation/oracle_expansion.json"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
