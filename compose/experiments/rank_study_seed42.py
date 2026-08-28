"""Prepare and validate the controlled seed42 Compose rank study.

This module never mutates the formal seed42 run.  It materializes frozen
pre-S6 boundaries in the independent rank-study root and can convert a
historical rank-8 pool to a behavior-equivalent higher-rank representation by
zero-padding LoRA factors while keeping alpha/rank fixed.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch
import yaml

from compose.experts.registry import ExpertRegistry


REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
FORMAL = REPO / "experiments/runs/compose_ucit_formal_seed42"
STUDY = REPO / "experiments/runs/compose_ucit_rank_study_seed42"
PRE_STAGES = (
    "s0_snapshot_load", "s1_features", "s2_teacher", "s3_residual",
    "s4_recall_audit", "s5_clustering", "s5_advance",
)
TASKS = {0: "ImageNet-R", 1: "ArxivQA", 4: "CLEVR"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(values):
    data = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def refuse_nonempty(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("refusing non-empty run root: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)


def upcast_pool(source, target, rank):
    source, target = Path(source), Path(target)
    if rank == 8:
        return source
    if target.exists():
        report = read_json(target / "equivalence_report.json")
        if report.get("verified") and report.get("target_rank") == rank:
            return target
        raise FileExistsError("unverified upcast target exists: {}".format(target))
    target.mkdir(parents=True)
    manifest = read_json(source / "compose_experts.json")
    old_rank = int(manifest["adapter"]["rank"])
    old_alpha = float(manifest["adapter"]["alpha"])
    if old_rank != 8 or abs(old_alpha / old_rank - 2.0) > 1e-12:
        raise ValueError("unexpected source scaling rank={} alpha={}".format(old_rank, old_alpha))
    state = torch.load(source / "compose_experts.bin", map_location="cpu")
    converted = {}
    max_restore_diff = 0.0
    padded_nonzero = 0
    for key, tensor in state.items():
        if key.endswith("lora_A.weight"):
            shape = (rank, tensor.shape[1])
            out = torch.zeros(shape, dtype=tensor.dtype)
            out[:old_rank] = tensor
            max_restore_diff = max(max_restore_diff, float((out[:old_rank] - tensor).abs().max()))
            padded_nonzero += int(torch.count_nonzero(out[old_rank:]))
        elif key.endswith("lora_B.weight"):
            shape = (tensor.shape[0], rank)
            out = torch.zeros(shape, dtype=tensor.dtype)
            out[:, :old_rank] = tensor
            max_restore_diff = max(max_restore_diff, float((out[:, :old_rank] - tensor).abs().max()))
            padded_nonzero += int(torch.count_nonzero(out[:, old_rank:]))
        else:
            raise ValueError("unexpected checkpoint tensor: {}".format(key))
        converted[key] = out
    torch.save(converted, target / "compose_experts.bin")
    manifest["adapter"]["rank"] = rank
    manifest["adapter"]["alpha"] = float(2 * rank)
    for expert in manifest.get("experts", []):
        if "rank" in expert:
            expert["rank"] = rank
        if "lora_alpha" in expert:
            expert["lora_alpha"] = float(2 * rank)
        if "alpha" in expert:
            expert["alpha"] = float(2 * rank)
    manifest["metrics"]["adapter_parameter_count"] = sum(v.numel() for v in converted.values())
    manifest["metrics"]["adapter_tensor_count"] = len(converted)
    manifest["metrics"]["checkpoint_bytes"] = (target / "compose_experts.bin").stat().st_size
    write_json(target / "compose_experts.json", manifest)
    report = {
        "source": str(source), "target": str(target), "source_rank": old_rank,
        "target_rank": rank, "source_alpha_over_rank": old_alpha / old_rank,
        "target_alpha_over_rank": (2 * rank) / rank,
        "max_restored_tensor_difference": max_restore_diff,
        "nonzero_values_in_padding": padded_nonzero,
        "verified": max_restore_diff == 0.0 and padded_nonzero == 0,
        "reason": "zero padding preserves B@A exactly when alpha/rank is unchanged",
    }
    write_json(target / "equivalence_report.json", report)
    if not report["verified"]:
        raise RuntimeError("CONFOUNDED_EXPERIMENT: historical pool upcast failed")
    return target


def copied_snapshot(run_root, task_id, rank):
    prev_id = task_id - 1
    source = FORMAL / "task{}".format(prev_id) / "snapshots/task{}".format(prev_id)
    target = run_root / "task{}".format(prev_id) / "snapshots/task{}".format(prev_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    manifest = read_json(target / "manifest.json")
    if rank > 8:
        source_pool = Path(manifest["pool_checkpoint_dir"])
        if not source_pool.is_absolute():
            source_pool = REPO / source_pool
        fixed = STUDY / "fixed_history" / "task{}_rank{}".format(prev_id, rank) / "pool"
        manifest["pool_checkpoint_dir"] = str(upcast_pool(source_pool, fixed, rank))
        write_json(target / "manifest.json", manifest)
    return target


def make_config(run_root, rank):
    config = yaml.safe_load((FORMAL / "metadata/resolved_config.yaml").read_text(encoding="utf-8"))
    config["lora"]["rank"] = rank
    config["lora"]["alpha"] = float(2 * rank)
    path = run_root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def fixed_boundary(task_id):
    base = FORMAL / "task{}".format(task_id)
    residual = read_json(base / "residual/residual.json")
    formation = read_json(base / "cluster/formation.json")
    features = read_json(base / "features/train_features.json")
    ids = [str(row["sample_id"]) for row in residual if not row.get("retrieval_diagnostic", False)]
    labels = {}
    for row in formation["training_manifest"]:
        labels[str(row["sample_id"])] = int(row["cluster_expert_id"])
    return {
        "task_id": task_id, "residual_count": len(ids),
        "residual_ids_sha256": sha256_text(ids),
        "cluster_assignments_sha256": sha256_text([[sid, labels[sid]] for sid in ids]),
        "cluster_sizes": [int(x["size"]) for x in formation["formed_experts"]],
        "query_hash": features["query_hash"],
        "query_encoder_hash": features["query_encoder_hash"],
        "formal_formation_path": str(base / "cluster/formation.json"),
    }


def make_single_formation(task_root):
    residual = read_json(task_root / "residual/residual.json")
    features = read_json(task_root / "features/train_features.json")
    ids = [str(row["sample_id"]) for row in residual if not row.get("retrieval_diagnostic", False)]
    queries = torch.tensor([features["records"][sid]["query"] for sid in ids], dtype=torch.float32)
    centroid = torch.nn.functional.normalize(queries.mean(dim=0), dim=0).tolist()
    formation = {
        "schema_version": 1, "task_id": 0, "query_hash": features["query_hash"],
        "noise_sample_ids": [],
        "formed_experts": [{"expert_id": 0, "cluster_id": 0, "sample_ids": ids,
                            "size": len(ids), "centroid": centroid,
                            "key_mode": "learnable", "creation_task": 0}],
        "training_manifest": [{"sample_id": sid, "cluster_expert_id": 0,
                               "expert_ids": [0], "new_expert_ids": [0],
                               "teacher_ids": []} for sid in ids],
    }
    write_json(task_root / "cluster/formation.json", formation)
    write_json(task_root / "cluster/assignment_stats.json", {
        "selected_k": 1, "selected_silhouette": None,
        "silhouette_by_k": {"1": None}, "noise_sample_ids": [],
        "cluster_sizes": [len(ids)], "source": "single-bootstrap-no-clustering",
    })
    write_json(task_root / "cluster/clusters.json", {
        "schema_version": 1, "query_hash": features["query_hash"],
        "result": {"task_id": 0, "selected_k": 1, "selected_silhouette": None,
                   "silhouette_by_k": {"1": None}, "clusters": formation["formed_experts"],
                   "noise_sample_ids": [], "seed": 42, "query_hash": features["query_hash"]},
    })


def prepare_run(run_root, task_id, rank, single=False):
    refuse_nonempty(run_root)
    task_root = run_root / "task{}".format(task_id)
    task_root.mkdir(parents=True)
    source = FORMAL / "task{}".format(task_id)
    for name in ("features", "residual"):
        os.symlink(source / name, task_root / name, target_is_directory=True)
    for rel in ("data/teacher_train.json", "data/teacher_val.json"):
        dst = task_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
    shutil.copy2(source / "query_encoder.pt", task_root / "query_encoder.pt")
    (task_root / "cluster").mkdir()
    for name in ("formation.json", "clusters.json", "assignment_stats.json"):
        shutil.copy2(source / "cluster" / name, task_root / "cluster" / name)
    if single:
        make_single_formation(task_root)
    (task_root / "state").mkdir()
    if task_id == 0:
        ExpertRegistry().save_json(str(task_root / "state/expert_registry.json"))
    else:
        snap = copied_snapshot(run_root, task_id, rank)
        shutil.copy2(snap / "expert_registry.json", task_root / "state/expert_registry.json")
    formal_state = read_json(source / "state/task_state.json")
    state = copy.deepcopy(formal_state)
    state["history"] = state["history"][:5]
    state["stage"] = "CLUSTERS_READY"
    if single:
        state["history"][-1] = {"from": "RESIDUAL_READY", "to": "CLUSTERS_READY",
                                "note": "1 single-bootstrap expert formed from fixed task0 boundary"}
    write_json(task_root / "state/task_state.json", state)
    (task_root / "stages").mkdir()
    for marker in PRE_STAGES:
        (task_root / "stages" / (marker + ".done")).write_text("done\n", encoding="utf-8")
    make_config(run_root, rank)
    boundary = fixed_boundary(task_id)
    boundary.update({"rank": rank, "alpha": 2 * rank, "alpha_over_rank": 2.0,
                     "num_experts": 1 if single else 2, "single_bootstrap": single})
    write_json(run_root / "fixed_boundary.json", boundary)
    return boundary


def prepare_all():
    STUDY.mkdir(parents=True, exist_ok=True)
    boundaries = []
    for task_id in (0, 1, 4):
        for rank in (8, 16, 32):
            root = STUDY / "A" / "task{}_rank{}".format(task_id, rank)
            boundaries.append(prepare_run(root, task_id, rank, single=False))
    for rank in (8, 16, 32):
        root = STUDY / "B" / "task0_single_rank{}".format(rank)
        boundaries.append(prepare_run(root, 0, rank, single=True))
    write_json(STUDY / "fixed_boundary_manifest.json", {"runs": boundaries})
    print("prepared {} controlled runs".format(len(boundaries)))


def validate_all():
    expected = []
    for task_id in (0, 1, 4):
        for rank in (8, 16, 32): expected.append(STUDY / "A" / "task{}_rank{}".format(task_id, rank))
    for rank in (8, 16, 32): expected.append(STUDY / "B" / "task0_single_rank{}".format(rank))
    rows = []
    for root in expected:
        boundary = read_json(root / "fixed_boundary.json")
        config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
        task_id = int(boundary["task_id"])
        current = fixed_boundary(task_id)
        checks = {
            "residual_ids": current["residual_ids_sha256"] == boundary["residual_ids_sha256"],
            "query_hash": current["query_hash"] == boundary["query_hash"],
            "query_encoder_hash": current["query_encoder_hash"] == boundary["query_encoder_hash"],
            "rank": int(config["lora"]["rank"]) == int(boundary["rank"]),
            "alpha": float(config["lora"]["alpha"]) == float(boundary["alpha"]),
            "scale": float(config["lora"]["alpha"]) / int(config["lora"]["rank"]) == 2.0,
            "seed42": int(config["data"]["seed"]) == 42,
        }
        if not boundary["single_bootstrap"]:
            checks["cluster_assignments"] = current["cluster_assignments_sha256"] == boundary["cluster_assignments_sha256"]
        status = "PASS" if all(checks.values()) else "CONFOUNDED_EXPERIMENT"
        rows.append({"run": str(root.relative_to(STUDY)), "status": status,
                     "checks": checks, "rank": boundary["rank"],
                     "alpha": boundary["alpha"], "alpha_over_rank": 2.0})
    write_json(STUDY / "controlled_comparison_validation.json", {"runs": rows})
    bad = [row for row in rows if row["status"] != "PASS"]
    if bad:
        raise RuntimeError("CONFOUNDED_EXPERIMENT: {} invalid runs".format(len(bad)))
    print("all {} controlled-run boundaries validated".format(len(rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "validate"))
    args = parser.parse_args()
    if args.command == "prepare": prepare_all()
    else: validate_all()


if __name__ == "__main__":
    main()
