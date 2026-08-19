"""Task0 multi rank-8 expert experiment -- preparation.

Answers the question: is the weak Task0 ImageNet-R bootstrap result caused by
single rank-8 expert capacity, or by low-rank experts being unable to learn
ImageNet-R at all?  This module materializes the frozen V6.2 Task0 boundary
(the official seed42 formal run) plus fixed-K (K=2, K=4) query clusterings,
and writes per-expert training data + selection manifests.  It never touches
the formal run, never trains a router, and never re-runs the continual
pipeline.

Frozen boundary source: experiments/runs/compose_ucit_v62_formal_seed42/task0
  - data/teacher_train.json   (2000 records, sample ids "<image>#<index>")
  - data/teacher_val.json     (200 validation records)
  - features/train_features.json / val_features.json  (128-D functional queries)
  - query_encoder.pt          (frozen task-0 query encoder)

Everything else (test query extraction, clustering, per-expert data) is
computed here and cached under the experiment root.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import yaml

from compose.expansion.query_clustering import (
    cosine_silhouette,
    spherical_kmeans_multi_init,
)

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
FORMAL = REPO / "experiments/runs/compose_ucit_v62_formal_seed42/task0"
TEST_FILE = "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json"
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR_PATH = os.path.join(BASE_MODEL, "mm_projector.bin")
BASE_CONFIG = REPO / "configs/compose_ucit.yaml"
SEED = 42
N_INIT = 20
MAX_ITERATIONS = 100
QUERY_DIM = 128

CONFIGS = {
    "single_r8": {"rank": 8, "alpha": 16.0, "k": 1, "experts": ["0"]},
    "two_r8": {"rank": 8, "alpha": 16.0, "k": 2, "experts": ["0", "1"]},
    "four_r8": {"rank": 8, "alpha": 16.0, "k": 4, "experts": ["0", "1", "2", "3"]},
    "rank48": {"rank": 48, "alpha": 96.0, "k": 1, "experts": ["0"]},
}


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def sha256_text(values: object) -> str:
    data = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("refusing non-empty run root: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)


def build_cluster_formation(
    task_id: int,
    query_hash: str,
    sample_ids: Sequence[str],
    queries: torch.Tensor,
    k: int,
    seed: int,
) -> Dict[str, object]:
    """Fixed-K spherical clustering + V6.2-compatible formation manifests."""
    assignments, centers = spherical_kmeans_multi_init(
        queries, k, max_iterations=MAX_ITERATIONS, seed=seed, n_init=N_INIT
    )
    silhouette = cosine_silhouette(queries, assignments, k, seed=seed, sample_size=0)
    cluster_sizes = [int((assignments == c).sum()) for c in range(k)]
    formed = []
    for c in range(k):
        members = [
            sid for sid, a in zip(sample_ids, assignments.tolist()) if int(a) == c
        ]
        formed.append(
            {
                "expert_id": c,
                "cluster_id": c,
                "sample_ids": members,
                "size": len(members),
                "centroid": torch.nn.functional.normalize(
                    centers[c].detach().float(), dim=0
                ).tolist(),
                "key_mode": "learnable",
                "creation_task": task_id,
            }
        )
    centroid_sim = torch.nn.functional.cosine_similarity(
        centers.unsqueeze(0), centers.unsqueeze(1), dim=-1
    )
    formation = {
        "schema_version": 1,
        "task_id": task_id,
        "query_hash": query_hash,
        "noise_sample_ids": [],
        "formed_experts": formed,
        "training_manifest": [
            {
                "sample_id": sid,
                "cluster_expert_id": int(assignments[i].item()),
                "expert_ids": [int(assignments[i].item())],
                "new_expert_ids": [int(assignments[i].item())],
                "teacher_ids": [],
            }
            for i, sid in enumerate(sample_ids)
        ],
    }
    return {
        "formation": formation,
        "assignment_stats": {
            "selected_k": k,
            "selected_silhouette": silhouette,
            "silhouette_by_k": {str(k): silhouette},
            "cluster_sizes": cluster_sizes,
            "noise_sample_ids": [],
            "random_seed": seed,
            "n_init": N_INIT,
            "max_iterations": MAX_ITERATIONS,
            "effective_cluster_count": k,
            "centroid_cosine_similarity": {
                "matrix": centroid_sim.tolist(),
                "mean_off_diagonal": (
                    float(
                        centroid_sim[
                            torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)
                        ].mean()
                    )
                    if k > 1
                    else None
                ),
            },
            "residual_count": len(sample_ids),
        },
        "assignments": [int(value) for value in assignments.tolist()],
    }


def make_config_yaml(config_name: str, rank: int, alpha: float, output: Path) -> Dict[str, object]:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["lora"]["rank"] = rank
    config["lora"]["alpha"] = float(alpha)
    # Keep the authoritative task order untouched; only task 0 runs here.
    config["task_sequence"] = [entry for entry in config["task_sequence"] if entry["task_id"] == 0]
    payload = {"config_name": config_name}
    payload.update(config)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    config_hash = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False) + "\n", encoding="utf-8"
    )
    return {"name": config_name, "path": str(output), "hash": config_hash,
            "rank": rank, "alpha": float(alpha), "alpha_over_rank": alpha / rank}


def prepare(root_arg: str, test_device: str = "cuda:7") -> None:
    root = Path(root_arg)
    refuse_nonempty(root)
    (root / "boundary").mkdir()
    (root / "query_cache").mkdir()
    (root / "clustering").mkdir()
    (root / "training").mkdir()
    (root / "configs").mkdir()

    # ---- frozen boundary ------------------------------------------------
    for name in ("features", "residual"):
        os.symlink(FORMAL / name, root / "boundary" / name, target_is_directory=True)
    shutil.copy2(FORMAL / "query_encoder.pt", root / "boundary" / "query_encoder.pt")
    for rel in ("data/teacher_train.json", "data/teacher_val.json"):
        dst = root / "boundary" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FORMAL / rel, dst)
    train_features = read_json(FORMAL / "features/train_features.json")
    val_features = read_json(FORMAL / "features/val_features.json")
    train_queries = torch.tensor(
        [train_features["records"][sid]["query"] for sid in train_features["records"]],
        dtype=torch.float32,
    )
    train_queries = torch.nn.functional.normalize(train_queries, dim=1)
    train_sample_ids = list(train_features["records"].keys())
    query_hash = train_features["query_hash"]
    query_encoder_hash = train_features["query_encoder_hash"]
    if len(train_sample_ids) != 2000:
        raise RuntimeError("expected 2000 train samples, got {}".format(len(train_sample_ids)))

    boundary = {
        "schema_version": 1,
        "formal_task0": str(FORMAL),
        "train_samples": len(train_sample_ids),
        "val_samples": len(val_features["records"]),
        "query_hash": query_hash,
        "query_encoder_hash": query_encoder_hash,
        "train_sample_ids_sha256": sha256_text(train_sample_ids),
        "formal_formation_k": read_json(FORMAL / "cluster/assignment_stats.json")["selected_k"],
    }
    write_json(root / "boundary" / "boundary.json", boundary)

    # ---- test queries (one-shot cache, frozen query encoder) ------------
    test_features_path = root / "query_cache" / "test_features.json"
    if not test_features_path.is_file():
        command = [
            PYTHON, "-m", "compose.eval.query_features",
            "--questions", TEST_FILE,
            "--images", IMAGE_FOLDER,
            "--output", str(test_features_path),
            "--query-encoder", str(root / "boundary" / "query_encoder.pt"),
            "--seed", str(SEED),
            "--device", test_device,
            "--batch-size", "16",
        ]
        print("extracting test features: " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
    test_features = read_json(test_features_path)
    if len(test_features["records"]) != 3000:
        raise RuntimeError("expected 3000 test queries, got {}".format(len(test_features["records"])))
    if test_features["query_encoder_hash"] != query_encoder_hash:
        raise RuntimeError("test query encoder hash does not match frozen boundary")
    # Note: test query_hash cannot equal the train query_hash -- the hash is
    # over {sample_id: query} and test ids are question_id strings while train
    # ids are "<image>#<index>".  The encoder hash above is the cross-split
    # integrity check; the test hash is self-consistent within test only.
    test_sample_ids = list(test_features["records"].keys())
    write_json(root / "query_cache" / "test_ids.json", test_sample_ids)

    # ---- fixed-K clusterings --------------------------------------------
    clustering_summary = {}
    for k in (2, 4):
        out = root / "clustering" / "k{}".format(k)
        result = build_cluster_formation(0, query_hash, train_sample_ids, train_queries, k, SEED)
        write_json(out / "formation.json", result["formation"])
        write_json(out / "assignment_stats.json", result["assignment_stats"])
        write_json(
            out / "clusters.json",
            {
                "schema_version": 1,
                "query_hash": query_hash,
                "result": {
                    "task_id": 0,
                    "selected_k": k,
                    "selected_silhouette": result["assignment_stats"]["selected_silhouette"],
                    "silhouette_by_k": result["assignment_stats"]["silhouette_by_k"],
                    "clusters": result["formation"]["formed_experts"],
                    "noise_sample_ids": [],
                    "seed": SEED,
                    "query_hash": query_hash,
                },
            },
        )
        write_json(out / "assignments.json", result["assignments"])
        clustering_summary[str(k)] = {
            "sizes": result["assignment_stats"]["cluster_sizes"],
            "silhouette": result["assignment_stats"]["selected_silhouette"],
            "centroid_cosine_similarity": result["assignment_stats"]["centroid_cosine_similarity"],
        }
        print("clustering k={}: sizes={} silhouette={}".format(
            k, result["assignment_stats"]["cluster_sizes"],
            result["assignment_stats"]["selected_silhouette"],
        ), flush=True)

    # ---- per-config per-expert training data + manifests ----------------
    teacher_train = read_json(FORMAL / "data/teacher_train.json")
    by_id = {str(record["id"]): record for record in teacher_train}
    if len(by_id) != len(teacher_train):
        raise RuntimeError("teacher_train ids are not unique")
    for sid in train_sample_ids:
        if sid not in by_id:
            raise KeyError("sample id {} missing from teacher_train".format(sid))

    jobs = []
    for config_name, spec in CONFIGS.items():
        k = int(spec["k"])
        if k == 1:
            cluster_map = {sid: 0 for sid in train_sample_ids}
        else:
            assignments = read_json(root / "clustering" / "k{}".format(k) / "assignments.json")
            cluster_map = dict(zip(train_sample_ids, assignments))
        for expert_str in spec["experts"]:
            expert_id = int(expert_str)
            members = [sid for sid in train_sample_ids if int(cluster_map[sid]) == expert_id]
            expert_dir = root / "training" / config_name / "expert_{}".format(expert_id)
            expert_dir.mkdir(parents=True, exist_ok=True)
            records = [by_id[sid] for sid in members]
            write_json(expert_dir / "residual_train.json", records)
            write_json(
                expert_dir / "selection_manifest.json",
                [
                    {
                        "sample_id": sid,
                        "cluster_expert_id": expert_id,
                        "expert_ids": [expert_id],
                        "new_expert_ids": [expert_id],
                        "teacher_ids": [],
                    }
                    for sid in members
                ],
            )
            steps_per_epoch = (len(members) + 7) // 8
            jobs.append(
                {
                    "config": config_name,
                    "expert_id": expert_id,
                    "rank": int(spec["rank"]),
                    "alpha": float(spec["alpha"]),
                    "cluster_size": len(members),
                    "steps_per_epoch": steps_per_epoch,
                    "optimizer_steps_3_epochs": steps_per_epoch * 3,
                    "data_path": str(expert_dir / "residual_train.json"),
                    "manifest_path": str(expert_dir / "selection_manifest.json"),
                }
            )
        # per-config summary
        sizes = [
            sum(1 for sid in train_sample_ids if int(cluster_map[sid]) == e)
            for e in range(k)
        ]
        config_info = {
            "name": config_name,
            "k": k,
            "cluster_sizes": sizes,
            "cluster_percentages": [round(100.0 * s / 2000, 2) for s in sizes],
        }
        write_json(root / "training" / "{}.json".format(config_name), config_info)
        print("{}: cluster_sizes={}".format(config_name, sizes), flush=True)

    write_json(root / "training" / "jobs.json", jobs)

    # ---- per-config yaml -------------------------------------------------
    config_meta = {}
    for config_name, spec in CONFIGS.items():
        meta = make_config_yaml(
            config_name,
            int(spec["rank"]),
            float(spec["alpha"]),
            root / "configs" / "{}.yaml".format(config_name),
        )
        config_meta[config_name] = meta

    write_json(
        root / "experiment_manifest.json",
        {
            "schema_version": 1,
            "branch": subprocess.check_output(
                ["git", "-C", str(REPO), "branch", "--show-current"], text=True
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "seed": SEED,
            "base_model": BASE_MODEL,
            "vision_tower": VISION_TOWER,
            "dataset": TEST_FILE,
            "formal_task0": str(FORMAL),
            "query_hash": query_hash,
            "query_encoder_hash": query_encoder_hash,
            "clustering": clustering_summary,
            "configs": config_meta,
            "jobs": jobs,
            "test_samples": len(test_sample_ids),
        },
    )
    print("prepare complete: {}".format(root), flush=True)


def validate(root_arg: str) -> None:
    root = Path(root_arg)
    manifest = read_json(root / "experiment_manifest.json")
    boundary = read_json(root / "boundary" / "boundary.json")
    train_features = read_json(FORMAL / "features/train_features.json")
    checks = {
        "query_hash": boundary["query_hash"] == train_features["query_hash"],
        "query_encoder_hash": boundary["query_encoder_hash"] == train_features["query_encoder_hash"],
        "train_samples": boundary["train_samples"] == 2000,
        "test_queries": len(read_json(root / "query_cache" / "test_features.json")["records"]) == 3000,
        "configs": all(
            (root / "configs" / "{}.yaml".format(name)).is_file()
            for name in CONFIGS
        ),
    }
    for config_name, spec in CONFIGS.items():
        k = int(spec["k"])
        for expert_str in spec["experts"]:
            expert_id = int(expert_str)
            data = read_json(root / "training" / config_name / "expert_{}".format(expert_id) / "residual_train.json")
            man = read_json(root / "training" / config_name / "expert_{}".format(expert_id) / "selection_manifest.json")
            data_ids = {str(record["id"]) for record in data}
            man_ids = {str(row["sample_id"]) for row in man}
            checks["data_manifest_match_{}_{}".format(config_name, expert_id)] = (
                data_ids == man_ids and len(data_ids) > 0
            )
            if k > 1:
                assignments = read_json(root / "clustering" / "k{}".format(k) / "assignments.json")
                expected = [sid for sid, a in zip(
                    list(train_features["records"].keys()), assignments
                ) if int(a) == expert_id]
                checks["cluster_match_{}_{}".format(config_name, expert_id)] = sorted(data_ids) == sorted(expected)
    bad = {key: value for key, value in checks.items() if not value}
    if bad:
        raise RuntimeError("validation failed: {}".format(bad))
    write_json(root / "validation.json", {"status": "PASS", "checks": checks})
    print("validate complete: PASS ({} checks)".format(len(checks)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--test-device", default="cuda:7")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.test_device)
    else:
        validate(args.root)


if __name__ == "__main__":
    main()
