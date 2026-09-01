"""Resume-safe Hyper-LLaVA V7 task runner.

Unlike V6.2 this runner has no teacher, residual, clustering, calibration or
warm-up stages. The declared full train split is used for query center and
training from optimization step one.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from compose.v7.commit import commit_retained_candidates
from compose.v7.config import V7Config
from compose.v7.pool import V7ExpertKeyPool
from compose.v7.pruning import CandidatePruner
from compose.v7.routing import GlobalTop2Router
from compose.v7.workflow import (
    mean_nll,
    prepare_candidate_pool,
    queries_from_cache,
    route_manifest,
    write_full_split_with_unique_ids,
)


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command, env, log_path):
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def marker(root, name):
    return Path(root) / "stages" / (name + ".done")


def mark(root, name):
    target = marker(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--test-file")
    parser.add_argument("--previous-checkpoint")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("full_data", "fixed_queries", "candidates", "training", "rms", "commit"),
        default="commit",
        help="bounded smoke/debug stop; completed stages remain resume-safe",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("non-empty V7 task root requires --resume")
    root.mkdir(parents=True, exist_ok=True)
    config = V7Config.from_dict(yaml.safe_load(Path(args.config).read_text()))
    if config.method != "v7_global_coevolution":
        raise ValueError("wrong method")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    worker_device = "cuda:0" if args.device.startswith("cuda") else args.device
    previous_keys = None
    if args.previous_checkpoint:
        previous_keys = str(Path(args.previous_checkpoint) / "v7_keys.pt")

    train_json = root / "data" / "train_full.json"
    val_json = root / "data" / "val_full.json"
    if not marker(root, "s0_full_data").is_file():
        train_count = write_full_split_with_unique_ids(
            args.train_file, str(train_json), args.task_index, "train"
        )
        val_count = write_full_split_with_unique_ids(
            args.val_file, str(val_json), args.task_index, "val"
        )
        write_json(root / "data" / "coverage.json", {
            "num_train_samples": train_count,
            "num_validation_samples": val_count,
            "train_source": args.train_file,
            "validation_source": args.val_file,
            "test_data_used": False,
        })
        mark(root, "s0_full_data")
    coverage = json.loads((root / "data" / "coverage.json").read_text())
    if args.stop_after == "full_data":
        return

    if not marker(root, "s1_fixed_queries").is_file():
        for split, path in (("train", train_json), ("val", val_json)):
            run([
                args.python, "-m", "compose.eval.query_features",
                "--questions", str(path), "--images", args.image_folder,
                "--output", str(root / "features" / (split + ".json")),
                "--query-mode", "v7_fixed", "--device", worker_device,
            ], env, root / "logs" / ("features_" + split + ".log"))
        mark(root, "s1_fixed_queries")
    if args.stop_after == "fixed_queries":
        return

    if not marker(root, "s2_candidates").is_file():
        pool, center, audit = prepare_candidate_pool(
            str(root / "features" / "train.json"),
            coverage["num_train_samples"], args.task_index, config.seed,
            config.candidates.key_perturbation, previous_keys,
        )
        (root / "state").mkdir(parents=True, exist_ok=True)
        torch.save(pool.export_state(), root / "state" / "candidate_keys.pt")
        write_json(root / "metrics" / "candidate_initialization.json", audit)
        mark(root, "s2_candidates")
    if args.stop_after == "candidates":
        return
    pool = V7ExpertKeyPool.from_state(
        torch.load(root / "state" / "candidate_keys.pt", weights_only=False)
    )
    candidate_ids = pool.current_ids

    output = root / "training"
    if not marker(root, "s3_training").is_file():
        command = [
            args.python, "-m", "compose.train.train_compose",
            "--model_name_or_path", args.model_path,
            "--vision_tower", args.vision_tower,
            "--data_path", str(train_json), "--image_folder", args.image_folder,
            "--output_dir", str(output), "--compose_mode", config.method,
            "--compose_rank", "8", "--compose_alpha", str(config.candidates.alpha),
            "--compose_cluster_expert_ids", ",".join(map(str, candidate_ids)),
            "--compose_expert_seeds", ",".join(
                "{}={}".format(value, config.seed + args.task_index * 100 + slot)
                for slot, value in enumerate(candidate_ids)
            ),
            "--compose_origin_task_id", str(args.task_index),
            "--compose_v7_key_state", str(root / "state" / "candidate_keys.pt"),
            "--compose_v7_query_cache", str(root / "features" / "train.json"),
            "--compose_v7_config", args.config,
            "--compose_v7_metrics_path", str(root / "metrics" / "train_steps.jsonl"),
            "--compose_v7_task_index", str(args.task_index),
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--max_steps", str(args.max_steps), "--save_steps", "10",
            "--logging_steps", "1", "--bf16", "True",
            "--gradient_checkpointing", "True", "--report_to", "none",
            "--model_max_length", "2048", "--remove_unused_columns", "False",
        ]
        if args.previous_checkpoint:
            command += ["--compose_checkpoint", args.previous_checkpoint]
            origins = [
                "{}={}".format(expert_id, pool.metadata[expert_id]["origin_task"])
                for expert_id in pool.historical_ids
            ]
            command += ["--compose_existing_expert_origins", ",".join(origins)]
        run(command, env, root / "logs" / "training.log")
        mark(root, "s3_training")
    if args.stop_after == "training":
        return

    trained_pool = V7ExpertKeyPool.from_state(
        torch.load(output / "v7_key_pool.pt", weights_only=False)
    )
    if not marker(root, "s4_rms").is_file():
        bin_path = output / "compose_experts.bin"
        rms_command = [
            args.python, "-m", "compose.eval.rms_stats",
            "--model-path", args.model_path, "--vision-tower", args.vision_tower,
            "--projector-path", args.projector_path,
            "--checkpoint-dir", str(output), "--question-file", str(val_json),
            "--image-folder", args.image_folder, "--checkpoint-hash", sha256(bin_path),
            "--composition-config-hash", hashlib.sha256(
                json.dumps(config.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
            "--output-dir", str(root / "rms"), "--device", worker_device,
            "--batch-size", "1", "--new-expert-ids", ",".join(map(str, candidate_ids)),
        ]
        if args.previous_checkpoint:
            previous_manifest = json.loads(
                (Path(args.previous_checkpoint) / "compose_experts.json").read_text()
            )
            previous_calibration = root / "rms" / "previous_calibration.json"
            write_json(previous_calibration, previous_manifest.get("rms_calibration", {}))
            rms_command += ["--frozen-calibration", str(previous_calibration)]
        run(rms_command, env, root / "logs" / "rms.log")
        mark(root, "s4_rms")
    if args.stop_after == "rms":
        return

    if not marker(root, "s5_pruning_commit").is_file():
        train_queries, train_ids = queries_from_cache(
            str(root / "features" / "train.json"), coverage["num_train_samples"]
        )
        val_queries, val_ids = queries_from_cache(
            str(root / "features" / "val.json"), coverage["num_validation_samples"]
        )
        center = torch.tensor(
            json.loads((root / "metrics" / "candidate_initialization.json").read_text())["task_center"]
        )
        score_index = {"value": 0}

        def scorer(routes):
            index = score_index["value"]
            score_index["value"] += 1
            selections = root / "pruning" / "selections_{}.json".format(index)
            nll_output = root / "pruning" / "nll_{}.json".format(index)
            write_json(selections, route_manifest(val_ids, routes))
            run([
                args.python, "-m", "compose.eval.nll_eval",
                "--model-path", args.model_path, "--vision-tower", args.vision_tower,
                "--projector-path", args.projector_path,
                "--checkpoint-dir", str(output), "--question-file", str(val_json),
                "--image-folder", args.image_folder, "--selections", str(selections),
                "--output", str(nll_output), "--device", worker_device, "--batch-size", "1",
            ], env, root / "logs" / "pruning_{}.log".format(index))
            loss = mean_nll(str(nll_output))
            # Uniform audit metric when a task-specific official evaluator is
            # not configured; reports identify it explicitly as an NLL proxy.
            return {"metric": -loss, "loss": loss}

        retained, metrics, audit = CandidatePruner(trained_pool, config.pruning).evaluate(
            train_queries, val_queries, center, scorer
        )
        audit["performance_metric"] = "negative_token_average_nll_proxy"
        write_json(root / "metrics" / "candidate_pruning.json", {
            "retained_candidate_ids": list(retained),
            "retained_candidate_count": len(retained),
            "pool_size_before_task": len(trained_pool.historical_ids),
            "pool_size_after_task": len(trained_pool.historical_ids) + len(retained),
            "candidates": {str(key): value for key, value in metrics.items()},
            "audit": audit,
        })
        commit_retained_candidates(
            str(output), str(root / "committed"), trained_pool, retained, metrics
        )
        mark(root, "s5_pruning_commit")

    if args.test_file and not args.skip_eval and not marker(root, "s6_inference").is_file():
        run([
            args.python, "-m", "compose.eval.eval_task", "--adapter-kind", "compose",
            "--model-path", args.model_path, "--checkpoint-dir", str(root / "committed"),
            "--projector-path", args.projector_path, "--vision-tower", args.vision_tower,
            "--question-file", args.test_file, "--image-folder", args.image_folder,
            "--answers-file", str(root / "eval" / "answers.jsonl"),
            "--run-summary-file", str(root / "eval" / "summary.json"),
            "--v7-key-state", str(root / "committed" / "v7_keys.pt"),
            "--device", worker_device,
        ], env, root / "logs" / "inference.log")
        mark(root, "s6_inference")
    write_json(root / "task_complete.json", {
        "method": config.method,
        "task_index": args.task_index,
        "num_train_samples": coverage["num_train_samples"],
        "num_queries_used_for_center": coverage["num_train_samples"],
        "checkpoint": str(root / "committed"),
    })


if __name__ == "__main__":
    main()
