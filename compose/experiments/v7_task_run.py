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
import tempfile
from pathlib import Path

import torch
import yaml

from compose.v7.commit import commit_retained_candidates
from compose.v7.config import V7Config
from compose.v7.pool import V7ExpertKeyPool
from compose.v7.provenance import (
    audit_split_isolation,
    bind_pipeline_data_usage,
    build_runtime_contract,
)
from compose.v7.pruning import CandidatePruner
from compose.v7.routing import GlobalTop2Router
from compose.v7.workflow import (
    mean_nll,
    prepare_candidate_pool,
    queries_from_cache,
    route_manifest,
    validate_query_cache_contract,
    write_full_split_with_unique_ids,
)


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(command, env, log_path):
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def marker(root, name):
    return Path(root) / "stages" / (name + ".done")


def stage_done(root, name, run_contract_hash):
    target = marker(root, name)
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError("unbound or invalid V7 stage marker: {}".format(target)) from error
    if payload.get("run_contract_hash") != run_contract_hash:
        raise ValueError("stale V7 stage marker contract: {}".format(target))
    return True


def mark(root, name, run_contract_hash):
    target = marker(root, name)
    write_json_atomic(target, {"stage": name, "run_contract_hash": run_contract_hash})


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path):
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return sha256(root)
    digest = hashlib.sha256()
    for candidate in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def build_run_contract(args, config, formal_run, gradient_accumulation_steps):
    paths = {
        "method_config": args.config,
        "train": args.train_file,
        "validation": args.val_file,
        "test": args.test_file,
        "validation_annotation": args.validation_annotation_file,
    }
    files = {}
    for name, value in paths.items():
        if value:
            resolved = Path(value).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(str(resolved))
            files[name] = {"path": str(resolved), "sha256": sha256(resolved)}
    orchestrator_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if orchestrator_world_size != 1:
        raise ValueError(
            "V7 task orchestrator must be single-process; WORLD_SIZE={} is unsupported. "
            "Only its S3 training subprocess may use torchrun/DDP.".format(
                orchestrator_world_size
            )
        )
    world_size = int(getattr(args, "training_world_size", 1))
    requested_batch = getattr(args, "training_per_device_batch_size", None)
    requested_workers = getattr(args, "training_dataloader_num_workers", None)
    per_device_batch = (
        requested_batch
        if requested_batch is not None
        else config.training.per_device_train_batch_size
    )
    dataloader_workers = (
        requested_workers
        if requested_workers is not None
        else config.training.dataloader_num_workers
    )
    actual_batch = (
        per_device_batch * gradient_accumulation_steps * world_size
    )
    target_batch = 63 if world_size == 3 else 64
    if formal_run and actual_batch != target_batch:
        raise ValueError(
            "formal V7 effective global batch must be {} for world_size={}, got {}".format(
                target_batch, world_size, actual_batch
            )
        )
    recipe = {
        "num_train_epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "effective_global_batch_size": actual_batch,
        "target_global_batch_size": target_batch,
        "global_batch_relative_difference": (actual_batch - target_batch) / target_batch,
        "training_gpus": getattr(args, "training_gpus", None),
        "distributed_backend": getattr(args, "distributed_backend", "nccl"),
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "bf16": config.training.bf16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "seed": config.training.seed,
        "dataloader_num_workers": dataloader_workers,
        "save_strategy": config.training.save_strategy,
        "dataloader_drop_last": False,
        "max_steps": -1 if formal_run else args.smoke_max_steps,
        "max_samples": None,
    }
    contract = {
        "schema_version": 1,
        "git_sha": git_head(),
        "task_index": args.task_index,
        "task_name": args.task_name,
        "formal_run": formal_run,
        "validation_metric": args.validation_metric,
        "files": files,
        "previous_checkpoint": {
            "path": str(Path(args.previous_checkpoint).resolve()),
            "sha256": sha256_tree(args.previous_checkpoint),
        } if args.previous_checkpoint else None,
        "model_path": str(Path(args.model_path).resolve()),
        "vision_tower": str(Path(args.vision_tower).resolve()),
        "projector_path": str(Path(args.projector_path).resolve()),
        "image_folder": str(Path(args.image_folder).resolve()),
        "recipe": recipe,
        "method": config.method,
        "method_seed": config.seed,
    }
    contract["contract_hash"] = stable_hash(contract)
    return contract


def bind_run_contract(root, expected, resume, had_entries):
    path = Path(root) / "data" / "run_contract.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(
                "V7 resume contract mismatch: expected {} but found {}".format(
                    expected["contract_hash"], observed.get("contract_hash")
                )
            )
    elif resume and had_entries:
        raise ValueError("non-empty V7 resume root has no bound run contract")
    else:
        write_json_atomic(path, expected)
    return expected["contract_hash"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--test-file")
    parser.add_argument("--task-name")
    parser.add_argument(
        "--validation-metric",
        choices=("official_ucit", "nll_fallback"),
        default=None,
        help="formal runs must select a task-specific official metric or explicit fallback",
    )
    parser.add_argument("--validation-annotation-file")
    parser.add_argument("--previous-checkpoint")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-world-size", type=int, default=1)
    parser.add_argument(
        "--training-gpus",
        help="comma-separated physical GPU IDs visible only to the S3 torchrun subprocess",
    )
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--training-gradient-accumulation-steps", type=int)
    parser.add_argument("--training-per-device-batch-size", type=int)
    parser.add_argument("--training-dataloader-num-workers", type=int)
    parser.add_argument(
        "--smoke-max-steps", type=int, default=None,
        help="explicit smoke/debug optimizer-step cap; formal runs omit max_steps",
    )
    parser.add_argument(
        "--smoke-gradient-accumulation-steps", type=int, default=1,
        help="explicit smoke-only accumulation override; formal runs use the YAML recipe",
    )
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
    had_entries = root.exists() and any(root.iterdir())
    if had_entries and not args.resume:
        raise FileExistsError("non-empty V7 task root requires --resume")
    root.mkdir(parents=True, exist_ok=True)
    config = V7Config.from_dict(yaml.safe_load(Path(args.config).read_text()))
    if config.method != "v7_global_coevolution":
        raise ValueError("wrong method")
    if args.smoke_max_steps is not None and args.smoke_max_steps <= 0:
        raise ValueError("--smoke-max-steps must be positive")
    if args.smoke_gradient_accumulation_steps <= 0:
        raise ValueError("--smoke-gradient-accumulation-steps must be positive")
    if args.training_world_size <= 0:
        raise ValueError("--training-world-size must be positive")
    training_gpu_ids = (
        [value.strip() for value in args.training_gpus.split(",") if value.strip()]
        if args.training_gpus else [args.device.split(":")[-1]]
    )
    if len(training_gpu_ids) != args.training_world_size:
        raise ValueError("training GPU count must equal --training-world-size")
    formal_run = args.smoke_max_steps is None
    if formal_run and not args.test_file:
        raise ValueError("formal V7 requires an explicit --test-file")
    if formal_run and args.validation_metric is None:
        raise ValueError("formal V7 requires an explicit --validation-metric")
    validation_metric = args.validation_metric or "nll_fallback"
    if validation_metric == "official_ucit" and not args.validation_annotation_file:
        raise ValueError("official validation metric requires --validation-annotation-file")
    gradient_accumulation_steps = args.training_gradient_accumulation_steps
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = (
            config.training.gradient_accumulation_steps
            if formal_run else args.smoke_gradient_accumulation_steps
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("training gradient accumulation must be positive")
    if args.training_per_device_batch_size is not None and args.training_per_device_batch_size <= 0:
        raise ValueError("training per-device batch size must be positive")
    if args.training_dataloader_num_workers is not None and args.training_dataloader_num_workers < 0:
        raise ValueError("training dataloader workers must be non-negative")
    run_contract = build_run_contract(
        args, config, formal_run, gradient_accumulation_steps
    )
    run_contract_hash = bind_run_contract(root, run_contract, args.resume, had_entries)
    write_json_atomic(root / "data" / "formal_recipe.json", run_contract["recipe"])
    print(
        "V7 recipe: world_size={world_size} per_device_batch={per_device_train_batch_size} "
        "gradient_accumulation={gradient_accumulation_steps} "
        "effective_global_batch={effective_global_batch_size} max_steps={max_steps}".format(
            **run_contract["recipe"]
        )
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    worker_device = "cuda:0" if args.device.startswith("cuda") else args.device
    previous_keys = None
    if args.previous_checkpoint:
        previous_keys = str(Path(args.previous_checkpoint) / "v7_keys.pt")

    train_json = root / "data" / "train_full.json"
    val_json = root / "data" / "val_full.json"
    runtime_contract_path = root / "data" / "runtime_contract.json"
    if not stage_done(root, "s0_full_data", run_contract_hash):
        split_audit = bind_pipeline_data_usage(
            audit_split_isolation(args.train_file, args.val_file, args.test_file),
            training_sources=(args.train_file,),
            key_learning_sources=(args.train_file,),
            rms_sources=(args.val_file,),
            pruning_sources=(args.val_file,),
        )
        train_count = write_full_split_with_unique_ids(
            args.train_file, str(train_json), args.task_index, "train"
        )
        val_count = write_full_split_with_unique_ids(
            args.val_file, str(val_json), args.task_index, "val"
        )
        splits = split_audit["splits"]
        overlap = split_audit["overlap_checks"]
        write_json(root / "data" / "coverage.json", {
            "num_train_samples": train_count,
            "num_validation_samples": val_count,
            "train_source": args.train_file,
            "validation_source": args.val_file,
            "train_sha256": splits["train"]["file_sha256"],
            "val_sha256": splits["validation"]["file_sha256"],
            "test_sha256": splits.get("test", {}).get("file_sha256"),
            "train_val_overlap_count": overlap["train_vs_validation"]["source_record_overlap"],
            "train_test_overlap_count": overlap.get("train_vs_test", {}).get("source_record_overlap", 0),
            "val_test_overlap_count": overlap.get("validation_vs_test", {}).get("source_record_overlap", 0),
            "test_data_used_for_training": split_audit["test_data_used_for_training"],
            "test_data_used_for_key_learning": split_audit["test_data_used_for_key_learning"],
            "test_data_used_for_rms": split_audit["test_data_used_for_rms"],
            "test_data_used_for_pruning": split_audit["test_data_used_for_pruning"],
            "formal_run": formal_run,
        })
        write_json(root / "data" / "split_provenance.json", split_audit)
        write_json(
            runtime_contract_path,
            build_runtime_contract(
                image_aspect_ratio=config.runtime.image_aspect_ratio,
                vision_tower=args.vision_tower,
                mm_vision_select_layer=config.runtime.mm_vision_select_layer,
                mm_vision_select_feature=config.runtime.mm_vision_select_feature,
                mm_projector_type=config.runtime.mm_projector_type,
                projector_path=args.projector_path,
            ),
        )
        mark(root, "s0_full_data", run_contract_hash)
    coverage = json.loads((root / "data" / "coverage.json").read_text())
    if args.stop_after == "full_data":
        return

    if not stage_done(root, "s1_fixed_queries", run_contract_hash):
        for split, path in (("train", train_json), ("val", val_json)):
            run([
                args.python, "-m", "compose.eval.query_features",
                "--questions", str(path), "--images", args.image_folder,
                "--output", str(root / "features" / (split + ".json")),
                "--query-vision-model", config.query.path,
                "--query-mode", "v7_fixed", "--device", worker_device,
            ], env, root / "logs" / ("features_" + split + ".log"))
        mark(root, "s1_fixed_queries", run_contract_hash)
    query_contract_path = root / "data" / "query_contract.json"
    query_contract = validate_query_cache_contract(
        (root / "features" / "train.json", root / "features" / "val.json"),
        config.query.backbone,
        config.query.path,
    )
    write_json(query_contract_path, query_contract)
    if args.stop_after == "fixed_queries":
        return

    if not stage_done(root, "s2_candidates", run_contract_hash):
        pool, center, audit = prepare_candidate_pool(
            str(root / "features" / "train.json"),
            coverage["num_train_samples"], args.task_index, config.seed,
            config.candidates.key_perturbation, previous_keys,
        )
        (root / "state").mkdir(parents=True, exist_ok=True)
        torch.save(pool.export_state(), root / "state" / "candidate_keys.pt")
        write_json(root / "metrics" / "candidate_initialization.json", audit)
        mark(root, "s2_candidates", run_contract_hash)
    if args.stop_after == "candidates":
        return
    pool = V7ExpertKeyPool.from_state(
        torch.load(root / "state" / "candidate_keys.pt", weights_only=False)
    )
    candidate_ids = pool.current_ids

    output = root / "training"
    if not stage_done(root, "s3_training", run_contract_hash):
        train_prefix = [args.python, "-m", "compose.train.train_compose"]
        if args.training_world_size > 1:
            train_prefix = [
                args.python, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(args.training_world_size),
                "-m", "compose.train.train_compose",
            ]
        command = train_prefix + [
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
            "--compose_v7_runtime_contract", str(runtime_contract_path),
            "--version", "v1",
            "--pretrain_mm_mlp_adapter", args.projector_path,
            "--mm_projector_type", config.runtime.mm_projector_type,
            "--mm_vision_select_layer", str(config.runtime.mm_vision_select_layer),
            "--mm_vision_select_feature", config.runtime.mm_vision_select_feature,
            "--image_aspect_ratio", config.runtime.image_aspect_ratio,
            "--per_device_train_batch_size",
            str(run_contract["recipe"]["per_device_train_batch_size"]),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--num_train_epochs", str(config.training.num_train_epochs),
            "--learning_rate", str(config.training.learning_rate),
            "--weight_decay", str(config.training.weight_decay),
            "--warmup_ratio", str(config.training.warmup_ratio),
            "--lr_scheduler_type", config.training.lr_scheduler_type,
            "--save_strategy", config.training.save_strategy,
            "--logging_steps", str(config.training.logging_steps),
            "--bf16", str(config.training.bf16), "--tf32", str(config.training.tf32),
            "--gradient_checkpointing", str(config.training.gradient_checkpointing),
            "--group_by_modality_length", str(config.training.group_by_modality_length),
            "--dataloader_num_workers",
            str(run_contract["recipe"]["dataloader_num_workers"]),
            "--seed", str(config.training.seed), "--report_to", "none",
            "--model_max_length", str(config.training.model_max_length),
            "--remove_unused_columns", "False",
            "--compose_v7_require_full_coverage", str(formal_run),
            "--ddp_find_unused_parameters", "True",
        ]
        if args.smoke_max_steps is not None:
            command += ["--max_steps", str(args.smoke_max_steps), "--save_steps", "10"]
        if args.previous_checkpoint:
            command += ["--compose_checkpoint", args.previous_checkpoint]
            origins = [
                "{}={}".format(expert_id, pool.metadata[expert_id]["origin_task"])
                for expert_id in pool.historical_ids
            ]
            command += ["--compose_existing_expert_origins", ",".join(origins)]
        training_env = dict(env)
        training_env["CUDA_VISIBLE_DEVICES"] = ",".join(training_gpu_ids)
        training_env["V7_DISTRIBUTED_BACKEND"] = args.distributed_backend
        run(command, training_env, root / "logs" / "training.log")
        mark(root, "s3_training", run_contract_hash)
    if args.stop_after == "training":
        return

    trained_pool = V7ExpertKeyPool.from_state(
        torch.load(output / "v7_key_pool.pt", weights_only=False)
    )
    if not stage_done(root, "s4_rms", run_contract_hash):
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
            "--runtime-contract", str(runtime_contract_path),
        ]
        if args.previous_checkpoint:
            previous_manifest = json.loads(
                (Path(args.previous_checkpoint) / "compose_experts.json").read_text()
            )
            previous_calibration = root / "rms" / "previous_calibration.json"
            write_json(previous_calibration, previous_manifest.get("rms_calibration", {}))
            rms_command += ["--frozen-calibration", str(previous_calibration)]
        run(rms_command, env, root / "logs" / "rms.log")
        mark(root, "s4_rms", run_contract_hash)
    if args.stop_after == "rms":
        return

    if not stage_done(root, "s5_pruning_commit", run_contract_hash):
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
                "--image-aspect-ratio", config.runtime.image_aspect_ratio,
                "--runtime-contract", str(runtime_contract_path),
            ], env, root / "logs" / "pruning_{}.log".format(index))
            loss = mean_nll(str(nll_output))
            if validation_metric == "official_ucit":
                answers = root / "pruning" / "answers_{}.jsonl".format(index)
                summary = root / "pruning" / "generation_{}.json".format(index)
                run([
                    args.python, "-m", "compose.eval.eval_task",
                    "--adapter-kind", "compose", "--model-path", args.model_path,
                    "--checkpoint-dir", str(output), "--projector-path", args.projector_path,
                    "--vision-tower", args.vision_tower, "--question-file", str(val_json),
                    "--image-folder", args.image_folder, "--answers-file", str(answers),
                    "--run-summary-file", str(summary), "--selection-manifest", str(selections),
                    "--device", worker_device, "--runtime-contract", str(runtime_contract_path),
                ], env, root / "logs" / "pruning_generation_{}.log".format(index))
                metric_output = root / "pruning" / "official_metric_{}.json".format(index)
                annotation_file = args.validation_annotation_file
                if Path(annotation_file).resolve() == Path(args.val_file).resolve():
                    # Classification/instruction annotations are rewritten with collision-safe
                    # validation IDs; score against that exact rewritten validation artifact.
                    annotation_file = str(val_json)
                run([
                    args.python, "-m", "compose.eval.v7_validation_metric",
                    "--task-index", str(args.task_index),
                    "--annotation-file", annotation_file,
                    "--predictions-file", str(answers),
                    "--work-root", str(root / "pruning" / "official_work_{}".format(index)),
                    "--output", str(metric_output),
                ], env, root / "logs" / "pruning_metric_{}.log".format(index))
                official = json.loads(metric_output.read_text())
                return {
                    "metric": float(official["value"]),
                    "loss": loss,
                    "official_metric": official,
                    "answer_nll": loss,
                    "metric_fallback": False,
                }
            return {
                "metric": -loss,
                "loss": loss,
                "official_metric": None,
                "answer_nll": loss,
                "metric_fallback": True,
                "fallback_reason": "explicit_nll_fallback",
            }

        retained, metrics, audit = CandidatePruner(trained_pool, config.pruning).evaluate(
            train_queries, val_queries, center, scorer
        )
        audit["performance_metric"] = (
            "task_specific_official_ucit"
            if validation_metric == "official_ucit"
            else "negative_answer_nll_explicit_fallback"
        )
        write_json(root / "metrics" / "candidate_pruning.json", {
            "retained_candidate_ids": list(retained),
            "retained_candidate_count": len(retained),
            "pool_size_before_task": len(trained_pool.historical_ids),
            "pool_size_after_task": len(trained_pool.historical_ids) + len(retained),
            "candidates": {str(key): value for key, value in metrics.items()},
            "audit": audit,
        })
        write_json(
            root / "metrics" / "candidate_pruning_trajectory.json",
            audit["pruning_trajectory"],
        )
        commit_retained_candidates(
            str(output), str(root / "committed"), trained_pool, retained, metrics
        )
        mark(root, "s5_pruning_commit", run_contract_hash)

    if args.test_file and not args.skip_eval and not stage_done(
        root, "s6_inference", run_contract_hash
    ):
        run([
            args.python, "-m", "compose.eval.eval_task", "--adapter-kind", "compose",
            "--model-path", args.model_path, "--checkpoint-dir", str(root / "committed"),
            "--projector-path", args.projector_path, "--vision-tower", args.vision_tower,
            "--question-file", args.test_file, "--image-folder", args.image_folder,
            "--answers-file", str(root / "eval" / "answers.jsonl"),
            "--run-summary-file", str(root / "eval" / "summary.json"),
            "--v7-key-state", str(root / "committed" / "v7_keys.pt"),
            "--query-vision-model", config.query.path,
            "--query-backbone-hash", str(query_contract["backbone_hash"]),
            "--device", worker_device,
            "--runtime-contract", str(runtime_contract_path),
        ], env, root / "logs" / "inference.log")
        mark(root, "s6_inference", run_contract_hash)
    artifact_provenance = {
        "git_sha": run_contract["git_sha"],
        "run_contract_hash": run_contract_hash,
        "config_hash": run_contract["files"]["method_config"]["sha256"],
        "train_sha256": run_contract["files"]["train"]["sha256"],
        "validation_sha256": run_contract["files"]["validation"]["sha256"],
        "test_sha256": run_contract["files"].get("test", {}).get("sha256"),
        "validation_annotation_sha256": run_contract["files"].get(
            "validation_annotation", {}
        ).get("sha256"),
        "query_backbone_hash": query_contract["backbone_hash"],
        "projector_sha256": json.loads(runtime_contract_path.read_text())["projector_sha256"],
        "previous_checkpoint_hash": (
            run_contract["previous_checkpoint"]["sha256"]
            if run_contract["previous_checkpoint"] else None
        ),
        "current_checkpoint_hash": sha256_tree(root / "committed"),
        "rms_artifact_hash": sha256_tree(root / "rms"),
        "pruning_artifact_hash": sha256_tree(root / "pruning"),
        "validation_metric": validation_metric,
        "seed": config.seed,
    }
    write_json_atomic(root / "artifact_provenance.json", artifact_provenance)
    write_json(root / "task_complete.json", {
        "method": config.method,
        "task_index": args.task_index,
        "num_train_samples": coverage["num_train_samples"],
        "num_queries_used_for_center": coverage["num_train_samples"],
        "checkpoint": str(root / "committed"),
        "checkpoint_hash": artifact_provenance["current_checkpoint_hash"],
        "run_contract_hash": run_contract_hash,
    })


if __name__ == "__main__":
    main()
