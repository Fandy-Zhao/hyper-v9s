"""Resume-safe Hyper-LLaVA V7 task runner.

Unlike V6.2 this runner has no teacher, residual, clustering, calibration or
warm-up stages. The declared full train split is used for query center and
training from optimization step one.

GPU-count adaptive execution (spec 0903):

- the orchestrator stays a single process (WORLD_SIZE = 1);
- ``--gpus`` (or ``V7_GPUS``) is the unified GPU entry; a deterministic
  :class:`V7GPUPlan` decides stage GPU assignment and the S3 recipe;
- heavy stages run as per-GPU worker subprocesses whose partial outputs are
  merged and atomically renamed by the orchestrator only;
- without ``--gpus`` every stage behaves exactly as before (legacy launchers
  and in-flight roots are byte-compatible).
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

from compose.eval.query_features import (
    merge_query_shard_payloads,
    shard_expected_ids,
)
from compose.eval.sharding import partial_path
from compose.v7.cache_to_s1 import emit_s1_payloads_from_cache
from compose.v7.commit import commit_retained_candidates
from compose.v7.config import V7Config
from compose.v7.gpu_plan import (
    DEFAULT_PER_DEVICE_BATCH,
    DEFAULT_TARGET_GLOBAL_BATCH,
    V7GPUPlan,
    resolve_available_gpu_ids,
)
from compose.v7.pool import V7ExpertKeyPool
from compose.v7.provenance import (
    audit_split_isolation,
    bind_pipeline_data_usage,
    build_runtime_contract,
)
from compose.v7.pruning import CandidatePruner
from compose.v7.routing import GlobalTop2Router
from compose.v7.workers import (
    PooledJobRunner,
    deep_resolve,
    deferred_from,
    make_worker_env,
    run_job_logged,
    run_worker_batch,
)
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


def write_text_atomic(path, text):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
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


def build_run_contract(args, config, formal_run, gradient_accumulation_steps,
                       gpu_plan=None):
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
    if gpu_plan is not None:
        world_size = gpu_plan.training_world_size
        requested_batch = gpu_plan.recipe.per_device_batch
        target_batch = gpu_plan.recipe.target_global_batch
    else:
        world_size = int(getattr(args, "training_world_size", 1))
        requested_batch = getattr(args, "training_per_device_batch_size", None)
        target_batch = 63 if world_size == 3 else 64
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
    if formal_run and actual_batch != target_batch:
        raise ValueError(
            "formal V7 effective global batch must be {} for world_size={}, got {}".format(
                target_batch, world_size, actual_batch
            )
        )
    if gpu_plan is not None and per_device_batch != gpu_plan.recipe.per_device_batch:
        raise ValueError(
            "adaptive V7 keeps per-device batch {} (spec 0903); got {}".format(
                gpu_plan.recipe.per_device_batch, per_device_batch
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
    if gpu_plan is not None:
        # Recipe-exactness facts are explicit for the adaptive contract
        # (spec 0903 §19); legacy contracts stay byte-identical.
        recipe["recipe_mode"] = gpu_plan.recipe.recipe_mode
        recipe["recipe_exact"] = gpu_plan.recipe_exact
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
    if gpu_plan is not None:
        contract["gpu_plan"] = gpu_plan.to_dict()
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


def _split_records(records_json):
    return json.loads(Path(records_json).read_text(encoding="utf-8"))


def _validate_query_partial(partial, expected_ids):
    """A reusable query shard must still cover exactly its sample slice."""
    if not Path(partial).is_file():
        return False
    try:
        payload = json.loads(Path(partial).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    if payload.get("query_mode") != "v7_fixed":
        return False
    records = payload.get("records") or {}
    return sorted(records) == sorted(expected_ids)


def _assert_s1_payload_origin(root):
    """Cache-mode guard: every query-consuming stage must read the S1
    payloads emitted from the fixed-query cache, never a live encoder.

    Refuses to run a stage when the S1-from-cache artifacts are missing or
    when a features payload carries no ``query_origin`` marker (a legacy
    live-encoder payload would silently change the query source).
    """
    for artifact in ("data/query_cache_binding.json", "metrics/query_encoder_calls.json"):
        if not (root / artifact).is_file():
            raise ValueError(
                "cache-mode stage requires {} (S1 did not run from the "
                "fixed-query cache)".format(artifact)
            )
    marker = b'"query_origin"'
    kind = b"v7_fixed_query_cache_derived"
    for split in ("train", "val"):
        payload_path = root / "features" / (split + ".json")
        if not payload_path.is_file():
            raise ValueError("cache-mode stage requires features/{}.json".format(split))
        with payload_path.open("rb") as handle:
            head = handle.read(8192)
        if marker not in head or kind not in head:
            raise ValueError(
                "features/{}.json is not fixed-query-cache derived (missing "
                "query_origin marker)".format(split)
            )
    return True


def run_adaptive_fixed_queries(args, config, root, env, gpu_plan, run_contract_hash):
    """S1 with ``query_world_size`` per-GPU workers per split.

    Workers write ``features/tmp/<split>.json.rank{k}`` partial payloads;
    the orchestrator validates coverage, deterministically merges them into
    ``features/<split>.json`` (atomic) and recomputes the full-split
    hashes.  A resume reuses partials whose sample slice still matches;
    only missing/invalid shards are recomputed.
    """
    python = args.python
    query_world = gpu_plan.query_world_size
    tmp_dir = root / "features" / "tmp"
    for split, records_json in (("train", root / "data" / "train_full.json"),
                                ("val", root / "data" / "val_full.json")):
        records = _split_records(records_json)
        full_ids = [str(r.get("id", r.get("question_id"))) for r in records]
        output_full = root / "features" / (split + ".json")
        if query_world == 1:
            # One worker writes the final file directly, like the legacy
            # single-GPU path (identical payload and semantics).
            run([
                python, "-m", "compose.eval.query_features",
                "--questions", str(records_json), "--images", args.image_folder,
                "--output", str(output_full),
                "--query-vision-model", config.query.path,
                "--query-mode", "v7_fixed", "--device", "cuda:0",
            ], make_worker_env(env, gpu_plan.query_gpu_ids[0]),
               root / "logs" / ("features_" + split + ".log"))
            continue
        partials = []
        jobs = []
        for shard_index in range(query_world):
            partial = partial_path(str(tmp_dir / (split + ".json")), shard_index)
            partials.append(partial)
            expected = shard_expected_ids(records, query_world, shard_index)
            if _validate_query_partial(partial, expected):
                continue
            command = [
                python, "-m", "compose.eval.query_features",
                "--questions", str(records_json), "--images", args.image_folder,
                "--output", str(tmp_dir / (split + ".json")),
                "--query-vision-model", config.query.path,
                "--query-mode", "v7_fixed", "--device", "cuda:0",
                "--num-shards", str(query_world),
                "--shard-index", str(shard_index),
            ]
            gpu_id = gpu_plan.query_gpu_ids[shard_index % query_world]
            jobs.append({
                "command": command,
                "env": make_worker_env(env, gpu_id),
                "log": str(root / "logs" / ("features_{}_rank{}.log".format(split, shard_index))),
            })
        run_worker_batch(jobs)
        count = merge_query_shard_payloads(
            partials, str(output_full),
            expected_ids=full_ids,
        )
        if count != len(full_ids):
            raise ValueError(
                "merged {} query cache has {} samples, expected {}".format(
                    split, count, len(full_ids)
                )
            )
        print(
            "merged {} fixed-query shards -> {} ({} samples, {} workers)".format(
                query_world, output_full, count, query_world
            )
        )


def _shard_slice_lengths(total, num_chunks):
    """eval_task-style contiguous chunking: ceil(total / num_chunks) sizes."""
    size = -(-total // num_chunks)
    return [max(0, min(size, total - index * size)) for index in range(num_chunks)]


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
    parser.add_argument(
        "--query-cache-manifest", default=None,
        help="precomputed fixed-query cache manifest (0903 spec): S1 emits the "
             "features payloads from the sample_id-keyed cache and never invokes "
             "the CLIP query encoder (encoder_calls=0); without it S1 keeps the "
             "legacy live-encoder behavior exactly",
    )
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
        "--gpus",
        help="unified physical GPU entry for adaptive execution "
             "(CLI --gpus > $V7_GPUS > $CUDA_VISIBLE_DEVICES > idle probe)",
    )
    parser.add_argument(
        "--recipe-mode", choices=("strict", "throughput"), default="strict",
        help="strict keeps the formal global batch 64 exactly (training may "
             "use a subset of GPUs); throughput uses every GPU with the "
             "closest integer global batch (recipe_exact=False, opt-in only)",
    )
    parser.add_argument(
        "--smoke-max-steps", type=int, default=None,
        help="explicit smoke/debug optimizer-step cap; formal runs omit max_steps",
    )
    parser.add_argument(
        "--smoke-gradient-accumulation-steps", type=int, default=None,
        help="explicit smoke-only accumulation override; formal runs use the recipe",
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
    if args.smoke_gradient_accumulation_steps is not None and args.smoke_gradient_accumulation_steps <= 0:
        raise ValueError("--smoke-gradient-accumulation-steps must be positive")
    if args.training_world_size <= 0:
        raise ValueError("--training-world-size must be positive")
    if args.training_per_device_batch_size is not None and args.training_per_device_batch_size <= 0:
        raise ValueError("training per-device batch size must be positive")
    if args.training_dataloader_num_workers is not None and args.training_dataloader_num_workers < 0:
        raise ValueError("training dataloader workers must be non-negative")

    gpu_plan = None
    if args.recipe_mode == "throughput" and args.gpus is None:
        raise ValueError("recipe_mode=throughput requires --gpus (it is an explicit opt-in)")
    if args.gpus is not None:
        for legacy, label in (
            (args.training_world_size != 1, "--training-world-size"),
            (args.training_gpus is not None, "--training-gpus"),
            (args.training_per_device_batch_size is not None, "--training-per-device-batch-size"),
            (args.training_gradient_accumulation_steps is not None,
             "--training-gradient-accumulation-steps"),
        ):
            if legacy:
                raise ValueError(
                    "adaptive execution (--gpus) owns the training recipe; "
                    "drop conflicting {}".format(label)
                )
        available = resolve_available_gpu_ids(cli_ids=args.gpus)
        gpu_plan = V7GPUPlan.build(
            available,
            target_global_batch=DEFAULT_TARGET_GLOBAL_BATCH,
            per_device_batch=DEFAULT_PER_DEVICE_BATCH,
            recipe_mode=args.recipe_mode,
        )
        if args.recipe_mode == "throughput" and args.smoke_max_steps is None:
            raise ValueError(
                "formal runs must use recipe_mode=strict; throughput is an "
                "explicit smoke/benchmark opt-in"
            )
        print(gpu_plan.render_plan_block())

    training_gpu_ids = (
        [value.strip() for value in args.training_gpus.split(",") if value.strip()]
        if args.training_gpus else [args.device.split(":")[-1]]
    )
    if gpu_plan is None and len(training_gpu_ids) != args.training_world_size:
        raise ValueError("training GPU count must equal --training-world-size")
    formal_run = args.smoke_max_steps is None
    if formal_run and not args.test_file:
        raise ValueError("formal V7 requires an explicit --test-file")
    if formal_run and args.validation_metric is None:
        raise ValueError("formal V7 requires an explicit --validation-metric")
    validation_metric = args.validation_metric or "nll_fallback"
    if validation_metric == "official_ucit" and not args.validation_annotation_file:
        raise ValueError("official validation metric requires --validation-annotation-file")

    if gpu_plan is not None:
        gradient_accumulation_steps = gpu_plan.recipe.gradient_accumulation_steps
        if not formal_run and args.smoke_gradient_accumulation_steps is not None:
            gradient_accumulation_steps = args.smoke_gradient_accumulation_steps
    else:
        gradient_accumulation_steps = args.training_gradient_accumulation_steps
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = (
                config.training.gradient_accumulation_steps
                if formal_run else args.smoke_gradient_accumulation_steps or 1
            )
    if gradient_accumulation_steps <= 0:
        raise ValueError("training gradient accumulation must be positive")

    # S3 resume policy: changing the DDP world size mid-run is forbidden
    # (spec 0903 §19) unless the whole S3 recipe is equal.
    stored_contract_path = root / "data" / "run_contract.json"
    if stored_contract_path.is_file():
        stored = json.loads(stored_contract_path.read_text(encoding="utf-8"))
        stored_world = stored.get("recipe", {}).get("world_size")
        planned_world = (
            gpu_plan.training_world_size if gpu_plan is not None
            else (args.training_world_size if args.training_gpus else 1)
        )
        if stored_world is not None and int(stored_world) != int(planned_world):
            raise ValueError(
                "S3 resume requires the identical training world size: stored "
                "contract has {} but this launch plans {} (elastic DDP resume "
                "is not open; use a fresh root or the same --gpus)".format(
                    stored_world, planned_world
                )
            )

    run_contract = build_run_contract(
        args, config, formal_run, gradient_accumulation_steps,
        gpu_plan=gpu_plan,
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
    worker_device = "cuda:0"
    if gpu_plan is None:
        # Legacy: the orchestrator's single worker device drives every
        # subprocess through one physical GPU.
        env["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
        worker_device = "cuda:0" if args.device.startswith("cuda") else args.device
    else:
        # Adaptive single-worker stages (1-GPU S4/S6, orchestrator-side
        # CPU stages) run through this env; pin it to the plan's first
        # device so nothing can drift onto an unplanned physical GPU.
        # Multi-worker stages build their own per-worker envs below.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_plan.available_gpu_ids[0])
    if gpu_plan is not None:
        write_json_atomic(root / "data" / "gpu_plan.json", gpu_plan.to_dict())
        usage_log = root / "data" / "stage_gpu_usage.jsonl"
        usage_log.parent.mkdir(parents=True, exist_ok=True)
        with usage_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "plan",
                "run_contract_hash": run_contract_hash,
                "gpu_plan": gpu_plan.to_dict(),
            }, sort_keys=True) + "\n")
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
        if args.query_cache_manifest:
            # 0903 spec §5-8: S1 emits the byte-compatible features payloads
            # from the precomputed fixed-query cache (sample_id primary key,
            # manifest contract validated, fail-closed on any mismatch).  The
            # CLIP query encoder is never invoked: encoder_calls is 0 by
            # construction and recorded for the formal gates.
            import time as _time

            started = _time.perf_counter()
            emit_audit = emit_s1_payloads_from_cache(
                args.query_cache_manifest, args.task_index,
                config.query.path,
                {"train": str(train_json), "val": str(val_json)},
                {
                    "train": str(root / "features" / "train.json"),
                    "val": str(root / "features" / "val.json"),
                },
                backbone_name=config.query.backbone,
            )
            emit_audit["wall_seconds"] = round(_time.perf_counter() - started, 3)
            write_json(root / "data" / "query_cache_binding.json", emit_audit)
            write_json(root / "metrics" / "query_encoder_calls.json", {
                "stage": "s1_fixed_queries",
                "query_source": emit_audit["source"],
                "manifest_sha256": emit_audit["manifest_sha256"],
                "encoder_calls": 0,
                "wall_seconds": emit_audit["wall_seconds"],
                "splits": {
                    split["split"]: {
                        "declared_count": split["declared_count"],
                        "cached_count": split["cached_count"],
                        "sequence_matches_cache": split["sequence_matches_cache"],
                        "encoder_calls": split["encoder_calls"],
                    }
                    for split in emit_audit["splits"]
                },
            })
            print(
                "S1 from fixed-query cache: manifest {} | splits {} | "
                "encoder_calls=0 | {:.1f}s".format(
                    args.query_cache_manifest,
                    ",".join(str(value["split"]) for value in emit_audit["splits"]),
                    emit_audit["wall_seconds"],
                )
            )
        elif gpu_plan is not None:
            run_adaptive_fixed_queries(
                args, config, root, env, gpu_plan, run_contract_hash
            )
        else:
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
        if args.query_cache_manifest:
            _assert_s1_payload_origin(root)
        training_world_size = (
            gpu_plan.training_world_size if gpu_plan is not None
            else args.training_world_size
        )
        train_prefix = [args.python, "-m", "compose.train.train_compose"]
        if training_world_size > 1:
            train_prefix = [
                args.python, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(training_world_size),
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
        if gpu_plan is not None:
            training_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(value) for value in gpu_plan.training_gpu_ids
            )
        else:
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
        if args.query_cache_manifest:
            _assert_s1_payload_origin(root)
        bin_path = output / "compose_experts.bin"
        rms_base = [
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
            rms_base += ["--frozen-calibration", str(previous_calibration)]
        if gpu_plan is not None and gpu_plan.rms_world_size > 1:
            rms_command = [
                args.python, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(gpu_plan.rms_world_size),
                "-m", "compose.eval.rms_stats",
            ] + rms_base
            rms_env = dict(env)
            rms_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(value) for value in gpu_plan.rms_gpu_ids
            )
            run(rms_command, rms_env, root / "logs" / "rms.log")
        else:
            rms_command = [args.python, "-m", "compose.eval.rms_stats"] + rms_base
            run(rms_command, env, root / "logs" / "rms.log")
        mark(root, "s4_rms", run_contract_hash)
    if args.stop_after == "rms":
        return

    if not stage_done(root, "s5_pruning_commit", run_contract_hash):
        if args.query_cache_manifest:
            _assert_s1_payload_origin(root)
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

        def resolve_annotation_file():
            annotation_file = args.validation_annotation_file
            if Path(annotation_file).resolve() == Path(args.val_file).resolve():
                # Classification/instruction annotations are rewritten with
                # collision-safe validation IDs; score against that exact
                # rewritten validation artifact.
                annotation_file = str(val_json)
            return annotation_file

        if gpu_plan is not None:
            # ---- adaptive S5: candidate hypotheses of one remove-and-reroute
            # iteration execute concurrently, one GPU each, while the serial
            # trajectory logic (score -> remove one -> reroute) stays exact.
            runner = PooledJobRunner(
                gpu_plan.pruning_gpu_ids,
                usage_log_path=str(root / "data" / "stage_gpu_usage.jsonl"),
            )

            def execute_scoring_job(job_index, rows, worker_env):
                selections = root / "pruning" / "selections_{}.json".format(job_index)
                nll_output = root / "pruning" / "nll_{}.json".format(job_index)
                write_json(selections, route_manifest(val_ids, rows))
                run_job_logged([
                    args.python, "-m", "compose.eval.nll_eval",
                    "--model-path", args.model_path, "--vision-tower", args.vision_tower,
                    "--projector-path", args.projector_path,
                    "--checkpoint-dir", str(output), "--question-file", str(val_json),
                    "--image-folder", args.image_folder, "--selections", str(selections),
                    "--output", str(nll_output), "--device", "cuda:0", "--batch-size", "1",
                    "--image-aspect-ratio", config.runtime.image_aspect_ratio,
                    "--runtime-contract", str(runtime_contract_path),
                ], worker_env, root / "logs" / "pruning_{}.log".format(job_index))
                loss = mean_nll(str(nll_output))
                if validation_metric == "official_ucit":
                    answers = root / "pruning" / "answers_{}.jsonl".format(job_index)
                    summary = root / "pruning" / "generation_{}.json".format(job_index)
                    run_job_logged([
                        args.python, "-m", "compose.eval.eval_task",
                        "--adapter-kind", "compose", "--model-path", args.model_path,
                        "--checkpoint-dir", str(output), "--projector-path", args.projector_path,
                        "--vision-tower", args.vision_tower, "--question-file", str(val_json),
                        "--image-folder", args.image_folder, "--answers-file", str(answers),
                        "--run-summary-file", str(summary), "--selection-manifest", str(selections),
                        "--device", "cuda:0", "--runtime-contract", str(runtime_contract_path),
                    ], worker_env, root / "logs" / "pruning_generation_{}.log".format(job_index))
                    metric_output = root / "pruning" / "official_metric_{}.json".format(job_index)
                    run_job_logged([
                        args.python, "-m", "compose.eval.v7_validation_metric",
                        "--task-index", str(args.task_index),
                        "--annotation-file", resolve_annotation_file(),
                        "--predictions-file", str(answers),
                        "--work-root", str(root / "pruning" / "official_work_{}".format(job_index)),
                        "--output", str(metric_output),
                    ], worker_env, root / "logs" / "pruning_metric_{}.log".format(job_index))
                    official = json.loads(metric_output.read_text())
                    resolved = {
                        "metric": float(official["value"]),
                        "loss": float(loss),
                        "official_metric": official,
                        "answer_nll": float(loss),
                        "metric_fallback": False,
                    }
                else:
                    resolved = {
                        "metric": -float(loss),
                        "loss": float(loss),
                        "official_metric": None,
                        "answer_nll": float(loss),
                        "metric_fallback": True,
                        "fallback_reason": "explicit_nll_fallback",
                    }
                mark(root / "pruning", "job_{}".format(job_index), run_contract_hash)
                return resolved

            def scoring_executor(job_index, rows):
                """Body adapter: bind index/rows, own one GPU for the job."""
                def body(gpu_id, extra_env):
                    worker_env = make_worker_env(env, gpu_id)
                    return execute_scoring_job(job_index, rows, worker_env)
                return body

            def score_job_cache_valid(job_index, rows):
                target = marker(root / "pruning", "job_{}".format(job_index))
                if not target.is_file():
                    return False
                try:
                    payload = json.loads(target.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    return False
                if payload.get("run_contract_hash") != run_contract_hash:
                    return False
                selections = root / "pruning" / "selections_{}.json".format(job_index)
                if not selections.is_file():
                    return False
                # The cache is keyed by trajectory index; verify the manifest
                # covers exactly the requested rerouted rows so a stale or
                # tampered shard can never be re-scored under a wrong label.
                if json.loads(selections.read_text(encoding="utf-8")) != route_manifest(
                    val_ids, rows
                ):
                    return False
                nll_output = root / "pruning" / "nll_{}.json".format(job_index)
                nll_payload = json.loads(nll_output.read_text(encoding="utf-8"))
                if len(nll_payload) != len(val_ids):
                    return False
                if validation_metric == "official_ucit":
                    metric_output = root / "pruning" / "official_metric_{}.json".format(job_index)
                    if not metric_output.is_file():
                        return False
                return True

            def cached_scoring_result(job_index):
                loss = mean_nll(str(root / "pruning" / "nll_{}.json".format(job_index)))
                if validation_metric == "official_ucit":
                    official = json.loads(
                        (root / "pruning" / "official_metric_{}.json".format(job_index)).read_text()
                    )
                    return {
                        "metric": float(official["value"]),
                        "loss": float(loss),
                        "official_metric": official,
                        "answer_nll": float(loss),
                        "metric_fallback": False,
                    }
                return {
                    "metric": -float(loss),
                    "loss": float(loss),
                    "official_metric": None,
                    "answer_nll": float(loss),
                    "metric_fallback": True,
                    "fallback_reason": "explicit_nll_fallback",
                }

            def scorer(routes):
                index = score_index["value"]
                score_index["value"] += 1
                rows = routes.detach().cpu().tolist()
                if score_job_cache_valid(index, rows):
                    return cached_scoring_result(index)
                future = runner.submit(scoring_executor(index, rows))
                result = {
                    "metric": deferred_from(future, "metric"),
                    "loss": deferred_from(future, "loss"),
                    "official_metric": deferred_from(future, "official_metric"),
                    "answer_nll": deferred_from(future, "answer_nll"),
                    "metric_fallback": (
                        False if validation_metric == "official_ucit" else True
                    ),
                }
                if validation_metric != "official_ucit":
                    result["fallback_reason"] = "explicit_nll_fallback"
                return result
        else:
            # ---- legacy S5: the exact original sequential scorer.
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
        if gpu_plan is not None:
            # The parallel scorer handed back lazily-resolved results; the
            # serial trajectory already consumed every decision value, so
            # materialize the remainder once before the plain-JSON writers
            # (candidate metrics, commit manifest).
            metrics = deep_resolve(metrics)
            audit = deep_resolve(audit)
        pruning_payload = {
            "retained_candidate_ids": list(retained),
            "retained_candidate_count": len(retained),
            "pool_size_before_task": len(trained_pool.historical_ids),
            "pool_size_after_task": len(trained_pool.historical_ids) + len(retained),
            "candidates": {str(key): value for key, value in metrics.items()},
            "audit": audit,
        }
        write_json(root / "metrics" / "candidate_pruning.json", pruning_payload)
        write_json(
            root / "metrics" / "candidate_pruning_trajectory.json",
            audit["pruning_trajectory"],
        )
        commit_retained_candidates(
            str(output), str(root / "committed"), trained_pool, retained, metrics
        )
        if gpu_plan is not None:
            runner.shutdown()
        mark(root, "s5_pruning_commit", run_contract_hash)

    if args.test_file and not args.skip_eval and not stage_done(
        root, "s6_inference", run_contract_hash
    ):
        eval_answers = root / "eval" / "answers.jsonl"
        eval_summary = root / "eval" / "summary.json"
        selection_manifest = None
        if args.query_cache_manifest:
            # S6 cache mode: committed-pool Global Top-2 selections are
            # precomputed once from the fixed-query cache (no CLIP model, no
            # query-encoder call) on the worker GPU of the evaluation chunk
            # plan, then eval_task consumes them via --selection-manifest
            # (spec §21).  The live CLIP path below is untouched when the
            # manifest flag is absent.
            selections_path = root / "eval" / "selections.json"
            if not selections_path.is_file():
                selection_job = {
                    "command": [
                        args.python, "-m", "compose.v7.cached_selections",
                        "--cache-manifest", args.query_cache_manifest,
                        "--key-state", str(root / "committed" / "v7_keys.pt"),
                        "--questions", args.test_file,
                        "--question-task-index", str(args.task_index),
                        "--output", str(selections_path),
                        "--audit-output", str(root / "eval" / "selections_audit.json"),
                        "--backbone-path", config.query.path,
                        "--device", "cuda:0",
                    ],
                    "log": str(root / "logs" / "inference_selections.log"),
                }
                if gpu_plan is not None and gpu_plan.evaluation_world_size > 1:
                    selection_job["env"] = make_worker_env(
                        env, gpu_plan.evaluation_gpu_ids[0]
                    )
                else:
                    selection_job["env"] = env
                run_worker_batch([selection_job])
            selection_manifest = str(selections_path)
        if selection_manifest is not None:
            routing_args = ["--selection-manifest", selection_manifest]
        else:
            routing_args = [
                "--v7-key-state", str(root / "committed" / "v7_keys.pt"),
                "--query-vision-model", config.query.path,
                "--query-backbone-hash", str(query_contract["backbone_hash"]),
            ]
        if gpu_plan is not None and gpu_plan.evaluation_world_size > 1:
            test_records = _split_records(args.test_file)
            test_ids = [
                str(record.get("question_id", record.get("id")))
                for record in test_records
            ]
            chunk_count = gpu_plan.evaluation_world_size
            sizes = _shard_slice_lengths(len(test_records), chunk_count)
            partials = [
                partial_path(str(eval_answers), index) for index in range(chunk_count)
            ]
            summaries = [
                partial_path(str(eval_summary), index) for index in range(chunk_count)
            ]
            jobs = []
            for chunk_index in range(chunk_count):
                partial_answers = partials[chunk_index]
                expected_lines = sizes[chunk_index]
                if (
                    Path(partial_answers).is_file()
                    and len(Path(partial_answers).read_text(encoding="utf-8").splitlines())
                    == expected_lines
                ):
                    continue
                gpu_id = gpu_plan.evaluation_gpu_ids[chunk_index % chunk_count]
                command = [
                    args.python, "-m", "compose.eval.eval_task",
                    "--adapter-kind", "compose", "--model-path", args.model_path,
                    "--checkpoint-dir", str(root / "committed"),
                    "--projector-path", args.projector_path,
                    "--vision-tower", args.vision_tower,
                    "--question-file", args.test_file, "--image-folder", args.image_folder,
                    "--answers-file", partial_answers,
                    "--run-summary-file", summaries[chunk_index],
                    *routing_args,
                    "--device", "cuda:0",
                    "--num-chunks", str(chunk_count),
                    "--chunk-idx", str(chunk_index),
                    "--runtime-contract", str(runtime_contract_path),
                ]
                jobs.append({
                    "command": command,
                    "env": make_worker_env(env, gpu_id),
                    "log": str(root / "logs" / ("inference_rank{}.log".format(chunk_index))),
                })
            run_worker_batch(jobs)
            merged_lines = []
            for partial in partials:
                if not Path(partial).is_file():
                    raise ValueError("missing inference shard {}".format(partial))
                merged_lines.extend(
                    Path(partial).read_text(encoding="utf-8").splitlines()
                )
            if len(merged_lines) != len(test_records):
                raise ValueError(
                    "merged inference answers have {} lines, expected {}".format(
                        len(merged_lines), len(test_records)
                    )
                )
            observed_ids = [
                json.loads(line).get("question_id", json.loads(line).get("id"))
                for line in merged_lines
            ]
            if observed_ids != test_ids:
                raise ValueError("merged inference answers are out of order or incomplete")
            write_text_atomic(eval_answers, "\n".join(merged_lines) + "\n")
            merged_summary = {
                "adapter_kind": "compose",
                "checkpoint": str(root / "committed"),
                "git_commit": run_contract["git_sha"],
                "seed": 42,
                "samples": len(test_records),
                "shard_count": chunk_count,
                "shard_merge": "orchestrator_deterministic",
                "shard_summaries": [str(value) for value in summaries],
                "selection_mode": (
                    "precomputed_global_top2_validation"
                    if selection_manifest is not None
                    else "v7_global_coevolution"
                ),
                "query_source": (
                    "v7_fixed_query_cache"
                    if selection_manifest is not None
                    else "frozen_clip_l14_336_live"
                ),
            }
            write_json_atomic(eval_summary, merged_summary)
        else:
            run([
                args.python, "-m", "compose.eval.eval_task", "--adapter-kind", "compose",
                "--model-path", args.model_path, "--checkpoint-dir", str(root / "committed"),
                "--projector-path", args.projector_path, "--vision-tower", args.vision_tower,
                "--question-file", args.test_file, "--image-folder", args.image_folder,
                "--answers-file", str(eval_answers),
                "--run-summary-file", str(eval_summary),
                *routing_args,
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
