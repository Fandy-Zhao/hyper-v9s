"""V9 task-level orchestrator: one task, end to end.

The V9 loop is *sequential by construction* -- a task's historical routing set is
built from the keys the previous task committed -- so this driver runs one task
per invocation and the shell script calls it in order.  That is deliberate: a
task that dies can be resumed exactly where it stopped, and a task that has to
be re-run cannot silently inherit a later task's pool.

Per task, in order:

1. **fixed query** -- reuse the V7 encoder entry to produce
   ``features/{train,val}.json`` if they are not already there.  If they are,
   the encoder is never invoked (spec §4: the query is cached, never re-derived).
2. **pool** -- load the previous task's key state, give every historical expert
   this task's own absolute routing key at its base key, seed the candidates
   by spherical k-means
   on the *training* queries, and write the initial state.  Clustering only ever
   *initialises* a key: it never decides which samples a candidate trains on.
3. **retrieval** -- rank frozen base keys against the fixed queries once and
   cache the frozen historical block for the whole task (spec §6).
4. **training** -- hand everything to ``compose.train.train_compose
   --compose_mode v9s_responsibility_distillation``, which owns the one-forward
   closed
   loop.  Nothing about the method is configured here.
5. **audit** -- read the globally reduced task statistics, decide which
   historical task keys to retire and which candidates to commit, apply the
   decision to the pool, and write the task's audit record (spec §18, §19).
6. **state** -- write the committed pool for the next task.

No V9 recipe value is read from this file: everything comes from
``configs/v9s_*.yaml`` through :class:`~compose.v9.config.V9Config`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from compose.v8.pool import LIFECYCLE_CANDIDATE
from compose.v9.audit import (
    apply_candidate_commit,
    apply_historical_task_key_audit,
    audit_candidates,
    audit_historical_task_keys,
    candidate_usage_entropy,
    pairwise_key_cosine,
)
from compose.v9.config import V9_COMPOSE_MODE, V9Config, load_v9_config
from compose.v9.data import (
    V9QuerySource, build_task_retrieval, resolve_split_query_source,
    write_retrieval_manifest,
)
from compose.v9.keys import V9KeyPool, initialize_candidate_keys
from compose.v9.retrieval import HistoricalTopC, retrieval_diagnostics


class V9RunError(RuntimeError):
    """Raised when a task cannot be started or cannot be trusted to continue."""


# ----------------------------------------------------------------------
# small io helpers
# ----------------------------------------------------------------------
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str], env: Dict[str, str], log_path: Path) -> None:
    """Run a stage, tee-ing its output to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[v9s] $ {}".format(" ".join(str(value) for value in command)), flush=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("\n$ {}\n".format(" ".join(str(value) for value in command)))
        handle.flush()
        completed = subprocess.run(
            [str(value) for value in command],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise V9RunError(
            "stage failed with exit code {}; see {}".format(
                completed.returncode, log_path
            )
        )


# ----------------------------------------------------------------------
# 1. fixed query
# ----------------------------------------------------------------------
def queries_from_cache(path: Path) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """``[N, 1536]`` queries and the sample-id order of their rows.

    The rows are returned in **sorted id order**, which is the order the cache
    stores them in.  Everything downstream (the historical recall cache, the
    candidate key initialisation) uses this order; the dataset re-aligns by id,
    so a different record order in the JSONL is corrected rather than trusted.
    """
    payload = read_json(path)
    if payload.get("query_mode") != "v7_fixed":
        raise V9RunError("{} is not V7 fixed-query data".format(path))
    records = payload["records"]
    if not records:
        raise V9RunError("{} contains no queries".format(path))
    sample_ids = tuple(sorted(str(value) for value in records))
    queries = torch.tensor(
        [records[value]["query"] for value in sample_ids], dtype=torch.float32
    )
    if queries.ndim != 2 or queries.shape[1] != 1536:
        raise V9RunError(
            "the fixed query cache must be [N, 1536], got {}".format(tuple(queries.shape))
        )
    return queries, sample_ids


def declared_sample_ids(path: str) -> Tuple[str, ...]:
    """Canonical ids of a declared V7/V9 split, in file order."""
    records = read_json(Path(path))
    if not isinstance(records, list):
        raise V9RunError("declared split {} is not a JSON record list".format(path))
    ids = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise V9RunError("declared split {} has non-object row {}".format(path, index))
        value = record.get("id", record.get("question_id"))
        if value is None:
            raise V9RunError("declared split {} row {} has no id/question_id".format(path, index))
        ids.append(str(value))
    if not ids or len(set(ids)) != len(ids):
        raise V9RunError("declared split {} has empty or duplicate sample ids".format(path))
    return tuple(ids)


def manifest_query_source(args, task_index: int, split: str, data_path: str) -> V9QuerySource:
    try:
        return resolve_split_query_source(args.query_cache_manifest, task_index=task_index,
            split=split, expected_ids=declared_sample_ids(data_path),
            query_cache_root=args.query_cache_root)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise V9RunError("cannot consume precomputed V7 query cache for task{}.{}: {}".format(
            task_index, split, error)) from error


def resolve_training_query_source(args, config, root, env, task_index):
    """Return the sole fixed-query source; manifest mode is strict."""
    if args.query_cache_manifest:
        source = manifest_query_source(args, task_index, "train", args.train_file)
        print("[v9s] using V7 precomputed query tensor: {}".format(source.tensor_path), flush=True)
        return source
    if not args.allow_live_query_build:
        raise V9RunError(
            "formal V9-S requires --query-cache-manifest; live query building needs --allow-live-query-build"
        )
    ensure_query_cache(args, config, root, env)
    return None


def ensure_query_cache(args, config: V9Config, root: Path, env: Dict[str, str]) -> None:
    """Encode the split files' fixed queries once; never re-derive them.

    The encoder is the V7 entry, invoked unchanged.  A split that already has a
    cache is left alone: re-encoding would be a second definition of ``q_i``.
    """
    from compose.experiments.v7_task_run import _live_query_features_command

    for split, split_file in (("train", args.train_file), ("val", args.val_file)):
        target = root / "features" / "{}.json".format(split)
        if target.is_file():
            print("[v9s] fixed query cache present: {}".format(target), flush=True)
            continue
        if not split_file:
            raise V9RunError(
                "no fixed query cache at {} and no --{}-file to encode".format(
                    target, split
                )
            )
        run(
            _live_query_features_command(
                args.python,
                questions=split_file,
                images=args.image_folder,
                output=target,
                vision_model=args.query_encoder,
                device=args.device,
                batch_size=args.query_batch_size,
            ),
            env,
            root / "logs" / "queries_{}.log".format(split),
        )


# ----------------------------------------------------------------------
# 2. pool
# ----------------------------------------------------------------------
def load_previous_pool(previous_state: Optional[str], task_index: int) -> V9KeyPool:
    if not previous_state:
        if int(task_index) != 0:
            raise V9RunError(
                "task {} needs the previous task's key state; pass "
                "--previous-checkpoint".format(task_index)
            )
        return None
    if not Path(previous_state).is_file():
        raise V9RunError("previous key state {} does not exist".format(previous_state))
    state = torch.load(previous_state, map_location="cpu", weights_only=False)
    pool = V9KeyPool.from_state(state, current_task=None)
    leftover = pool.current_ids
    if leftover:
        raise V9RunError(
            "the previous task left uncommitted candidates {}: its audit did not "
            "finish, so its pool is not a valid starting point".format(leftover)
        )
    return pool


def prepare_task_pool(
    config: V9Config,
    task_index: int,
    queries: torch.Tensor,
    previous_state: Optional[str],
) -> Tuple[V9KeyPool, List[int], Dict[str, Any]]:
    """Build the pool this task trains against, with its candidate keys seeded."""
    pool = load_previous_pool(previous_state, task_index)
    if pool is None:
        pool = V9KeyPool(query_dim=config.query.query_dim)
    pool.validate()

    historical_ids = pool.historical_ids
    # Every historical expert receives this task's own routing key, initialised
    # at the expert's own base key so it starts bit-identical to the key the
    # expert was committed with.  It is an independent absolute key -- there is
    # no base-plus-delta mixing to configure -- and only this task's ``L_key``
    # trains it.  An expert that never helps has its key reset by the task-end
    # audit; not creating the key at all would make "this expert was not needed"
    # and "this expert was never considered" indistinguishable.
    for expert_id in historical_ids:
        if not pool.has_task_key(expert_id, task_index):
            pool.add_task_key(expert_id, task_index)

    existing = pool.expert_ids()
    first_id = (max(existing) + 1) if existing else 0
    candidate_ids = list(range(first_id, first_id + config.candidate_count))
    keys = initialize_candidate_keys(
        queries,
        config.candidate_count,
        strategy=config.key.candidate_init,
        perturbation=config.key.candidate_init_perturbation,
        max_samples=config.key.candidate_init_samples,
        seed=config.key.candidate_init_seed + int(task_index),
    )
    for offset, expert_id in enumerate(candidate_ids):
        pool.add_expert_with_base(
            expert_id,
            keys[offset],
            origin_task=task_index,
            lifecycle=LIFECYCLE_CANDIDATE,
            rms_state={},
            extra={"candidate_init": config.key.candidate_init},
        )
    pool.validate()
    audit = {
        "task_index": int(task_index),
        "historical_experts": sorted(historical_ids),
        "candidate_ids": sorted(candidate_ids),
        "candidate_init": config.key.candidate_init,
        "candidate_init_seed": int(config.key.candidate_init_seed) + int(task_index),
        "candidate_key_pairwise_cosine": (
            torch.nn.functional.normalize(keys, dim=-1)
            @ torch.nn.functional.normalize(keys, dim=-1).T
        ).tolist(),
        "num_task_keys": len(pool.task_key_ids(task_index)),
    }
    return pool, candidate_ids, audit


def load_trained_task_pool(
    training_dir: Path,
    initial_pool: V9KeyPool,
    task_index: int,
    candidate_ids: Sequence[int],
) -> V9KeyPool:
    """Load the only pool state that is eligible for task-end audit."""
    state_path = Path(training_dir) / "v9_key_pool.pt"
    if not state_path.is_file():
        raise V9RunError(
            "{} is missing; training did not persist its trained V9 key pool".format(
                state_path
            )
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    trained = V9KeyPool.from_state(state, current_task=None)
    trained.validate()
    if trained.query_dim != initial_pool.query_dim:
        raise V9RunError("trained key pool changed query_dim")
    if trained.expert_ids() != initial_pool.expert_ids():
        raise V9RunError("trained key pool changed expert ids before audit")
    if trained.historical_ids != initial_pool.historical_ids:
        raise V9RunError("trained key pool changed historical ids before audit")
    if sorted(trained.current_ids) != sorted(int(value) for value in candidate_ids):
        raise V9RunError("trained key pool lost or added a candidate before audit")
    if trained.key_ids() != initial_pool.key_ids():
        raise V9RunError("trained key pool changed key identities before audit")
    if trained.historical_checksums() != initial_pool.historical_checksums():
        raise V9RunError("training mutated a frozen historical key")
    for expert_id in candidate_ids:
        if trained.expert_record(int(expert_id)).get("lifecycle") != LIFECYCLE_CANDIDATE:
            raise V9RunError("candidate lifecycle changed before task-end audit")
    return trained


# ----------------------------------------------------------------------
# 4. training command
# ----------------------------------------------------------------------
def expert_seed_map(config: V9Config, task_index: int, candidate_ids: Sequence[int]) -> str:
    return ",".join(
        "{}={}".format(expert_id, int(config.seed) + int(task_index) * 100 + slot)
        for slot, expert_id in enumerate(candidate_ids)
    )


def build_training_command(
    args,
    config: V9Config,
    root: Path,
    task_index: int,
    candidate_ids: Sequence[int],
    key_state: Path,
    output_dir: Path,
    calibration_manifest: Optional[Path],
    query_source: Optional[V9QuerySource],
) -> List[str]:
    prefix = [args.python]
    world_size = int(args.training_world_size)
    launcher = str(args.training_launcher)
    if world_size > 1 and launcher == "torchrun":
        prefix += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(world_size),
        ]
    elif world_size > 1:
        # One accelerator, several ranks: ``torchrun`` would ask device 1 to
        # exist and NCCL refuses two ranks on one device.  See
        # compose/experiments/local_ranks.py -- the rendezvous is the same, the
        # device assignment and the backend are not.
        prefix += [
            "-m",
            "compose.experiments.local_ranks",
            "--nproc-per-node",
            str(world_size),
            "--",
        ]
    command = prefix + [
        "-m",
        "compose.train.train_compose",
        "--model_name_or_path", args.model_path,
        "--vision_tower", args.vision_tower,
        "--data_path", args.train_file,
        "--image_folder", args.image_folder,
        "--output_dir", str(output_dir),
        "--compose_mode", V9_COMPOSE_MODE,
        "--compose_rank", str(config.expert.rank),
        "--compose_alpha", str(config.expert.alpha),
        "--compose_dropout", str(config.expert.lora_dropout),
        "--compose_cluster_expert_ids",
        ",".join(str(value) for value in candidate_ids),
        "--compose_expert_seeds",
        expert_seed_map(config, task_index, candidate_ids),
        "--compose_origin_task_id", str(task_index),
        "--compose_v9_config", str(args.config),
        "--compose_v9_key_state", str(key_state),
        "--compose_v9_query_cache", str(query_source.tensor_path if query_source is not None else root / "features" / "train.json"),
        "--compose_v9_retrieval_cache",
        str(root / "features" / "historical_topc_task{}.pt".format(task_index)),
        "--compose_v9_metrics_path",
        str(root / "metrics" / "task{}_train_steps.jsonl".format(task_index)),
        "--compose_v9_task_index", str(task_index),
        "--compose_v9_require_full_coverage",
        "True" if args.require_full_coverage else "False",
        "--version", args.version,
        "--pretrain_mm_mlp_adapter", args.projector_path,
        "--mm_projector_type", args.mm_projector_type,
        "--mm_vision_select_layer", str(args.mm_vision_select_layer),
        "--mm_vision_select_feature", args.mm_vision_select_feature,
        "--image_aspect_ratio", args.image_aspect_ratio,
        "--per_device_train_batch_size", str(args.training_per_device_batch_size),
        "--gradient_accumulation_steps", str(args.training_gradient_accumulation_steps),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--v9_key_learning_rate", str(args.v9_key_learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_ratio", str(args.warmup_ratio),
        "--lr_scheduler_type", args.lr_scheduler_type,
        "--logging_steps", str(args.logging_steps),
        "--save_strategy", args.save_strategy,
        "--save_total_limit", str(args.save_total_limit),
        "--bf16", "True",
        "--tf32", "True",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--seed", str(config.seed),
        "--report_to", "none",
        "--model_max_length", str(args.model_max_length),
        "--remove_unused_columns", "False",
        "--ddp_find_unused_parameters", "True",
    ]
    if world_size > 1 and launcher != "torchrun":
        command += ["--ddp_backend", "gloo"]
    if args.save_strategy == "steps":
        command += ["--save_steps", str(args.save_steps)]
    if args.previous_checkpoint:
        command += ["--compose_checkpoint", str(args.previous_checkpoint)]
    if query_source is not None:
        command += ["--compose_v7_query_tensor", query_source.tensor_path]
    if calibration_manifest is not None:
        command += ["--compose_v9_calibration", str(calibration_manifest)]
    contract = root / "data" / "v9_runtime_contract.json"
    if contract.is_file():
        command += ["--compose_v9_runtime_contract", str(contract)]
    return command


# ----------------------------------------------------------------------
# 5. audit
# ----------------------------------------------------------------------
def audit_task(
    args,
    config: V9Config,
    root: Path,
    task_index: int,
    pool: V9KeyPool,
    candidate_ids: Sequence[int],
    training_dir: Path,
    manager=None,
) -> Dict[str, Any]:
    """Apply the task-end decisions to the pool (spec §18, §19).

    The statistics are the globally reduced ones the trainer wrote, so every
    rank of the finished run agrees with the decision taken here; this process
    re-reads them rather than re-deriving them.
    """
    statistics_path = training_dir / "v9_task_statistics.json"
    if not statistics_path.is_file():
        raise V9RunError("{} is missing; training did not finish".format(statistics_path))
    statistics = read_json(statistics_path)["per_expert"]

    key_decisions = audit_historical_task_keys(
        pool, statistics, config.audit, task_index
    )
    key_applied = apply_historical_task_key_audit(pool, key_decisions, task_index)

    validation_gain = None
    gain_path = training_dir / "v9_candidate_validation_gain.json"
    if gain_path.is_file():
        validation_gain = {
            int(key): float(value) for key, value in read_json(gain_path).items()
        }
    elif args.calibrate:
        raise V9RunError(
            "{} is missing although formal calibration is enabled".format(gain_path)
        )
    candidate_decisions = audit_candidates(
        pool,
        statistics,
        config.audit,
        task_index,
        validation_gain=validation_gain,
    )
    # The decision has to be *applied*, not only recorded.  Deciding without
    # applying leaves every candidate labelled a candidate, so the next task
    # finds no historical expert to recall, hands this task's candidates back to
    # ``L_ans`` (they are the only trainable experts), and refuses to start at
    # all.  ``manager`` is None here -- see ``apply_candidate_commit``.
    candidate_applied = apply_candidate_commit(
        pool, manager, candidate_decisions, task_index
    )
    pool.validate()
    return {
        "task_index": int(task_index),
        "task_key_audit": key_decisions,
        "task_key_applied": key_applied,
        "new_key_retention_rate": key_decisions["new_key_retention_rate"],
        "candidate_audit": candidate_decisions,
        "candidate_commit_applied": candidate_applied,
        "candidate_usage_entropy": candidate_usage_entropy(statistics, candidate_ids),
        "historical_slot_usage": {
            str(expert_id): statistics.get(str(expert_id), {}).get("usage_rate", 0.0)
            for expert_id in pool.historical_ids
        },
        "pairwise_key_cosine": pairwise_key_cosine(
            pool, pool.historical_ids + list(candidate_ids)
        ),
        "statistics_source": str(statistics_path),
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--mm-projector-type", default="mlp2x_gelu")
    parser.add_argument("--mm-vision-select-layer", type=int, default=-2)
    parser.add_argument("--mm-vision-select-feature", default="patch")
    parser.add_argument("--image-aspect-ratio", default="square")
    # The query encoder is the V7/V8 CLIP entry; never re-implemented here.
    parser.add_argument("--query-encoder", default=None)
    parser.add_argument("--query-cache-manifest", default=None,
        help="V7 query_cache_manifest.json or directory; uses validated queries.pt only.")
    parser.add_argument("--query-cache-root", default=None,
        help="Physical cache root; provenance remains bound to --query-cache-manifest.")
    parser.add_argument("--allow-live-query-build", action="store_true",
        help="Debug only: explicitly permit legacy live query construction without a manifest.")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--previous-checkpoint", default=None)
    parser.add_argument("--training-world-size", type=int, default=1)
    parser.add_argument(
        "--training-launcher",
        default="torchrun",
        choices=("torchrun", "local-ranks"),
        help=(
            "torchrun (default) launches one process per accelerator over NCCL, "
            "which is what a multi-GPU host uses.  local-ranks pins every rank "
            "to device 0 and switches to gloo, which is the only way to run more "
            "than one rank on this single-GPU box; it is a development launcher "
            "for the preflight, never for a reported run"
        ),
    )
    parser.add_argument("--training-per-device-batch-size", type=int, default=2)
    parser.add_argument("--training-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--v9-key-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", default="steps", choices=("steps", "no", "epoch"))
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="run the spec §30 gate-gradient vs exact-removal calibration in-process",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        choices=("queries", "pool", "retrieval", "train", "audit"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_v9_config(args.config)
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("features", "metrics", "state", "data", "logs", "training"):
        (root / name).mkdir(exist_ok=True)
    env = dict(os.environ)
    task_index = int(args.task_index)

    query_source = resolve_training_query_source(args, config, root, env, task_index)
    if query_source is None:
        queries, sample_ids = queries_from_cache(root / "features" / "train.json")
        query_source_record = {"kind": "legacy_json_query_cache", "path": str((root / "features" / "train.json").resolve()), "sha256": sha256_file(root / "features" / "train.json"), "sample_count": len(sample_ids)}
    else:
        queries, sample_ids = query_source.queries, query_source.sample_ids
        query_source_record = query_source.contract_record()

    run_contract = {
        "method": config.method,
        "task_index": task_index,
        "config_path": str(Path(args.config).resolve()),
        "config": config.to_dict(),
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "train_file": str(Path(args.train_file).resolve()),
        "previous_checkpoint": args.previous_checkpoint,
        "training_world_size": int(args.training_world_size),
        "require_full_coverage": bool(args.require_full_coverage),
        "query_source": query_source_record,
    }
    contract_path = root / "data" / "run_contract_task{}.json".format(task_index)
    if contract_path.is_file():
        previous = read_json(contract_path)
        if previous.get("config_sha256") != run_contract["config_sha256"]:
            raise V9RunError(
                "the recorded run contract for task {} was produced by a "
                "different V9 config; refusing to continue".format(task_index)
            )
        if previous.get("query_source") != run_contract["query_source"]:
            raise V9RunError(
                "the recorded run contract for task {} names a different fixed "
                "query cache; refusing to resume with shifted routing geometry".format(task_index)
            )
    write_json(contract_path, run_contract)

    # ---- 1. fixed query ----
    if args.stop_after == "queries":
        print("[v9s] fixed query cache ready for {} train samples".format(len(sample_ids)))
        return

    # ---- 2. pool ----
    state_dir = root / "state"
    initial_state = state_dir / "key_pool_task{}_initial.pt".format(task_index)
    pool, candidate_ids, pool_audit = prepare_task_pool(
        config,
        task_index,
        queries,
        _previous_state_path(state_dir, args.previous_checkpoint, task_index),
    )
    torch.save(pool.export_state(), initial_state)
    write_json(root / "data" / "pool_task{}.json".format(task_index), pool_audit)
    print(
        "[v9s] task {} pool: {} historical, candidates {}, task keys {}".format(
            task_index,
            len(pool.historical_ids),
            candidate_ids,
            pool_audit["num_task_keys"],
        ),
        flush=True,
    )
    if args.stop_after == "pool":
        return

    # ---- 3. retrieval ----
    topc = build_task_retrieval(
        pool,
        queries,
        sample_ids,
        config,
        task_index,
        str(root / "features" / "historical_topc_task{}.pt".format(task_index)),
    )
    diagnostics = retrieval_diagnostics(topc)
    write_retrieval_manifest(
        str(root / "data" / "retrieval_task{}.json".format(task_index)),
        topc,
        sample_ids,
        diagnostics,
    )
    print(
        "[v9s] historical recall: {} rows x {} slots, Top-{} reaches {} experts, "
        "wide Top-{} reaches {} ({} only in the wide recall)".format(
            diagnostics["rows"],
            diagnostics["historical_slots"],
            diagnostics["top_c"],
            diagnostics["distinct_in_topc"],
            diagnostics["wide_top_c"],
            diagnostics["distinct_in_wide_recall"],
            len(diagnostics["wide_only_experts"]),
        ),
        flush=True,
    )
    if args.stop_after == "retrieval":
        return

    calibration_manifest = None
    if args.calibrate:
        calibration_manifest = _build_calibration_manifest(
            args, config, root, task_index, query_source
        )
    if args.stop_after == "retrieval":
        return

    # ---- 4. training ----
    training_dir = root / "training" / "task{}".format(task_index)
    command = build_training_command(
        args,
        config,
        root,
        task_index,
        candidate_ids,
        initial_state,
        training_dir,
        calibration_manifest,
        query_source,
    )
    run(command, env, root / "logs" / "train_task{}.log".format(task_index))
    if args.stop_after == "train":
        return

    # ---- 5. audit ----
    trained_pool = load_trained_task_pool(
        training_dir, pool, task_index, candidate_ids
    )
    audit = audit_task(
        args, config, root, task_index, trained_pool, candidate_ids, training_dir
    )
    committed_state = state_dir / "key_pool_task{}.pt".format(task_index)
    trained_pool.validate()
    torch.save(trained_pool.export_state(), committed_state)
    audit["committed_state"] = str(committed_state)
    write_json(root / "data" / "audit_task{}.json".format(task_index), audit)
    print(
        "[v9s] task {} audit: committed {}, deleted {}, reset task keys {}, "
        "new-key retention {:.3f}".format(
            task_index,
            audit["candidate_audit"]["commit"],
            audit["candidate_audit"]["delete"],
            audit["task_key_audit"]["reset_task_key"],
            audit["new_key_retention_rate"],
        ),
        flush=True,
    )


def _previous_state_path(
    state_dir: Path, previous_checkpoint: Optional[str], task_index: int
) -> Optional[str]:
    """Where the previous task's *committed* key state lives.

    The search order matters, because the obvious answer is wrong.  A chain that
    shares one ``--root`` finds the committed state at
    ``state/key_pool_task{N-1}.pt``.  The formal sequence does not share a root
    -- the query cache is ``features/train.json``, a name without the task in it
    that is never re-encoded (spec §4), so a shared root would hand every task
    after the first the previous task's queries -- and reaches the same file
    through the third candidate below instead.

    Failing that, it is looked for beside the checkpoint directory, not inside
    it.  ``--previous-checkpoint`` names the directory ``--compose_checkpoint``
    loads the LoRA weights from, i.e. ``<root>/training/task{N-1}``, and that
    directory *does* contain a ``v9_key_pool.pt`` -- the pool as training left
    it, with this task's candidates still candidates.  Loading that one would
    hand the next task an uncommitted pool, which ``load_previous_pool``
    correctly refuses; the committed pool is written one level up, by the audit,
    after the training process has exited.
    """
    if int(task_index) == 0:
        return None
    name = "key_pool_task{}.pt".format(int(task_index) - 1)
    candidates = [state_dir / name]
    if previous_checkpoint:
        root = Path(previous_checkpoint).expanduser()
        candidates.append(root / "state" / name)
        # ``<run>/training/taskN`` -> ``<run>/state``.
        candidates.append(root.parent.parent / "state" / name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise V9RunError(
        "no committed key state for task {}; looked in {}".format(
            int(task_index) - 1, [str(value) for value in candidates]
        )
    )


def _build_calibration_manifest(
    args, config: V9Config, root: Path, task_index: int,
    train_query_source: Optional[V9QuerySource],
) -> Path:
    """Point the in-process §30 calibration at the held-out split.

    The calibration must run on samples the candidates were not fitted to, and
    it needs the same fixed query and the same recall geometry as training --
    otherwise "exact removal" would be measured on a different routing problem
    than the one the gate gradient was taken on.
    """
    if not args.val_file:
        raise V9RunError("--calibrate requires --val-file")
    val_source = None
    if train_query_source is not None:
        val_source = manifest_query_source(args, task_index, "val", args.val_file)
        val_queries, val_ids = val_source.queries, val_source.sample_ids
    else:
        val_cache = root / "features" / "val.json"
        if not val_cache.is_file():
            raise V9RunError(
                "--calibrate requires the fixed query cache {}".format(val_cache)
            )
        val_queries, val_ids = queries_from_cache(val_cache)
    pool = V9KeyPool.from_state(
        torch.load(
            root / "state" / "key_pool_task{}_initial.pt".format(task_index),
            map_location="cpu",
            weights_only=False,
        ),
        current_task=None,
    )
    topc = build_task_retrieval(
        pool,
        val_queries,
        val_ids,
        config,
        task_index,
        str(root / "features" / "historical_topc_task{}_val.pt".format(task_index)),
        # The validation recall must be built, not shared: it is a different
        # split, and a cached train artefact would be silently misaligned.
        force_build=True,
    )
    manifest = {
        "data_path": str(Path(args.val_file).resolve()),
        "query_cache": str(val_source.tensor_path if val_source is not None else val_cache),
        "query_tensor": val_source.tensor_path if val_source is not None else None,
        "query_source": val_source.contract_record() if val_source is not None else {"kind": "legacy_json_query_cache", "path": str(val_cache.resolve()), "sha256": sha256_file(val_cache), "sample_count": len(val_ids)},
        "retrieval_cache": str(
            root / "features" / "historical_topc_task{}_val.pt".format(task_index)
        ),
        "sample_budget": int(config.validation.exact_contribution_samples),
        "split": config.validation.calibration_split,
        "rows": int(topc.rows),
    }
    path = root / "data" / "calibration_task{}.json".format(task_index)
    write_json(path, manifest)
    return path


if __name__ == "__main__":
    main()
