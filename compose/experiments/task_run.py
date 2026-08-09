"""Compose unified task runner for tasks 0..5 (pipeline stages S0..S12).

Implements the Query-Clustered Residual Expert Discovery pipeline:

  S0  snapshot load       -> DATA_READY
  S1  functional queries  -> QUERY_READY   (frozen CLIP + deterministic
                              query encoder; image/instruction only)
  S2  teacher search      -> OLD_TEACHER_READY (task > 0; empty/single/
                              pair over the per-sample Top-M). Task 0 is a
                              legal cold start: the stage is skipped with a
                              recorded reason and the base NLL is computed.
  S3  residual split      -> RESIDUAL_READY  (old_teacher_loss > tau_res;
                              empty pool => all cold-start residual)
  S4  recall audit        -> diagnostic only (OracleRecall@M; never gates)
  S5  clustering          -> CLUSTERS_READY or NO_EXPANSION_REQUIRED
                              (spherical K-means K=1..4, cosine silhouette)
  S6  cluster LoRA train  -> CLUSTER_EXPERTS_TRAINING -> TRAINED
                              (conditional residual: old teacher + new expert)
  S7  key learning        -> KEYS_TRAINING -> KEYS_READY (cluster-supervised
                              learnable keys; prototype mode skips training)
  S8  direct commit       -> EXPERTS_COMMITTED (no validation gates; exactly
                              one pool_version bump per expert)
  S9  RMS                 -> RMS_READY (runtime kappa, reference = mean over
                              all active experts in the layer)
  S10 snapshot            -> SNAPSHOT_READY
  S11 eval                -> EVALUATION_COMPLETE (router-based inference:
                              frozen query encoder + keys, no answers/oracle/
                              task-id lookup/clustering at test time)
  S12 report              -> COMPLETED

Every stage is idempotent (``stages/<name>.done`` marker); a resumed run
skips completed stages, never re-commits and never bumps pool_version
twice. New expert ids come exclusively from
``ExpertRegistry.next_expert_id()`` (never ``task_id * 10 + slot``), and
all heavy work is delegated to ``python -m compose...`` subprocesses.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from compose.eval.sharding import (
    merge_partial_answer_files,
    merge_partial_maps,
    partial_path,
)
from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction
from compose.expansion.commit import (
    build_cluster_expert_metadata,
    commit_cluster_expert,
)
from compose.expansion.expert_formation import (
    build_training_manifest,
    form_cluster_experts,
    load_expert_formation,
    write_expert_formation,
)
from compose.expansion.query_clustering import (
    ComposeClusteringConfig,
    cluster_residual_queries,
    load_cluster_manifest,
    write_cluster_manifest,
)
from compose.expansion.residual import (
    build_residual_records,
    should_create_experts,
    write_residual_split,
)
from compose.experiments.snapshot import ComposeSnapshot
from compose.router.key_learning import (
    ComposeKeyLearningConfig,
    initialize_key_from_centroid,
    learn_cluster_keys,
)
from compose.router.functional_query import (
    ComposeQueryEncoder,
    load_query_encoder_checkpoint,
    save_query_encoder_checkpoint,
)
from compose.router.router import (
    ComposeRouter,
    load_compose_router_checkpoint,
    save_compose_router_checkpoint,
)
from compose.teacher.oracle_set import OracleConfig
from compose.teacher.teacher import (
    ComposeTeacherSearcher,
    build_teacher_multi_hot,
    run_recall_audit,
)

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR_PATH = os.path.join(BASE_MODEL, "mm_projector.bin")
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
ROUTER_VERSION = "compose_router_v1"
FEATURE_EXTRACTOR_VERSION = "frozen_clip_l14_336_v1"

QUERY_ENCODER_NAME = "query_encoder.pt"
ROUTER_KEYS_NAME = "keys_ready_checkpoint.pt"
ROUTER_FINAL_NAME = "final_router_checkpoint.pt"

#: Default configuration. ``configs/compose_ucit.yaml`` overrides it when
#: present; the merged result is hashed and carried through every stage.
DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "purpose": "compose_ucit_formal",
    "data": {
        "seed": 42,
        "data_root": "/data/dataset/zhaozhuofan/UCIT",
        "images_root": IMAGE_FOLDER,
        "base_model": BASE_MODEL,
        "vision_tower": VISION_TOWER,
    },
    "task_sequence": [
        {"task_id": 0, "name": "ImageNet-R",
         "train_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/train.json",
         "test_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json"},
        {"task_id": 1, "name": "ArxivQA",
         "train_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/ArxivQA/train_4w.json",
         "test_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/ArxivQA/test_3000.json"},
        {"task_id": 2, "name": "VizWiz",
         "train_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/train.json",
         "test_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/test.json"},
        {"task_id": 3, "name": "IconQA",
         "train_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/IconQA/train.json",
         "test_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/IconQA/test.json"},
        {"task_id": 4, "name": "CLEVR",
         "train_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/train_4w.json",
         "test_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/test.json"},
        {"task_id": 5, "name": "Flickr30k",
         "train_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/train_brief_4w.json",
         "test_instructions": "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/test.json"},
    ],
    "tasks": {
        "teacher_search_train_samples": 2000,
        "teacher_search_validation_samples": 200,
    },
    "teacher": {
        "lambda_expert": 0.01,
        "delta_pair_raw": 0.02,
        "top_k_for_pair": 4,
        "max_pairs": 6,
        "min_recall_audit_ratio": 0.05,
    },
    "residual": {
        "tau_res": 2.0,
        "min_residual_samples": 8,
    },
    "router": {
        "top_m": 8,
        "tau_none": 0.5,
        "tau_second": 0.5,
        "max_active_experts": 2,
        "query_dim": 128,
        "key_mode": "learnable",
    },
    "clustering": {
        "max_clusters": 4,
        "silhouette_threshold": 0.15,
        "min_cluster_samples": 8,
        "random_seed": 42,
        "max_iterations": 100,
        "silhouette_sample_size": 2000,
    },
    "key_learning": {
        "learning_rate": 3.0e-4,
        "epochs": 50,
        "temperature": 0.07,
        "margin": 0.3,
        "lambda_old": 0.5,
        "lambda_div": 0.1,
        "key_separation_margin": 0.1,
        "batch_size": 256,
        "seed": 42,
    },
    "rms": {
        "epsilon": 1.0e-8,
        "kappa_min": 0.25,
        "kappa_max": 4.0,
    },
    "lora": {
        "rank": 8,
        "alpha": 16.0,
    },
    "training": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 2.0e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "save_steps": 999999,
        "model_max_length": 2048,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "cache_dir": "/tmp/compose_hf_cache",
    },
    "eval": {
        "max_new_tokens": 128,
        "nll_batch_size": 8,
    },
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _record_id(record: Dict[str, Any]) -> str:
    if "id" in record:
        return str(record["id"])
    return str(record["question_id"])


def _assign_unique_record_ids(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Stable unique per-record internal ids.

    UCIT train files are not identity-clean: question_ids repeat within a
    file (VizWiz 32.7%, Flickr30k 31.9%, CLEVR 1.5%). The pipeline keys
    features, teacher selections, residuals, cluster membership and the
    training manifest by sample_id (= record["id"] when present, else
    question_id), so two records sharing a question_id collide: clustering
    may assign the same id to two clusters and manifest construction
    crashes with "duplicate sample <id> in training manifest" (reproduced
    on the formal seed42 run, task2 S5, 2026-08-09). Tag every record
    with a unique deterministic "<question_id>#<absolute index in the
    train file>" id before features are computed; every downstream keying
    site (query_features, nll_eval, selections, residuals, cluster
    manifest, training manifest, compose selection dataset) is id-first
    and stays consistent.
    """
    annotated = []
    for index, record in enumerate(records):
        copy = dict(record)
        copy["id"] = "{}#{}".format(_record_id(record), index)
        annotated.append(copy)
    return annotated


def _question_text(record: Dict[str, Any]) -> str:
    if "conversations" in record:
        for message in record["conversations"]:
            if message["from"] == "human":
                return message["value"]
    return record.get("text", "")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _stage_done(root: Path, stage: str) -> bool:
    return (root / "stages" / "{}.done".format(stage)).is_file()


def _mark_stage(root: Path, stage: str) -> None:
    (root / "stages").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "{}.done".format(stage)).write_text(
        "done\n", encoding="utf-8"
    )


def _advance(root: Path, machine: TaskStateMachine, stage: TaskStage, note: str) -> None:
    # Idempotent under resume: a crashed run that entered the stage before
    # its heavy subprocess finished (e.g. CLUSTER_EXPERTS_TRAINING) re-enters
    # it on restart; the machine itself stays strictly one-way.
    if machine.stage is not stage:
        machine.advance(stage, note=note)
    machine.save(str(root / "state" / "task_state.json"))


def _run(
    command: List[str], env: Dict[str, str], root: Path, tag: str
) -> subprocess.CompletedProcess:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (root / "logs" / "{}_stdout.log".format(tag)).write_text(
        result.stdout, encoding="utf-8"
    )
    (root / "logs" / "{}_stderr.log".format(tag)).write_text(
        result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} failed (exit {}):\n{}".format(tag, result.returncode, result.stderr[-4000:])
        )
    return result


def _execution_plan(gpus: str) -> Dict[str, Any]:
    """Execution strategy derived from ``--gpus`` (spec §1-§4).

    Exactly 1 GPU -> single-GPU serial execution (the frozen protocol);
    exactly 4 GPUs -> 4-GPU execution: sample-sharded stages (S1/S2/S9/
    S11) run one worker per physical GPU, cluster LoRA training (S6) runs
    under torchrun DDP, and clustering/keys/commit/snapshot stay rank-0
    only on the orchestrator process. Any other count is a hard STOP.
    """
    gpu_list = [value.strip() for value in gpus.split(",") if value.strip()]
    if len(gpu_list) == 1:
        return {"mode": "single", "gpus": gpu_list, "world_size": 1}
    if len(gpu_list) == 4:
        return {
            "mode": "4gpu",
            "gpus": gpu_list,
            "world_size": 4,
            "torchrun_prefix": [
                PYTHON, "-m", "torch.distributed.run",
                "--standalone", "--max-restarts=0", "--nproc_per_node=4", "--module",
            ],
        }
    raise ValueError(
        "--gpus must name exactly 1 or 4 physical GPUs; got {!r}".format(gpus)
    )


def _worker_env(plan: Dict[str, Any], index: int) -> Dict[str, str]:
    """Env for one shard worker: exactly one physical GPU (logical cuda:0)."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpus"][index])


def _all_gpu_env(plan: Dict[str, Any]) -> Dict[str, str]:
    """Env for a torchrun launch: all physical GPUs visible, one per rank."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(plan["gpus"]))


def _torchrun_launch(plan: Dict[str, Any], command: List[str]) -> List[str]:
    """Assemble a torchrun launch from a ``python -m <module>`` command.

    torchrun's ``--module`` consumes the next positional as the module
    name, so the command's own leading ``[PYTHON, "-m"]`` is dropped:
    ``python -m torch.distributed.run ... --module <module> <args>``.
    """
    return plan["torchrun_prefix"] + command[2:]


def _run_shards(
    command_per_shard: List[List[str]],
    env_per_shard: List[Dict[str, str]],
    root: Path,
    tag: str,
) -> None:
    """Run one worker per shard concurrently (4-GPU execution, §13/§15/§18).

    Each worker sees exactly one physical GPU; stdout/stderr are captured
    per shard; any nonzero exit raises with the failing shard's stderr
    tail (BLOCKING -> STOP).
    """
    (root / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for index, (command, env) in enumerate(zip(command_per_shard, env_per_shard)):
        stdout_path = root / "logs" / "{}_rank{}_stdout.log".format(tag, index)
        stderr_path = root / "logs" / "{}_rank{}_stderr.log".format(tag, index)
        stdout_handle = open(stdout_path, "w", encoding="utf-8")
        stderr_handle = open(stderr_path, "w", encoding="utf-8")
        procs.append(
            (
                index,
                subprocess.Popen(
                    command, env=env, stdout=stdout_handle, stderr=stderr_handle
                ),
                stdout_handle,
                stderr_handle,
            )
        )
    for index, proc, stdout_handle, stderr_handle in procs:
        returncode = proc.wait()
        stdout_handle.close()
        stderr_handle.close()
        if returncode != 0:
            stderr_tail = (
                root / "logs" / "{}_rank{}_stderr.log".format(tag, index)
            ).read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                "{} shard {} failed (exit {}):\n{}".format(
                    tag, index, returncode, stderr_tail
                )
            )


def _run_sharded_nll(
    base_command: List[str],
    expected_records: Sequence[Dict[str, Any]],
    root: Path,
    tag: str,
    plan: Dict[str, Any],
) -> None:
    """S2 teacher-search NLL across sample shards, then merge and verify
    (spec §13: merged ids == expected ids, duplicates=0, missing=0).

    Sharding slices SAMPLES only; the per-sample candidate sets come from
    the shared selections file, so every sample's teacher candidate space
    is identical to single-GPU (spec §14).
    """
    commands = [
        base_command
        + ["--num-shards", str(plan["world_size"]), "--shard-index", str(index)]
        for index in range(plan["world_size"])
    ]
    _run_shards(
        commands,
        [_worker_env(plan, index) for index in range(plan["world_size"])],
        root,
        tag,
    )
    output = base_command[base_command.index("--output") + 1]
    expected_ids = [_record_id(record) for record in expected_records]
    merged = merge_partial_maps(
        [partial_path(output, index) for index in range(plan["world_size"])],
        expected_ids,
    )
    _write_json(output, merged)


def _merge_feature_shards(
    output_path: Path,
    records: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
) -> Path:
    """Merge per-shard feature payloads into the canonical features file
    (spec §15) and verify the union equals the expected sample ids exactly.

    ``output_path`` is the canonical target the workers were launched with
    (their ``--output``), so partials are partial_path(output_path, rank),
    i.e. train_features.json.rank{index} — the same convention
    ``_run_sharded_nll`` relies on; the caller's log tag never enters the
    file naming.

    Per-sample features are deterministic (frozen CLIP), so the merged
    payload is identical to the single-GPU payload up to the recomputed
    provenance hashes; the query_encoder provenance comes from shard 0.
    """
    partials = [
        Path(partial_path(str(output_path), index))
        for index in range(plan["world_size"])
    ]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in partials]
    records_out = {}
    for payload in payloads:
        records_out.update(payload["records"])
    expected_ids = [_record_id(record) for record in records]
    if sorted(records_out) != sorted(expected_ids):
        missing = sorted(set(expected_ids) - set(records_out))
        foreign = sorted(set(records_out) - set(expected_ids))
        raise ValueError(
            "feature shard merge mismatch: {} missing, {} foreign".format(
                len(missing), len(foreign)
            )
        )
    payload = {
        "schema_version": payloads[0]["schema_version"],
        "feature_source": payloads[0]["feature_source"],
        "query_encoder_provenance": payloads[0]["query_encoder_provenance"],
        "query_encoder_hash": payloads[0]["query_encoder_hash"],
        "feature_hash": _stable_hash(
            {
                sample_id: (record["visual_feature"], record["text_feature"])
                for sample_id, record in sorted(records_out.items())
            }
        ),
        "query_hash": _stable_hash(
            {
                sample_id: record["query"]
                for sample_id, record in sorted(records_out.items())
            }
        ),
        "records": records_out,
    }
    _write_json(str(output_path), payload)
    return output_path


def _run_sharded_features(
    base_command: List[str],
    records: Sequence[Dict[str, Any]],
    root: Path,
    tag: str,
    plan: Dict[str, Any],
) -> None:
    commands = [
        base_command
        + ["--num-shards", str(plan["world_size"]), "--shard-index", str(index)]
        for index in range(plan["world_size"])
    ]
    _run_shards(
        commands,
        [_worker_env(plan, index) for index in range(plan["world_size"])],
        root,
        tag,
    )
    output_path = Path(base_command[base_command.index("--output") + 1])
    _merge_feature_shards(output_path, records, plan)


def _write_distributed_training_contract(
    root: Path,
    task_id: int,
    config: Dict[str, Any],
    n_samples: int,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Spec §6: the 4-GPU training contract with equality assertions.

    Single-GPU reference (frozen protocol): ``per_device x grad_accum x 1``
    with SUM gradient accumulation (HF Trainer 4.33 backpropagates the
    undivided micro-batch loss). 4-GPU: ``per_device x grad_accum' x
    world_size`` with the same global batch; ComposeTrainer scales the
    loss by world_size so the DDP-averaged accumulated gradient is exactly
    the single-GPU gradient (no LR change, spec §5). The optimizer step
    counts are recomputed for the ACTUAL manifest size (including
    DistributedSampler divisibility padding, spec §7) and asserted equal;
    any mismatch raises (STOP).
    """
    per_device = int(config["training"]["per_device_train_batch_size"])
    accum_single = int(config["training"]["gradient_accumulation_steps"])
    epochs = int(config["training"]["num_train_epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    warmup_ratio = float(config["training"]["warmup_ratio"])
    scheduler = str(config["training"]["lr_scheduler_type"])
    seed = int(config["data"]["seed"])
    world = int(plan["world_size"])
    if per_device != 1:
        raise ValueError(
            "training contract recomposition requires per_device_train_batch_size "
            "== 1; got {}".format(per_device)
        )
    if accum_single % world != 0:
        raise ValueError(
            "gradient_accumulation_steps {} not divisible by world_size {}; "
            "global batch cannot be preserved".format(accum_single, world)
        )
    accum_four = accum_single // world
    global_batch_single = per_device * accum_single * 1
    global_batch_four = per_device * accum_four * world
    # DistributedSampler: every rank draws ceil(n/world) samples; the
    # padding repeats are explicit and recorded (§7).
    n_rank = int(math.ceil(n_samples / world))
    padding = n_rank * world - n_samples
    steps_single = int(math.ceil(n_samples / global_batch_single))
    steps_four = int(math.ceil(n_rank / accum_four))
    total_single = steps_single * epochs
    total_four = steps_four * epochs
    warmup_single = int(math.ceil(warmup_ratio * total_single))
    warmup_four = int(math.ceil(warmup_ratio * total_four))
    assertions = {
        "global_batch_equal": global_batch_single == global_batch_four,
        "steps_per_epoch_equal": steps_single == steps_four,
        "total_optimizer_steps_equal": total_single == total_four,
        "epochs_equal": True,
        "learning_rate_equal": True,
        "warmup_ratio_equal": warmup_ratio == float(config["training"]["warmup_ratio"]),
        "scheduler_equal": True,
        "seed_equal": True,
    }
    if not all(assertions.values()):
        raise RuntimeError(
            "distributed training contract violated (STOP): {}".format(assertions)
        )
    contract = {
        "task_id": task_id,
        "execution_mode": "4gpu_torchrun_ddp",
        "samples": n_samples,
        "single_gpu_reference": {
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": accum_single,
            "world_size": 1,
            "global_batch": global_batch_single,
            "steps_per_epoch": steps_single,
            "total_optimizer_steps": total_single,
            "warmup_steps": warmup_single,
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": scheduler,
            "seed": seed,
        },
        "four_gpu": {
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": accum_four,
            "world_size": world,
            "global_batch": global_batch_four,
            "steps_per_epoch": steps_four,
            "total_optimizer_steps": total_four,
            "warmup_steps": warmup_four,
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": scheduler,
            "seed": seed,
        },
        "distributed_sampler": {
            "samples_per_rank": n_rank,
            "divisibility_padding": padding,
            "note": "padding repeats DistributedSampler's last samples on "
                    "the tail ranks' final shards; recorded explicitly (spec §7)",
        },
        "gradient_equivalence": {
            "mechanism": "HF Trainer 4.33 SUM accumulation + DDP average + "
                         "ComposeTrainer loss x world_size scaling",
            "lora_learning_rate_unchanged": True,
            "key_learning_unchanged": True,
        },
        "assertions": assertions,
    }
    _write_json(str(root / "distributed_training_contract.json"), contract)
    return contract


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.isfile(path):
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}

        def merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merge(base[key], value)
                else:
                    base[key] = value

        merge(config, loaded)
    config["config_hash"] = _stable_hash(
        {
            "seed": config["data"]["seed"],
            "teacher": config["teacher"],
            "residual": config["residual"],
            "router": config["router"],
            "clustering": config["clustering"],
            "key_learning": config["key_learning"],
            "rms": config["rms"],
            "lora": config["lora"],
            "training": config["training"],
        }
    )
    return config


def _task_def(config: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    for entry in config["task_sequence"]:
        if int(entry["task_id"]) == int(task_id):
            return entry
    raise KeyError("task {} not in task_sequence".format(task_id))


def _data_hash(train_path: str, test_path: str, seed: int) -> str:
    return _stable_hash(
        {
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "seed": int(seed),
        }
    )


def _load_or_create_state(root: Path, task_id: int, task_name: str):
    state_dir = root / "state"
    state_path = state_dir / "task_state.json"
    registry_path = state_dir / "expert_registry.json"
    machine = (
        TaskStateMachine.load(str(state_path))
        if state_path.is_file()
        else TaskStateMachine(int(task_id), str(task_name))
    )
    registry = (
        ExpertRegistry.load_json(str(registry_path))
        if registry_path.is_file()
        else ExpertRegistry()
    )
    return machine, registry


def _query_encoder_from_checkpoint(path: str) -> ComposeQueryEncoder:
    info = load_query_encoder_checkpoint(path)
    encoder = ComposeQueryEncoder(
        visual_dim=int(info["visual_dim"]),
        text_dim=int(info["text_dim"]),
        query_dim=int(info["query_dim"]),
        seed=int(info["init_seed"]),
        initialize=True,
    )
    load_query_encoder_checkpoint(path, encoder)
    return encoder


def _load_prev_router(prev_root: Path, task_id: int) -> Optional[ComposeRouter]:
    """The historical router (old keys, frozen query encoder) from the
    previous task-boundary snapshot, or None for task 0."""
    if task_id <= 0:
        return None
    snapshot = ComposeSnapshot.load(
        str(prev_root / "snapshots" / "task{}".format(task_id - 1))
    )
    if snapshot.router_path is None:
        return None
    router = ComposeRouter()
    load_compose_router_checkpoint(snapshot.router_path, router)
    return router


def _working_router(
    root: Path,
    prev_root: Path,
    task_id: int,
    config: Dict[str, Any],
) -> ComposeRouter:
    """Router used by the current task: historical keys (task > 0) plus the
    learned new keys once S7 completed (resume-safe: reloads the S7
    checkpoint when key learning already ran)."""
    if task_id > 0:
        prev = _load_prev_router(prev_root, task_id)
        if prev is not None:
            router = prev
        else:
            router = ComposeRouter(
                query_encoder=_query_encoder_from_checkpoint(
                    str(root / QUERY_ENCODER_NAME)
                ),
                top_m=int(config["router"]["top_m"]),
                max_active_experts=int(config["router"]["max_active_experts"]),
                tau_none=float(config["router"]["tau_none"]),
                tau_second=float(config["router"]["tau_second"]),
                seed=int(config["data"]["seed"]),
            )
    else:
        router = ComposeRouter(
            query_encoder=_query_encoder_from_checkpoint(
                str(root / QUERY_ENCODER_NAME)
            ),
            top_m=int(config["router"]["top_m"]),
            max_active_experts=int(config["router"]["max_active_experts"]),
            tau_none=float(config["router"]["tau_none"]),
            tau_second=float(config["router"]["tau_second"]),
            seed=int(config["data"]["seed"]),
        )
    keys_path = root / "keys" / ROUTER_KEYS_NAME
    if keys_path.is_file():
        load_compose_router_checkpoint(str(keys_path), router)
    return router


def _prev_pool_dir(prev_root: Path, task_id: int) -> Optional[Path]:
    if task_id <= 0:
        return None
    snapshot = ComposeSnapshot.load(
        str(prev_root / "snapshots" / "task{}".format(task_id - 1))
    )
    declared = snapshot.manifest.get("pool_checkpoint_dir")
    if declared and (Path(declared) / "compose_experts.json").is_file():
        return Path(declared)
    raise RuntimeError(
        "no pool checkpoint dir resolvable from previous snapshot {}".format(
            snapshot.directory
        )
    )


#: Decoder layer count of llava-v1.5-7b (the only supported backbone).
_LLAVA_LAYER_COUNT = 32


def _decoder_layer_names() -> List[str]:
    """All Compose decoder projection names (matches the checkpoint layer
    validator regex) for the base-only checkpoint manifest."""
    names = []
    for depth in range(_LLAVA_LAYER_COUNT):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append("model.layers.{}.self_attn.{}".format(depth, projection))
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.append("model.layers.{}.mlp.{}".format(depth, projection))
    return names


def _ensure_base_only(root: Path) -> Path:
    """Base-only checkpoint (zero experts): the current system for task 0 is
    the frozen backbone. The manifest lists every decoder projection so the
    loader's layer validation passes; the weight file is an empty state."""
    base_dir = root / "base_only"
    manifest_path = base_dir / "compose_experts.json"
    if manifest_path.is_file():
        return base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save({}, str(base_dir / "compose_experts.bin"))
    manifest = {
        "format_version": 1,
        "adapter": {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "layers": _decoder_layer_names(),
        },
        "experts": [],
        "metrics": {
            "adapter_tensor_count": 0,
            "adapter_parameter_count": 0,
            "checkpoint_bytes": 0,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return base_dir


def _residual_pool_dir(root: Path) -> Optional[Path]:
    """The committed pool dir of this task (covers resume: derived from the
    formation manifest instead of in-memory state)."""
    formation_path = root / "cluster" / "formation.json"
    if not formation_path.is_file():
        return None
    payload = load_expert_formation(str(formation_path))
    ids = sorted(int(e["expert_id"]) for e in payload["formed_experts"])
    if not ids:
        return None
    candidate = root / "committed" / "pool_{:04d}".format(ids[-1])
    if (candidate / "compose_experts.json").is_file():
        return candidate
    return None


def _build_features_matrix(
    features: Dict[str, Any], sample_ids: Sequence[str]
) -> torch.Tensor:
    records = features["records"]
    rows = []
    for sample_id in sample_ids:
        if sample_id not in records:
            raise KeyError(
                "missing query features for sample {}".format(sample_id)
            )
        rows.append(records[sample_id]["query"])
    return torch.tensor(rows, dtype=torch.float32)


def _nll_selections(records: Sequence[Dict[str, Any]], active_ids: Sequence[int]):
    """Round-1 NLL selections: empty + every single over the visible pool
    (used when no per-sample Top-M retrieval is available)."""
    rows = {}
    for record in records:
        sample_id = _record_id(record)
        row = {"empty": []}
        for expert_id in active_ids:
            row["single_{}".format(expert_id)] = [expert_id]
        rows[sample_id] = row
    return rows


def run_task(
    root: Path,
    prev_root: Path,
    task_id: int,
    gpus: str,
    config: Dict[str, Any],
    config_path: Optional[str] = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    task_def = _task_def(config, task_id)
    task_name = task_def["name"]
    train_path = task_def["train_instructions"]
    test_path = task_def["test_instructions"]
    seed = int(config["data"]["seed"])
    config_hash = str(config["config_hash"])
    data_hash = _data_hash(train_path, test_path, seed)
    empty_pool = task_id == 0
    machine, registry = _load_or_create_state(root, task_id, task_name)
    plan = _execution_plan(gpus)
    env = _worker_env(plan, 0)
    if plan["mode"] == "4gpu":
        print(
            "execution plan: 4-GPU distributed (GPUs {}) — task-level "
            "strictly sequential, same-task stages sharded per spec §1-§29".format(
                ",".join(plan["gpus"])
            )
        )

    # ---- S0: snapshot load (task > 0) / cold start (task 0) ---------------
    if not _stage_done(root, "s0_snapshot_load"):
        if task_id > 0:
            prev_snapshot = ComposeSnapshot.load(
                str(prev_root / "snapshots" / "task{}".format(task_id - 1))
            )
            if prev_snapshot.manifest["task_id"] != task_id - 1:
                raise ValueError(
                    "prev snapshot task_id {} != expected {}".format(
                        prev_snapshot.manifest["task_id"], task_id - 1
                    )
                )
            if not registry.get_active_experts():
                registry.load_state_dict(prev_snapshot.registry.state_dict())
                registry.save_json(str(root / "state" / "expert_registry.json"))
            _write_json(
                str(root / "data" / "prev_snapshot_info.json"),
                {
                    "pool_version": registry.pool_version,
                    "active_expert_ids": [
                        expert.expert_id for expert in registry.get_active_experts()
                    ],
                    "pool_checkpoint_dir": prev_snapshot.manifest.get(
                        "pool_checkpoint_dir"
                    ),
                    "git_commit": prev_snapshot.manifest["git_commit"],
                    "has_router": prev_snapshot.router_path is not None,
                },
            )
        else:
            _write_json(
                str(root / "data" / "cold_start.json"),
                {"note": "task 0: empty registry is a legal cold start"},
            )
        _advance(root, machine, TaskStage.DATA_READY, note="snapshot loaded")
        _mark_stage(root, "s0_snapshot_load")

    records_all = json.load(open(train_path, "r", encoding="utf-8"))
    train_limit = int(config["tasks"]["teacher_search_train_samples"])
    val_limit = int(config["tasks"]["teacher_search_validation_samples"])
    # UCIT train files repeat question_ids (VizWiz/Flickr30k/CLEVR); a
    # unique internal id per record keeps features/selections/residuals/
    # clusters/manifest keyed consistently (see _assign_unique_record_ids).
    records_all = _assign_unique_record_ids(records_all)
    teacher_train = records_all[:train_limit]
    teacher_val = records_all[train_limit: train_limit + val_limit]

    # ---- S1: functional query features (frozen, deterministic) ------------
    if not _stage_done(root, "s1_features"):
        _write_json(str(root / "data" / "teacher_train.json"), teacher_train)
        _write_json(str(root / "data" / "teacher_val.json"), teacher_val)
        encoder_path = root / QUERY_ENCODER_NAME
        query_encoder_args = []
        if encoder_path.is_file():
            query_encoder_args = ["--query-encoder", str(encoder_path)]
        for tag in ("train", "val"):
            feature_command = [
                PYTHON, "-m", "compose.eval.query_features",
                "--questions", str(root / "data" / "teacher_{}.json".format(tag)),
                "--images", IMAGE_FOLDER,
                "--output", str(root / "features" / "{}_features.json".format(tag)),
                "--seed", str(seed),
                "--device", "cuda:0",
            ] + query_encoder_args
            if plan["mode"] == "4gpu":
                # Spec §15: sample-sharded workers (one physical GPU each),
                # merged and verified on the orchestrator.
                _run_sharded_features(
                    feature_command,
                    teacher_train if tag == "train" else teacher_val,
                    root,
                    "s1_{}".format(tag),
                    plan,
                )
            else:
                _run(feature_command, env, root, "s1_{}".format(tag))
        # Persist the deterministic task-0 query encoder for every later task
        # (frozen; never re-randomized).
        if not encoder_path.is_file():
            encoder = ComposeQueryEncoder(seed=seed)
            save_query_encoder_checkpoint(str(encoder_path), encoder)
        _advance(root, machine, TaskStage.QUERY_READY, note="query features done")
        _mark_stage(root, "s1_features")

    # ---- S2: answer-supervised teacher search over the per-sample Top-M ---
    if not _stage_done(root, "s2_teacher"):
        active_ids = [
            metadata.expert_id for metadata in registry.get_active_experts()
        ]
        prev_router = _load_prev_router(prev_root, task_id)
        teacher_records = {"train": [], "val": []}
        for tag, split in (("train", "train"), ("val", "val")):
            subset = teacher_train if tag == "train" else teacher_val
            sample_ids = [_record_id(record) for record in subset]
            if prev_router is not None:
                features = json.loads(
                    (root / "features" / "{}_features.json".format(tag)).read_text()
                )
                queries = _build_features_matrix(features, sample_ids)
                retrieval = prev_router.retrieve(
                    queries, list(prev_router.expert_ids)
                )
                top_m_rows = retrieval.expert_ids.tolist()
                retrieved_by_sample = {
                    sample_id: tuple(
                        int(value) for value in row if int(value) != -1
                    )
                    for sample_id, row in zip(sample_ids, top_m_rows)
                }
            else:
                retrieved_by_sample = {
                    sample_id: () for sample_id in sample_ids
                }
            if task_id == 0:
                # Cold start: no historical experts to search; evaluate the
                # base NLL (the current system is the frozen backbone).
                selections = {
                    sample_id: {"empty": []} for sample_id in sample_ids
                }
            else:
                selections = {
                    sample_id: {"empty": []}
                    for sample_id, retrieved in retrieved_by_sample.items()
                }
                for sample_id, retrieved in retrieved_by_sample.items():
                    for expert_id in retrieved:
                        selections[sample_id]["single_{}".format(expert_id)] = [
                            expert_id
                        ]
            _write_json(
                str(root / "data" / "{}_selections_r1.json".format(tag)),
                selections,
            )
            scoring_pool = (
                _prev_pool_dir(prev_root, task_id)
                if task_id > 0
                else _ensure_base_only(root)
            )
            nll_r1_command = [
                PYTHON, "-m", "compose.eval.nll_eval",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", PROJECTOR_PATH,
                "--checkpoint-dir", str(scoring_pool),
                "--question-file", str(root / "data" / "teacher_{}.json".format(tag)),
                "--image-folder", IMAGE_FOLDER,
                "--selections", str(root / "data" / "{}_selections_r1.json".format(tag)),
                "--output", str(root / "teacher" / "{}_nll_r1.json".format(tag)),
                "--device", "cuda:0",
            ]
            if plan["mode"] == "4gpu":
                # Spec §13/§14: sample shards only; candidate sets shared
                # verbatim; merge verifies 0 missing / 0 duplicates.
                _run_sharded_nll(nll_r1_command, subset, root, "s2_{}_r1".format(tag), plan)
            else:
                _run(nll_r1_command, env, root, "s2_{}_r1".format(tag))
            if task_id > 0:
                lambda_expert = float(config["teacher"]["lambda_expert"])
                top_k_for_pair = int(config["teacher"]["top_k_for_pair"])
                max_pairs = int(config["teacher"]["max_pairs"])
                nll_r1 = json.loads(
                    (root / "teacher" / "{}_nll_r1.json".format(tag)).read_text()
                )
                pair_selections = {}
                for sample_id, values in sorted(nll_r1.items()):
                    single_scores = []
                    for key, loss in values.items():
                        if key.startswith("single_"):
                            expert_id = int(key.split("_", 1)[1])
                            single_scores.append((loss + lambda_expert, expert_id))
                    single_scores.sort()
                    best = single_scores[:top_k_for_pair]
                    pairs = []
                    for left in range(len(best)):
                        for right in range(left + 1, len(best)):
                            pairs.append((best[left][1], best[right][1]))
                            if len(pairs) >= max_pairs:
                                break
                        if len(pairs) >= max_pairs:
                            break
                    if pairs:
                        pair_selections[sample_id] = {
                            "pair_{}_{}".format(a, b): [a, b]
                            for a, b in pairs
                        }
                _write_json(
                    str(root / "data" / "{}_selections_r2.json".format(tag)),
                    pair_selections,
                )
                nll_r2_command = [
                    PYTHON, "-m", "compose.eval.nll_eval",
                    "--model-path", BASE_MODEL,
                    "--vision-tower", VISION_TOWER,
                    "--projector-path", PROJECTOR_PATH,
                    "--checkpoint-dir", str(scoring_pool),
                    "--question-file", str(root / "data" / "teacher_{}.json".format(tag)),
                    "--image-folder", IMAGE_FOLDER,
                    "--selections", str(root / "data" / "{}_selections_r2.json".format(tag)),
                    "--output", str(root / "teacher" / "{}_nll_r2.json".format(tag)),
                    "--device", "cuda:0",
                ]
                if plan["mode"] == "4gpu":
                    _run_sharded_nll(nll_r2_command, subset, root, "s2_{}_r2".format(tag), plan)
                else:
                    _run(nll_r2_command, env, root, "s2_{}_r2".format(tag))

            nll_r1 = json.loads(
                (root / "teacher" / "{}_nll_r1.json".format(tag)).read_text()
            )
            nll_r2 = json.loads(
                (root / "teacher" / "{}_nll_r2.json".format(tag)).read_text()
            ) if task_id > 0 else {}
            searcher = ComposeTeacherSearcher(
                config=OracleConfig(
                    oracle_name="compose_teacher",
                    composition_mode="rms_calibrated",
                    lambda_expert=float(config["teacher"]["lambda_expert"]),
                    delta_pair_raw=float(config["teacher"]["delta_pair_raw"]),
                    top_k_for_pair=int(config["teacher"]["top_k_for_pair"]),
                    max_pairs=int(config["teacher"]["max_pairs"]),
                ),
                router_version=ROUTER_VERSION,
                pool_version=registry.pool_version,
                top_m=int(config["router"]["top_m"]),
            )
            provenance = {
                "router_version": ROUTER_VERSION,
                "pool_version": registry.pool_version,
                "dataset_hash": data_hash,
                "config_hash": config_hash,
                "seed": seed,
            }
            records_out = []
            for index, record in enumerate(subset):
                sample_id = sample_ids[index]
                values = nll_r1.get(sample_id, {})
                nll_by_set = {(): float(values.get("empty", 0.0))}
                for key, loss in values.items():
                    if key.startswith("single_"):
                        nll_by_set[(int(key.split("_", 1)[1]),)] = float(loss)
                for key, loss in nll_r2.get(sample_id, {}).items():
                    if key.startswith("pair_"):
                        left, right = key.split("_")[1:3]
                        nll_by_set[(int(left), int(right))] = float(loss)
                if task_id == 0:
                    empty_loss = float(values["empty"])
                    record_out = {
                        "sample_id": sample_id,
                        "task_id": task_id,
                        "pool_version": registry.pool_version,
                        "router_version": ROUTER_VERSION,
                        "candidate_experts": [],
                        "retrieved_top_m": [],
                        "empty_loss": empty_loss,
                        "single_losses": {},
                        "pair_losses": {},
                        "best_single": [],
                        "best_pair": [],
                        "pair_gain": None,
                        "teacher_set": [],
                        "teacher_loss": empty_loss,
                        "teacher_multi_hot": {},
                        "cache_key": _stable_hash(
                            {"task_id": task_id, "sample_id": sample_id, "seed": seed}
                        ),
                    }
                else:
                    teacher = searcher.search_from_nll(
                        sample_id=sample_id,
                        task_id=task_id,
                        retrieved_top_m=retrieved_by_sample[sample_id],
                        nll_by_set=nll_by_set,
                        pool_size=len(active_ids),
                        provenance=provenance,
                    )
                    record_out = teacher.to_dict()
                    record_out["teacher_multi_hot"] = build_teacher_multi_hot(
                        record_out["teacher_set"], active_ids
                    )
                records_out.append(record_out)
            teacher_records[split] = records_out
        _write_json(
            str(root / "teacher" / "teacher_records_train.json"),
            teacher_records["train"],
        )
        _write_json(
            str(root / "teacher" / "teacher_records_val.json"),
            teacher_records["val"],
        )
        _write_json(
            str(root / "teacher" / "summary.json"),
            {
                "task_id": task_id,
                "empty_pool": empty_pool,
                "train_samples": len(teacher_records["train"]),
                "val_samples": len(teacher_records["val"]),
                "train_teacher_empty_count": sum(
                    1 for record in teacher_records["train"] if not record["teacher_set"]
                ),
                "train_teacher_single_count": sum(
                    1 for record in teacher_records["train"] if len(record["teacher_set"]) == 1
                ),
                "train_teacher_pair_count": sum(
                    1 for record in teacher_records["train"] if len(record["teacher_set"]) == 2
                ),
            },
        )
        if task_id == 0:
            machine.skip_with_reason(
                TaskStage.OLD_TEACHER_READY,
                "task 0 cold start: no historical experts to search",
            )
        else:
            _advance(
                root, machine, TaskStage.OLD_TEACHER_READY,
                note="teacher search done",
            )
        _mark_stage(root, "s2_teacher")

    # ---- S3: residual split (old_teacher_loss > tau_res) ------------------
    if not _stage_done(root, "s3_residual"):
        tau_res = float(config["residual"]["tau_res"])
        for tag, split in (("train", "train"), ("val", "val")):
            raw = json.loads(
                (root / "teacher" / "teacher_records_{}.json".format(tag)).read_text()
            )
            coverage = []
            for record in raw:
                teacher_set = [int(value) for value in record["teacher_set"]]
                retrieved = set(int(value) for value in record["retrieved_top_m"])
                coverage.append(
                    not teacher_set
                    or all(expert_id in retrieved for expert_id in teacher_set)
                )
            reuse, residual = build_residual_records(
                raw,
                tau_res,
                split=split,
                top_m_covered=coverage,
                empty_pool=empty_pool,
            )
            if split == "train":
                # write_residual_split(root) writes into root/residual/
                # (it appends the "residual" segment itself).
                summary = write_residual_split(
                    root,
                    reuse,
                    residual,
                    tau_res,
                    {
                        "task_id": task_id,
                        "min_residual_samples": int(
                            config["residual"]["min_residual_samples"]
                        ),
                        "should_create_experts": should_create_experts(
                            len(
                                [
                                    record
                                    for record in residual
                                    if not record.retrieval_diagnostic
                                ]
                            ),
                            int(config["residual"]["min_residual_samples"]),
                        ),
                    },
                )
                # Training file must mirror the S5 clustering set: only the
                # non-diagnostic residual samples (retrieval_diagnostic records
                # never produce cluster experts, so their training data must
                # not be fed either — ComposeSelectionDataset requires a
                # manifest row for every sample).
                _write_json(
                    str(root / "data" / "residual_train.json"),
                    [
                        record
                        for record in json.loads(
                            (root / "data" / "teacher_train.json").read_text()
                        )
                        if _record_id(record)
                        in {
                            record.sample_id
                            for record in residual
                            if not record.retrieval_diagnostic
                        }
                    ],
                )
            else:
                _write_json(
                    str(root / "residual" / "val_reuse.json"),
                    [record.to_dict() for record in reuse],
                )
                _write_json(
                    str(root / "residual" / "val_residual.json"),
                    [record.to_dict() for record in residual],
                )
        _advance(root, machine, TaskStage.RESIDUAL_READY, note="residual split done")
        _mark_stage(root, "s3_residual")

    # ---- S4: recall audit (diagnostic only) -------------------------------
    if not _stage_done(root, "s4_recall_audit"):
        raw = json.loads(
            (root / "teacher" / "teacher_records_train.json").read_text()
        )
        active_ids = [
            metadata.expert_id for metadata in registry.get_active_experts()
        ]
        audit_ratio = float(config["teacher"].get("min_recall_audit_ratio", 0.05))
        audit_count = max(1, int(len(raw) * audit_ratio))
        audited = raw[: max(1, min(audit_count, len(raw)))]
        by_id = {record["sample_id"]: record for record in audited}
        retrieved_sets = {
            record["sample_id"]: tuple(
                int(value) for value in record["retrieved_top_m"]
            )
            for record in audited
        }
        audit = run_recall_audit(
            samples=[{"sample_id": record["sample_id"]} for record in audited],
            full_pool_eval=lambda sample: tuple(
                int(value)
                for value in by_id.get(sample["sample_id"], {}).get(
                    "teacher_set", ()
                )
            ),
            retrieved_sets=retrieved_sets,
            top_m=int(config["router"]["top_m"]),
            pool_expert_ids=active_ids,
        )
        audit["note"] = (
            "diagnostic only; recall misses are never wrapped into "
            "'need a new capability expert'"
        )
        _write_json(str(root / "teacher" / "recall_audit.json"), audit)
        _mark_stage(root, "s4_recall_audit")

    # ---- S5: clustering (spherical K-means, dynamic K) --------------------
    formed = []
    training_manifest = []
    no_expansion = False
    if not _stage_done(root, "s5_clustering"):
        residual = [
            record
            for record in json.loads(
                (root / "residual" / "residual.json").read_text()
            )
            if not bool(record["retrieval_diagnostic"])
        ]
        min_residual = int(config["residual"]["min_residual_samples"])
        if not should_create_experts(len(residual), min_residual):
            _write_json(
                str(root / "cluster" / "no_expansion.json"),
                {
                    "reason": "insufficient_residual",
                    "residual_train_count": len(residual),
                    "min_residual_samples": min_residual,
                },
            )
            no_expansion = True
        else:
            features = json.loads(
                (root / "features" / "train_features.json").read_text()
            )
            sample_ids = [record["sample_id"] for record in residual]
            queries = _build_features_matrix(features, sample_ids)
            cluster_config = ComposeClusteringConfig(**config["clustering"])
            result = cluster_residual_queries(
                queries,
                sample_ids,
                cluster_config,
                task_id=task_id,
                query_hash=features.get("query_hash", ""),
            )
            write_cluster_manifest(
                result,
                str(root / "cluster" / "clusters.json"),
                extra={"query_hash": features.get("query_hash", "")},
            )
            formed, noise_ids = form_cluster_experts(
                result,
                registry,
                creation_task=task_id,
                key_mode=str(config["router"]["key_mode"]),
            )
            training_manifest = build_training_manifest(
                formed,
                [
                    {
                        "sample_id": record["sample_id"],
                        "old_teacher_set": record["old_teacher_set"],
                    }
                    for record in residual
                ],
            )
            write_expert_formation(
                formed,
                training_manifest,
                str(root / "cluster" / "formation.json"),
                extra={
                    "task_id": task_id,
                    "query_hash": features.get("query_hash", ""),
                    "noise_sample_ids": noise_ids,
                },
            )
            _write_json(
                str(root / "cluster" / "assignment_stats.json"),
                {
                    "selected_k": result.selected_k,
                    "selected_silhouette": result.selected_silhouette,
                    "silhouette_by_k": result.silhouette_by_k,
                    "noise_sample_ids": noise_ids,
                    "cluster_sizes": [
                        cluster.size for cluster in result.clusters
                    ],
                },
            )
        _mark_stage(root, "s5_clustering")

    if _stage_done(root, "s5_clustering") and not no_expansion:
        formation_path = root / "cluster" / "formation.json"
        if formation_path.is_file():
            payload = load_expert_formation(str(formation_path))
            formed = _formed_from_payload(payload)
            training_manifest = payload["training_manifest"]
        else:
            no_expansion = True

    if not _stage_done(root, "s5_advance"):
        if no_expansion or not formed:
            _advance(
                root, machine, TaskStage.NO_EXPANSION_REQUIRED,
                note="insufficient residual or no valid clusters",
            )
        else:
            _advance(
                root, machine, TaskStage.CLUSTERS_READY,
                note="{} cluster experts formed".format(len(formed)),
            )
        _mark_stage(root, "s5_advance")

    cluster_expert_ids = sorted(expert.expert_id for expert in formed)

    # ---- S6: cluster-wise conditional residual LoRA training --------------
    if cluster_expert_ids and not _stage_done(root, "s6_cluster_training"):
        selection_manifest = training_manifest
        _write_json(
            str(root / "cluster" / "selection_manifest.json"),
            selection_manifest,
        )
        # Training data must match the manifest exactly: samples that
        # clustering assigned to the noise cluster have no manifest row and
        # no cluster expert, so ComposeSelectionDataset would raise KeyError
        # on them. The S3 file already excluded retrieval_diagnostic records;
        # this rewrite is authoritative over the final cluster assignment.
        manifest_sample_ids = {
            str(row["sample_id"]) for row in training_manifest
        }
        _write_json(
            str(root / "data" / "residual_train.json"),
            [
                record
                for record in json.loads(
                    (root / "data" / "teacher_train.json").read_text()
                )
                if _record_id(record) in manifest_sample_ids
            ],
        )
        lora_dir = root / "lora" / "cluster_training"
        lora_dir.mkdir(parents=True, exist_ok=True)
        _advance(
            root, machine, TaskStage.CLUSTER_EXPERTS_TRAINING,
            note="cluster LoRA training started",
        )
        train_command = [
            PYTHON, "-m", "compose.train.train_compose",
            "--model_name_or_path", BASE_MODEL,
            "--vision_tower", VISION_TOWER,
            "--pretrain_mm_mlp_adapter", PROJECTOR_PATH,
            "--version", "v1",
            "--data_path", str(root / "data" / "residual_train.json"),
            "--image_folder", IMAGE_FOLDER,
            "--compose-mode", "cluster_expert",
            "--compose-selection-manifest", str(root / "cluster" / "selection_manifest.json"),
            "--compose-cluster-expert-ids", ",".join(str(value) for value in cluster_expert_ids),
            "--output_dir", str(lora_dir),
            "--bf16", "True",
            "--tf32", "True",
            "--num_train_epochs", str(config["training"]["num_train_epochs"]),
            "--learning_rate", str(config["training"]["learning_rate"]),
            "--warmup_ratio", str(config["training"]["warmup_ratio"]),
            "--lr_scheduler_type", str(config["training"]["lr_scheduler_type"]),
            "--logging_steps", str(config["training"]["logging_steps"]),
            "--save_steps", str(config["training"]["save_steps"]),
            "--model_max_length", str(config["training"]["model_max_length"]),
            "--gradient_checkpointing", str(config["training"]["gradient_checkpointing"]),
            "--dataloader_num_workers", str(config["training"]["dataloader_num_workers"]),
            "--cache_dir", str(config["training"]["cache_dir"]),
            "--seed", str(seed),
            "--report_to", "none",
        ]
        if task_id > 0:
            prev_pool = _prev_pool_dir(prev_root, task_id)
            train_command += ["--compose-checkpoint", str(prev_pool)]
        if plan["mode"] == "4gpu":
            # Spec §4/§5/§6: same effective global batch
            # (per_device x grad_accum x world_size), same optimizer steps,
            # same LR/scheduler/epochs — asserted by the contract before
            # training starts (any violation raises: STOP).
            contract = _write_distributed_training_contract(
                root, task_id, config, len(training_manifest), plan
            )
            train_command = (
                _torchrun_launch(plan, train_command)
                + [
                    "--per_device_train_batch_size", "1",
                    "--gradient_accumulation_steps", str(
                        contract["four_gpu"]["gradient_accumulation_steps"]
                    ),
                    "--ddp_find_unused_parameters", "False",
                ]
            )
            _run(train_command, _all_gpu_env(plan), root, "s6_cluster_training")
        else:
            train_command += [
                "--per_device_train_batch_size", str(
                    config["training"]["per_device_train_batch_size"]
                ),
                "--gradient_accumulation_steps", str(
                    config["training"]["gradient_accumulation_steps"]
                ),
            ]
            _run(train_command, env, root, "s6_cluster_training")
        missing = [
            expert_id
            for expert_id in cluster_expert_ids
            if not (lora_dir / "expert_{:04d}.pt".format(expert_id)).is_file()
        ]
        if missing:
            raise RuntimeError(
                "cluster training did not produce expert state dicts for {}".format(
                    missing
                )
            )
        _advance(
            root, machine, TaskStage.CLUSTER_EXPERTS_TRAINED,
            note="cluster LoRA training done",
        )
        _mark_stage(root, "s6_cluster_training")

    # ---- S7: assemble + cluster-supervised key learning -------------------
    final_pool_dir: Optional[Path] = None
    if cluster_expert_ids and not _stage_done(root, "s7_keys"):
        prev_pool = _prev_pool_dir(prev_root, task_id)
        pool_dir = prev_pool
        for expert_id in cluster_expert_ids:
            output_dir = root / "committed" / "pool_{:04d}".format(expert_id)
            assemble_command = [
                PYTHON, "-m", "compose.eval.assemble_expert",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", PROJECTOR_PATH,
                "--expert-state-dict", str(root / "lora" / "cluster_training" / "expert_{:04d}.pt".format(expert_id)),
                "--expert-id", str(expert_id),
                "--output-dir", str(output_dir),
            ]
            if pool_dir is not None:
                assemble_command += ["--old-expert-checkpoint", str(pool_dir)]
            _run(assemble_command, env, root, "s7_assemble_{:04d}".format(expert_id))
            pool_dir = output_dir
        final_pool_dir = pool_dir
        # Load the assembly hashes for the key metadata binding.
        assembly = json.loads(
            (final_pool_dir / "assembly.json").read_text(encoding="utf-8")
        )
        bin_sha256 = str(assembly["compose_experts_bin_sha256"])
        router = _working_router(root, prev_root, task_id, config)
        payload = load_expert_formation(str(root / "cluster" / "formation.json"))
        # The manifest is the authoritative training set: clustering noise
        # samples have no cluster expert and no manifest row, so they must
        # not contribute queries, labels, or old-key negatives here.
        sample_to_expert = {
            row["sample_id"]: int(row["cluster_expert_id"])
            for row in payload["training_manifest"]
        }
        residual = [
            record
            for record in json.loads(
                (root / "residual" / "residual.json").read_text()
            )
            if not bool(record["retrieval_diagnostic"])
            and record["sample_id"] in sample_to_expert
        ]
        features = json.loads(
            (root / "features" / "train_features.json").read_text()
        )
        sample_ids = [record["sample_id"] for record in residual]
        queries = _build_features_matrix(features, sample_ids)
        label_map = {
            expert_id: index for index, expert_id in enumerate(cluster_expert_ids)
        }
        labels = torch.tensor(
            [label_map[sample_to_expert[sample_id]] for sample_id in sample_ids],
            dtype=torch.long,
        )
        # Historical Top-M keys of the residual samples (hard negatives).
        old_negative_keys = None
        if task_id > 0:
            old_router = _load_prev_router(prev_root, task_id)
            old_ids = sorted(
                {
                    int(expert_id)
                    for record in residual
                    for expert_id in record["retrieved_top_m"]
                }
            )
            if old_ids:
                old_negative_keys = old_router.key_store.normalized(old_ids)
        # Register the new expert keys (initialized from the centroids).
        for expert in formed:
            router.add_expert(
                expert.expert_id,
                creation_task=task_id,
                checkpoint_sha256=bin_sha256,
                key=torch.tensor(expert.centroid, dtype=torch.float32),
                key_initialization="centroid",
            )
        key_config = ComposeKeyLearningConfig(
            **config["key_learning"], key_mode=str(config["router"]["key_mode"])
        )
        _advance(
            root, machine, TaskStage.KEYS_TRAINING,
            note="key learning started",
        )
        if old_negative_keys is None:
            old_negative_keys = torch.zeros(0, queries.shape[1], dtype=torch.float32)
        else:
            old_negative_keys = old_negative_keys.to(torch.float32)
        results = learn_cluster_keys(
            queries,
            labels,
            old_negative_keys,
            router.key_store.keys,
            cluster_expert_ids,
            key_config,
            device="cpu",
        )
        _write_json(
            str(root / "keys" / "key_learning_results.json"),
            {str(key): value.to_dict() for key, value in results.items()},
        )
        save_compose_router_checkpoint(
            str(root / "keys" / ROUTER_KEYS_NAME),
            router,
            pool_version=registry.pool_version,
            config_hash=config_hash,
        )
        _write_json(
            str(root / "keys" / "assembly.json"),
            {"bin_sha256": bin_sha256, "pool_checkpoint_dir": str(final_pool_dir)},
        )
        _advance(root, machine, TaskStage.KEYS_READY, note="key learning done")
        _mark_stage(root, "s7_keys")

    if cluster_expert_ids:
        final_pool_dir = _residual_pool_dir(root)
        if final_pool_dir is None:
            raise RuntimeError(
                "cluster experts formed but no committed pool dir exists; "
                "resume the S7 stage first"
            )

    # ---- S8: direct commit (no validation gates) --------------------------
    if cluster_expert_ids and not _stage_done(root, "s8_commit"):
        if final_pool_dir is None:
            raise RuntimeError("no assembled pool dir before commit")
        bin_sha256 = _sha256_file(str(final_pool_dir / "compose_experts.bin"))
        router = _working_router(root, prev_root, task_id, config)
        # The key store metadata already records this exact hash: S7 created
        # it from assembly.json's compose_experts_bin_sha256, the sha256 of
        # this same bin (assemble_expert.py). No patch here — the metadata is
        # frozen by design, and recomputing the hash only feeds the commit
        # transaction below.
        # The final router checkpoint records the POST-commit pool version
        # (registry.pool_version + the experts about to commit), so
        # snapshot.load_router's pool-version validation matches.
        final_pool_version = registry.pool_version + len(cluster_expert_ids)
        save_compose_router_checkpoint(
            str(root / "keys" / ROUTER_FINAL_NAME),
            router,
            pool_version=final_pool_version,
            config_hash=config_hash,
            extra={"pool_version": final_pool_version},
        )
        router_sha256 = _sha256_file(str(root / "keys" / ROUTER_FINAL_NAME))
        payload = load_expert_formation(str(root / "cluster" / "formation.json"))
        formed_experts = _formed_from_payload(payload)
        transaction = CommitTransaction(
            root / "state", registry
        )
        committed = []
        for expert in formed_experts:
            metadata = build_cluster_expert_metadata(
                expert,
                task_id=task_id,
                task_name=task_name,
                seed=seed,
                checkpoint_path=str(final_pool_dir),
                checkpoint_sha256=bin_sha256,
                key_path=str(root / "keys" / ROUTER_FINAL_NAME),
                key_sha256=router_sha256,
                rms_stats_path=None,
                config_hash=config_hash,
                pool_version=registry.pool_version,
            )
            result = commit_cluster_expert(
                transaction,
                expert,
                metadata,
                artifacts={
                    str(final_pool_dir / "compose_experts.bin"): bin_sha256,
                    str(root / "keys" / ROUTER_FINAL_NAME): router_sha256,
                },
                condition_record={
                    "rule": "direct_cluster_commit",
                    "cluster_id": expert.cluster_id,
                    "cluster_size": expert.size,
                    "key_mode": expert.key_mode,
                },
            )
            committed.append(result)
        _write_json(
            str(root / "committed" / "commit_summary.json"),
            {"committed": committed, "pool_version": registry.pool_version},
        )
        _advance(
            root, machine, TaskStage.EXPERTS_COMMITTED,
            note="{} experts committed (direct)".format(len(committed)),
        )
        _mark_stage(root, "s8_commit")

    # Resolve the task's scoring pool dir: the committed pool after
    # expansion, the previous task's pool, or the base-only checkpoint
    # (task 0 cold start).
    if cluster_expert_ids:
        pool_dir = final_pool_dir or _residual_pool_dir(root)
        if pool_dir is None:
            raise RuntimeError(
                "no committed pool dir resolvable for task {}".format(task_id)
            )
    elif task_id > 0:
        pool_dir = _prev_pool_dir(prev_root, task_id)
    else:
        pool_dir = _ensure_base_only(root)

    # ---- S9: RMS statistics + runtime kappa -------------------------------
    if not _stage_done(root, "s9_rms"):
        if cluster_expert_ids:
            bin_sha256 = _sha256_file(str(pool_dir / "compose_experts.bin"))
            rms_output = root / "rms"
            rms_command = [
                PYTHON, "-m", "compose.eval.rms_stats",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", PROJECTOR_PATH,
                "--checkpoint-dir", str(pool_dir),
                "--question-file", str(root / "data" / "teacher_val.json"),
                "--image-folder", IMAGE_FOLDER,
                "--checkpoint-hash", bin_sha256,
                "--dataset-manifest-hash", data_hash,
                "--composition-config-hash", config_hash,
                "--output-dir", str(rms_output),
                "--device", "cuda:0",
            ]
            if plan["mode"] == "4gpu":
                # Spec §17: torchrun — each rank computes its sample shard,
                # the per-layer moments are all-reduced by exact fp64 sums,
                # rank 0 builds the kappa calibration and patches the
                # manifest (no averaging; single-vs-four parity is checked
                # by the smoke, spec §17).
                _run(
                    _torchrun_launch(plan, rms_command),
                    _all_gpu_env(plan),
                    root,
                    "s9_rms",
                )
            else:
                _run(rms_command, env, root, "s9_rms")
        else:
            _write_json(
                str(root / "rms" / "vacuous.json"),
                {"note": "no new experts; runtime kappa vacuous"},
            )
        _advance(root, machine, TaskStage.RMS_READY, note="RMS done")
        _mark_stage(root, "s9_rms")

    # ---- S10: task-boundary snapshot --------------------------------------
    if not _stage_done(root, "s10_snapshot"):
        snapshot_dir = root / "snapshots" / "task{}".format(task_id)
        # A crashed mid-write snapshot is never complete; clear it so
        # ``ComposeSnapshot.create`` can write atomically (manifest last).
        if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            shutil.rmtree(snapshot_dir)
        query_encoder = _query_encoder_from_checkpoint(
            str(root / QUERY_ENCODER_NAME)
        )
        router = _working_router(root, prev_root, task_id, config)
        calibration = None
        calibration_path = root / "rms" / "rms_calibration.json"
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        residual_manifest = None
        summary_path = root / "residual" / "summary.json"
        if summary_path.is_file():
            residual_manifest = json.loads(summary_path.read_text(encoding="utf-8"))
        formation_manifest = None
        formation_path = root / "cluster" / "formation.json"
        if formation_path.is_file():
            formation_manifest = json.loads(formation_path.read_text(encoding="utf-8"))
        ComposeSnapshot.create(
            directory=str(snapshot_dir),
            task_id=task_id,
            task_name=task_name,
            registry=registry,
            task_state=machine,
            git_commit=_git_commit(),
            command=" ".join(sys.argv),
            data_hash=data_hash,
            config_copy_path=config_path,
            query_encoder=query_encoder,
            router=router,
            router_config_hash=config_hash,
            rms_stats=None,
            calibration=calibration,
            residual_manifest=residual_manifest,
            formation_manifest=formation_manifest,
            pool_checkpoint_dir=str(pool_dir) if pool_dir is not None else None,
        )
        _advance(root, machine, TaskStage.SNAPSHOT_READY, note="snapshot written")
        _mark_stage(root, "s10_snapshot")

    # ---- S11: router-based evaluation -------------------------------------
    if not _stage_done(root, "s11_eval"):
        snapshot_dir = root / "snapshots" / "task{}".format(task_id)
        eval_output = root / "eval_output"
        router_checkpoint = snapshot_dir / "router_checkpoint.pt"
        eval_command = [
            PYTHON, "-m", "compose.eval.eval_task",
            "--adapter-kind", "compose",
            "--model-path", BASE_MODEL,
            "--vision-tower", VISION_TOWER,
            "--projector-path", PROJECTOR_PATH,
            "--question-file", test_path,
            "--image-folder", IMAGE_FOLDER,
            "--answers-file", str(eval_output / "answers.jsonl"),
            "--run-summary-file", str(eval_output / "run_summary.json"),
            "--device", "cuda:0",
            "--max-new-tokens", str(config["eval"]["max_new_tokens"]),
        ]
        if router_checkpoint.is_file():
            # Per-sample ComposeRouter.select(): frozen query encoder +
            # expert keys; no answers, no oracle, no task-id lookup, no
            # clustering at test time (spec §22).
            eval_command += [
                "--checkpoint-dir", str(pool_dir),
                "--router-checkpoint", str(router_checkpoint),
            ]
        else:
            # No router (degenerate smoke case): the frozen backbone (or
            # previous pool) is evaluated with an empty selection.
            eval_command += ["--checkpoint-dir", str(pool_dir), "--expert-ids", ""]
        if plan["mode"] == "4gpu":
            # Spec §18 mode B: 3000 test samples sharded across 4 ranks
            # (~750 each); deterministic generation -> merged answers are
            # identical to single-GPU; merge verifies 0 missing/0 duplicate
            # and restores the original record order.
            all_records = json.load(open(test_path, "r", encoding="utf-8"))
            chunk_commands = [
                eval_command
                + [
                    "--num-chunks", str(plan["world_size"]),
                    "--chunk-idx", str(index),
                    "--answers-file", str(eval_output / "answers.rank{}.jsonl".format(index)),
                    "--run-summary-file", str(eval_output / "run_summary.rank{}.json".format(index)),
                ]
                for index in range(plan["world_size"])
            ]
            _run_shards(
                chunk_commands,
                [_worker_env(plan, index) for index in range(plan["world_size"])],
                root,
                "s11_eval",
            )
            merged_answers = merge_partial_answer_files(
                [
                    str(eval_output / "answers.rank{}.jsonl".format(index))
                    for index in range(plan["world_size"])
                ],
                len(all_records),
            )
            (eval_output / "answers.jsonl").write_text(merged_answers, encoding="utf-8")
            summaries = [
                json.loads(
                    (eval_output / "run_summary.rank{}.json".format(index)).read_text(
                        encoding="utf-8"
                    )
                )
                for index in range(plan["world_size"])
            ]
            merged_summary = dict(summaries[0])
            merged_summary["samples"] = sum(
                entry["samples"] for entry in summaries
            )
            merged_summary["duration_seconds"] = sum(
                entry["duration_seconds"] for entry in summaries
            )
            merged_summary["samples_per_second"] = (
                merged_summary["samples"] / merged_summary["duration_seconds"]
                if merged_summary["duration_seconds"]
                else 0.0
            )
            merged_summary["peak_memory_bytes"] = max(
                entry["peak_memory_bytes"] for entry in summaries
            )
            histograms = [
                entry.get("router_selection_histogram")
                for entry in summaries
                if entry.get("router_selection_histogram") is not None
            ]
            if histograms:
                merged_histogram = {}
                for entry in histograms:
                    for key, count in entry.items():
                        merged_histogram[str(key)] = (
                            merged_histogram.get(str(key), 0) + int(count)
                        )
                merged_summary["router_selection_histogram"] = merged_histogram
            merged_summary["execution"] = {
                "mode": "4gpu_sharded",
                "world_size": plan["world_size"],
                "shards": plan["world_size"],
                "merge_verified": {
                    "missing": 0,
                    "duplicates": 0,
                    "lines": len(all_records),
                },
            }
            _write_json(str(eval_output / "run_summary.json"), merged_summary)
        else:
            _run(eval_command, env, root, "s11_eval")
        _advance(
            root, machine, TaskStage.EVALUATION_COMPLETE,
            note="router-based evaluation done",
        )
        _mark_stage(root, "s11_eval")

    # ---- S12: report ------------------------------------------------------
    if not _stage_done(root, "s12_report"):
        _write_report(
            root, machine, registry, task_id, task_name, cluster_expert_ids,
            config,
        )
        _advance(root, machine, TaskStage.COMPLETED, note="task report written")
        _mark_stage(root, "s12_report")


def _formed_from_payload(payload: Dict[str, Any]):
    """Rebuild FormedExpert objects from a persisted formation manifest."""
    from compose.expansion.expert_formation import FormedExpert

    return [
        FormedExpert(
            expert_id=int(entry["expert_id"]),
            cluster_id=int(entry["cluster_id"]),
            sample_ids=tuple(entry["sample_ids"]),
            size=int(entry["size"]),
            centroid=tuple(float(value) for value in entry["centroid"]),
            key_mode=str(entry["key_mode"]),
            creation_task=int(entry["creation_task"]),
        )
        for entry in payload["formed_experts"]
    ]


def _write_report(
    root: Path,
    machine: TaskStateMachine,
    registry: ExpertRegistry,
    task_id: int,
    task_name: str,
    cluster_expert_ids: Sequence[int],
    config: Dict[str, Any],
) -> None:
    report: List[str] = []
    report.append("# Compose Task {} ({}) report".format(task_id, task_name))
    report.append("")
    report.append("- task_id: {}".format(task_id))
    report.append("- stage: {}".format(machine.stage.value))
    report.append("- pool_version: {}".format(registry.pool_version))
    report.append(
        "- active expert ids: {}".format(
            [expert.expert_id for expert in registry.get_active_experts()]
        )
    )
    residual_summary = json.loads(
        (root / "residual" / "summary.json").read_text(encoding="utf-8")
    )
    report.append("- residual: {}".format(residual_summary))
    recall_audit_path = root / "teacher" / "recall_audit.json"
    if recall_audit_path.is_file():
        audit = json.loads(recall_audit_path.read_text(encoding="utf-8"))
        report.append(
            "- recall audit: OracleMemberRecall@1={}, @{}={}, MRR={}".format(
                audit.get("OracleMemberRecall@1"),
                audit.get("top_m"),
                audit.get("OracleMemberRecall@{}".format(audit.get("top_m"))),
                audit.get("MRR"),
            )
        )
    assignment_path = root / "cluster" / "assignment_stats.json"
    if assignment_path.is_file():
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
        report.append(
            "- clustering: selected_k={}, silhouette={}, sizes={}".format(
                assignment["selected_k"],
                assignment["selected_silhouette"],
                assignment["cluster_sizes"],
            )
        )
    formation_path = root / "cluster" / "formation.json"
    if formation_path.is_file():
        formation = load_expert_formation(str(formation_path))
        report.append(
            "- formed experts: {}".format(
                [
                    {
                        "expert_id": entry["expert_id"],
                        "cluster_id": entry["cluster_id"],
                        "size": entry["size"],
                        "key_mode": entry["key_mode"],
                    }
                    for entry in formation["formed_experts"]
                ]
            )
        )
    keys_path = root / "keys" / "key_learning_results.json"
    if keys_path.is_file():
        keys = json.loads(keys_path.read_text(encoding="utf-8"))
        report.append(
            "- key learning: {}".format(
                {
                    expert_id: {
                        "final_loss": value["final_loss"],
                        "positive_similarity_after": value[
                            "positive_similarity_after"
                        ],
                        "epochs_run": value["epochs_run"],
                    }
                    for expert_id, value in keys.items()
                }
            )
        )
    commit_path = root / "committed" / "commit_summary.json"
    if commit_path.is_file():
        commits = json.loads(commit_path.read_text(encoding="utf-8"))
        report.append(
            "- commits: {} (pool_version -> {})".format(
                [item["expert_id"] for item in commits["committed"]],
                commits["pool_version"],
            )
        )
    rms_summary_path = root / "rms" / "rms_summary.json"
    if rms_summary_path.is_file():
        rms = json.loads(rms_summary_path.read_text(encoding="utf-8"))
        report.append(
            "- rms: split={}, layers_with_kappa={}, calibration_sha256={}".format(
                rms["calibration_split"],
                rms["layers_with_kappa"],
                rms["calibration_sha256"][:16],
            )
        )
    eval_path = root / "eval_output" / "run_summary.json"
    if eval_path.is_file():
        run_summary = json.loads(eval_path.read_text(encoding="utf-8"))
        report.append(
            "- eval: samples={}, selection_mode={}".format(
                run_summary.get("samples"),
                run_summary.get("selection_mode"),
            )
        )
        histogram = run_summary.get("router_selection_histogram")
        if histogram is not None:
            report.append(
                "- eval selection histogram (0/1/2 experts): {}".format(histogram)
            )
    (root / "report").mkdir(parents=True, exist_ok=True)
    (root / "report" / "task_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="compose_ucit.yaml path (defaults when absent)")
    parser.add_argument("--root", required=True,
                        help="run root directory; tasks live under task{id}/")
    parser.add_argument("--first-task", type=int, default=0)
    parser.add_argument("--last-task", type=int, default=5)
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()

    config = _load_config(args.config)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for task_id in range(args.first_task, args.last_task + 1):
        task_root = root / "task{}".format(task_id)
        prev_root = root / "task{}".format(task_id - 1)
        run_task(task_root, prev_root, task_id, args.gpus, config, args.config)
    print("run complete: {}".format(root))


if __name__ == "__main__":
    main()
