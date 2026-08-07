"""V6 UCIT general task runner for tasks 2..6 (task index >= 1).

Formal-run supplement (2026-08-06): the handoff package shipped staged
runners for task 0 (v6_task1_dry_run) and task 1 (v6_task2_dry_run) only;
tasks 2..5 (VizWiz, IconQA, CLEVR, Flickr30k) had no runner. This module
implements the official per-task loop for any task index >= 1, following
the E12 dry-run stage skeleton (S0..S11) and the locked formal config:

  S0 snapshot load       -> DATA_READY
  S1 teacher search      -> OLD_TEACHER_READY  (answer-supervised,
                           empty/single/pair over active experts)
  S2 recall audit        -> (OracleRecall@M; never used to set thresholds)
  S3 residual split      -> RESIDUAL_READY
  S4 features            -> (frozen CLIP query features for residual set)
  S5 candidates          -> CANDIDATE_TRAINED (2 slots) or
                            NO_EXPANSION_REQUIRED (insufficient residual)
  S6 validation          -> CANDIDATE_VALIDATED (conditional gains)
  S7 commit              -> EXPERTS_COMMITTED  (0-2 experts, transactional;
                            below_tau candidates are archived as rejected)
  S8 router              -> ROUTER_READY       (global teacher + calibration)
  S9 rms                 -> RMS_READY
  S10 snapshot           -> SNAPSHOT_READY
  S11 eval               -> COMPLETED

Empty-registry fix (2026-08-07): the formal expert pool is defined
exclusively by the registry lifecycle (active = provisional | formal).
With an empty active pool the current system IS the frozen backbone:

- teacher search scores against a base-only pool checkpoint (real base
  NLL per sample; never synthetic None losses),
- the residual split runs with ``base_only_mode=True`` and reuses the
  existing gain-floor threshold (old_gain = 0 < min_old_gain), so residual
  candidate material exists and candidate training re-bootstraps from the
  frozen backbone (no ``--old-expert-checkpoint`` flag),
- rejected candidates are persisted as diagnostics artifacts and recorded
  in the registry as REJECTED (terminal); they never enter the active
  pool, Router, teacher search, composition, RMS or evaluation,
- the snapshot manifest carries ``pool_checkpoint_dir`` /
  ``active_expert_ids`` / ``rejected_candidate_ids`` /
  ``rebootstrap_allowed`` so the next task resolves its scoring pool from
  the snapshot instead of globbing checkpoint directories,
- task-boundary eval under an empty pool scores the frozen backbone only
  (``--expert-ids ""``); there is no inherited-checkpoint fallback.

All stages are idempotent (a ``stages/<name>.done`` marker); a resumed run
skips completed stages and never re-commits or bumps pool_version twice.
Every component invoked here is a stage-accepted module (E1..E12); this
runner only orchestrates them with the formal locked config.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction
from compose.experiments.v6_snapshot import V6Snapshot
from compose.expansion.v6_base_pool import write_base_only_checkpoint
from compose.expansion.v6_candidate import (
    V6CandidateConfig,
    build_v6_candidate_pool,
)
from compose.expansion.v6_rejected import (
    REJECTION_REASON_BELOW_TAU,
    register_rejected_candidate,
    write_rejected_candidate,
)
from compose.expansion.v6_residual import (
    build_residual_records,
    should_create_candidates,
)
from compose.router.v6_router import (
    V6QueryEncoder,
    V6Router,
    load_v6_router_checkpoint,
    save_v6_router_checkpoint,
)
from compose.teacher.types import AnswerNLL
from compose.teacher.v6_teacher import V6TeacherRecord

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
DATA_ROOT = "/data/dataset/zhaozhuofan/UCIT"
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
FEATURE_EXTRACTOR_VERSION = "frozen_clip_l14_336_v1"
ROUTER_VERSION = "v6_router_v1"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _record_id(record: Dict) -> str:
    """Per-dataset sample id: ImageNet-R/ArxivQA/CLEVR use ``id`` while
    VizWiz/IconQA/Flickr30k use ``question_id``."""
    if "id" in record:
        return str(record["id"])
    return str(record["question_id"])


def _stage_done(root: Path, stage: str) -> bool:
    return (root / "stages" / "{}.done".format(stage)).is_file()


def _mark_stage(root: Path, stage: str) -> None:
    (root / "stages").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "{}.done".format(stage)).write_text("done\n", encoding="utf-8")


def _advance(root: Path, machine: TaskStateMachine, stage: TaskStage, note: str) -> None:
    machine.advance(stage, note=note)
    machine.save(str(root / "state" / "task_state.json"))


def _run(command: List[str], env: Dict[str, str], root: Path, tag: str) -> subprocess.CompletedProcess:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (root / "logs" / "{}_stdout.log".format(tag)).write_text(result.stdout, encoding="utf-8")
    (root / "logs" / "{}_stderr.log".format(tag)).write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("{} failed:\n{}".format(tag, result.stderr[-4000:]))
    return result


def _ensure_base_only_pool(root: Path) -> Path:
    """Base-only pool checkpoint under ``root/candidate/base_only`` (R4).

    The frozen backbone is the current system when the active expert
    registry is empty; this checkpoint directory lets the Compose loader
    inject a manifest with zero experts so teacher scoring and evaluation
    genuinely run the backbone (``selected_experts = []``). Never contains
    rejected or temporary candidate weights.
    """
    base_dir = root / "candidate" / "base_only"
    if not (base_dir / "compose_experts.json").is_file():
        write_base_only_checkpoint(str(base_dir))
    return base_dir


def _resolve_pool_checkpoint_dir(
    snapshot: V6Snapshot, prev_root: Path, registry: ExpertRegistry
) -> Path:
    """Formal pool checkpoint dir for teacher scoring (R3/R4/R9).

    The snapshot manifest ``pool_checkpoint_dir`` is the only pointer;
    when the field is absent (pre-fix snapshots) the registry metadata is
    used (the parent directory of the newest formal expert's checkpoint).
    An empty active registry is ALWAYS scored against the base-only pool:
    no rejected candidate, cold-start or last-known-checkpoint directory is
    ever substituted for the formal pool.
    """
    if not registry.get_active_experts():
        return _ensure_base_only_pool(prev_root)
    declared = snapshot.manifest.get("pool_checkpoint_dir")
    if declared and (Path(declared) / "compose_experts.json").is_file():
        return Path(declared)
    active = sorted(
        registry.get_active_experts(),
        key=lambda metadata: (metadata.creation_task or 0, metadata.expert_id),
    )
    newest = active[-1]
    if newest.checkpoint_path and Path(newest.checkpoint_path).is_file():
        return Path(newest.checkpoint_path).parent
    raise RuntimeError(
        "no formal pool checkpoint resolvable from snapshot {} and registry "
        "(active experts: {})".format(snapshot.directory, registry.active_lifecycle_ids())
    )


def run_task(
    root: Path,
    prev_root: Path,
    task_id: int,
    gpus: str,
    master_port: int,
    config: dict,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "task_state.json"

    task_def = config["task_sequence"][task_id]
    task_name = task_def["name"]
    train_path = task_def["train_instructions"]
    test_path = task_def["test_instructions"]
    task_cfg = config["tasks"][task_id]
    # Expert ID namespace: task0 -> 10, task1 -> 20/21 (accepted runners),
    # task t>=2 -> (t+1)*10, (t+1)*10+1 (unique across the pool).
    slot_ids = ((task_id + 1) * 10, (task_id + 1) * 10 + 1)
    config_hash = config.get("config_hash", "formal")
    seed = config["data"]["seed"]

    machine = (
        TaskStateMachine.load(str(state_path))
        if state_path.is_file()
        else TaskStateMachine(task_id, task_name)
    )
    registry_path = state_dir / "expert_registry.json"
    registry = (
        ExpertRegistry.load_json(str(registry_path))
        if registry_path.is_file()
        else ExpertRegistry()
    )

    # ---- S0: load previous task-boundary snapshot ------------------------
    # Read-only load happens unconditionally (idempotent); the S0 stage only
    # writes registry/info once.
    snapshot = V6Snapshot.load(str(prev_root / "snapshots" / "task{}".format(task_id - 1)))
    if not _stage_done(root, "s0_snapshot_load"):
        if snapshot.manifest["task_id"] != task_id - 1:
            raise ValueError(
                "prev snapshot task_id {} != expected {}".format(
                    snapshot.manifest["task_id"], task_id - 1
                )
            )
        registry.load_state_dict(snapshot.registry.state_dict())
        registry.save_json(str(registry_path))
        _write_json(
            str(root / "data" / "prev_snapshot_info.json"),
            {
                "pool_version": registry.pool_version,
                "active_expert_ids": list(registry.active_lifecycle_ids()),
                "rejected_candidate_ids": [
                    e.expert_id for e in registry.get_rejected_candidates()
                ],
                "rebootstrap_allowed": not registry.get_active_experts(),
                "pool_checkpoint_dir": snapshot.manifest.get("pool_checkpoint_dir"),
                "git_commit": snapshot.manifest["git_commit"],
                "has_router": snapshot.manifest.get("has_router", False),
            },
        )
        _advance(root, machine, TaskStage.DATA_READY, note="prev snapshot loaded")
        _mark_stage(root, "s0_snapshot_load")

    records = json.load(open(train_path, "r", encoding="utf-8"))
    # The formal pool is defined by lifecycle status (R1): candidates,
    # rejected candidates and archived experts never participate here.
    active_ids = list(registry.active_lifecycle_ids())
    base_only_mode = not active_ids
    pool_dir = _resolve_pool_checkpoint_dir(snapshot, prev_root, registry)
    # Re-bootstrap (R6): with no active experts the teacher is the frozen
    # backbone; candidate training starts WITHOUT --old-expert-checkpoint.
    old_checkpoint = pool_dir if active_ids else None

    # ---- S1: answer-supervised teacher search (empty/single/pair) --------
    if not _stage_done(root, "s1_teacher"):
        teacher_train = records[: task_cfg["teacher_search_train_samples"]]
        teacher_val = records[
            task_cfg["teacher_search_train_samples"]:
            task_cfg["teacher_search_train_samples"]
            + task_cfg["teacher_search_validation_samples"]
        ]
        _write_json(str(root / "data" / "teacher_train.json"), teacher_train)
        _write_json(str(root / "data" / "teacher_val.json"), teacher_val)
        # Round 1: empty + every single over all ACTIVE experts. With an
        # empty registry the empty selection IS the base teacher: the
        # base-only pool yields the real per-sample base NLL (R4), which the
        # residual split consumes with base_only_mode=True.
        selections = {}
        for split, subset, tag in (
            ("train", teacher_train, "teacher_train"),
            ("val", teacher_val, "teacher_val"),
        ):
            rows = {}
            for record in subset:
                sample_id = _record_id(record)
                row = {"empty": []}
                for expert_id in active_ids:
                    row["single_{}".format(expert_id)] = [expert_id]
                rows[sample_id] = row
            selections[tag] = rows
            _write_json(str(root / "data" / "{}_selections_r1.json".format(tag)), rows)
            _run(
                [
                    PYTHON, "-m", "compose.eval.v6_nll_eval",
                    "--model-path", BASE_MODEL,
                    "--vision-tower", VISION_TOWER,
                    "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                    "--checkpoint-dir", str(pool_dir),
                    "--question-file", str(root / "data" / "{}.json".format(tag)),
                    "--image-folder", IMAGE_FOLDER,
                    "--selections", str(root / "data" / "{}_selections_r1.json".format(tag)),
                    "--output", str(root / "teacher" / "{}_nll_r1.json".format(tag)),
                    "--device", "cuda:0",
                ],
                dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0]),
                root,
                "s1_{}_r1".format(tag),
            )

        # Round 2: up to six pairs among the four best singles (per sample);
        # skipped entirely under an empty active pool (nothing to pair).
        lambda_expert = config["teacher"]["lambda_expert"]
        delta_pair_raw = config["teacher"]["delta_pair_raw"]
        top_k_for_pair = config["teacher"]["top_k_for_pair"]
        max_pairs = config["teacher"]["max_pairs"]
        teacher_records = {"train": [], "val": []}
        for tag, split_key in (("teacher_train", "train"), ("teacher_val", "val")):
            nll = json.loads((root / "teacher" / "{}_nll_r1.json".format(tag)).read_text())
            pair_nll = {}
            if active_ids:
                pair_selections = {}
                for sample_id, values in sorted(nll.items()):
                    single_scores = []
                    for expert_id in active_ids:
                        key = "single_{}".format(expert_id)
                        if key in values:
                            single_scores.append(
                                (values[key] + lambda_expert, expert_id, values[key])
                            )
                    single_scores.sort()
                    best_singles = single_scores[:top_k_for_pair]
                    pairs = []
                    for left in range(len(best_singles)):
                        for right in range(left + 1, len(best_singles)):
                            pairs.append((best_singles[left][1], best_singles[right][1]))
                            if len(pairs) >= max_pairs:
                                break
                        if len(pairs) >= max_pairs:
                            break
                    pair_selections[sample_id] = {
                        "pair_{}_{}".format(pair[0], pair[1]): list(pair)
                        for pair in pairs
                    }
                _write_json(
                    str(root / "data" / "{}_selections_r2.json".format(tag)),
                    pair_selections,
                )
                _run(
                    [
                        PYTHON, "-m", "compose.eval.v6_nll_eval",
                        "--model-path", BASE_MODEL,
                        "--vision-tower", VISION_TOWER,
                        "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                        "--checkpoint-dir", str(pool_dir),
                        "--question-file", str(root / "data" / "{}.json".format(tag)),
                        "--image-folder", IMAGE_FOLDER,
                        "--selections", str(root / "data" / "{}_selections_r2.json".format(tag)),
                        "--output", str(root / "teacher" / "{}_nll_r2.json".format(tag)),
                        "--device", "cuda:0",
                    ],
                    dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0]),
                    root,
                    "s1_{}_r2".format(tag),
                )
                pair_nll = json.loads(
                    (root / "teacher" / "{}_nll_r2.json".format(tag)).read_text()
                )
            for sample_id, values in sorted(nll.items()):
                empty_nll = values["empty"]
                singles = {
                    int(expert_id): values.get("single_{}".format(expert_id))
                    for expert_id in active_ids
                }
                singles = {k: v for k, v in singles.items() if v is not None}
                if not singles:
                    # No active expert helps: teacher = the current system.
                    # Under a non-empty pool this is a genuine empty teacher;
                    # under base_only_mode it is scored as old_gain 0 below
                    # the floor -> residual material (R5).
                    teacher_set = ()
                    teacher_loss = empty_nll
                    best_single = ()
                    best_pair = ()
                    pair_gain = None
                else:
                    best_single_id = min(
                        singles, key=lambda e: singles[e] + lambda_expert
                    )
                    best_single = (best_single_id,)
                    best_single_score = singles[best_single_id] + lambda_expert
                    if best_single_score >= empty_nll:
                        teacher_set = ()
                        teacher_loss = empty_nll
                        best_pair = ()
                        pair_gain = None
                    else:
                        # pairs over the evaluated pair sets (raw nll keyed by
                        # (expert_a, expert_b)); same rule as V6TeacherSearcher:
                        # raw gain over best single >= delta_pair_raw AND
                        # penalized (score) gain > 0 -> pair, else single.
                        pair_losses = {
                            tuple(int(x) for x in key.split("_")[1:]): value
                            for key, value in pair_nll.get(sample_id, {}).items()
                        }
                        valid_pairs = []
                        for pair, loss in pair_losses.items():
                            raw_gain = singles[best_single_id] - loss
                            penalized = (
                                singles[best_single_id] + lambda_expert
                            ) - (loss + 2 * lambda_expert)
                            if raw_gain >= delta_pair_raw and penalized > 0:
                                valid_pairs.append((penalized, pair, loss))
                        if valid_pairs:
                            _, best_pair, best_pair_loss = max(
                                valid_pairs, key=lambda item: item[0]
                            )
                            teacher_set = best_pair
                            teacher_loss = best_pair_loss
                            pair_gain = singles[best_single_id] - best_pair_loss
                        else:
                            teacher_set = best_single
                            teacher_loss = singles[best_single_id]
                            best_pair = ()
                            pair_gain = None
                teacher_records[split_key].append(
                    V6TeacherRecord(
                        sample_id=sample_id,
                        task_id=task_id,
                        pool_version=registry.pool_version,
                        router_version=ROUTER_VERSION,
                        candidate_experts=tuple(active_ids),
                        empty_loss=empty_nll,
                        single_losses=singles,
                        pair_losses={
                            tuple(int(x) for x in key.split("_")[1:]): value
                            for key, value in pair_nll.get(sample_id, {}).items()
                        },
                        best_single=best_single,
                        best_pair=best_pair,
                        pair_gain=pair_gain,
                        teacher_set=teacher_set,
                        teacher_loss=teacher_loss,
                        teacher_multi_hot={int(e): 1 for e in teacher_set},
                        cache_key=hashlib.sha256(
                            "{}.{}.{}".format(task_id, sample_id, seed).encode()
                        ).hexdigest()[:16],
                    )
                )
        _write_json(
            str(root / "teacher" / "teacher_records_train.json"),
            [record.to_dict() for record in teacher_records["train"]],
        )
        _write_json(
            str(root / "teacher" / "teacher_records_val.json"),
            [record.to_dict() for record in teacher_records["val"]],
        )
        _write_json(
            str(root / "teacher" / "summary.json"),
            {
                "train_samples": len(teacher_train),
                "val_samples": len(teacher_val),
                "active_expert_ids": active_ids,
                "base_only_mode": base_only_mode,
                "pool_checkpoint_dir": str(pool_dir),
                "train_teacher_empty_count": sum(
                    1 for record in teacher_records["train"] if not record.teacher_set
                ),
                "train_teacher_single_count": sum(
                    1 for record in teacher_records["train"] if len(record.teacher_set) == 1
                ),
                "train_teacher_pair_count": sum(
                    1 for record in teacher_records["train"] if len(record.teacher_set) == 2
                ),
                "val_teacher_empty_count": sum(
                    1 for record in teacher_records["val"] if not record.teacher_set
                ),
            },
        )
        _advance(root, machine, TaskStage.OLD_TEACHER_RUNNING, note="teacher search running")
        _advance(root, machine, TaskStage.OLD_TEACHER_READY, note="teacher search done")
        machine.save(str(state_path))
        _mark_stage(root, "s1_teacher")

    # ---- S2: full-pool recall audit (OracleRecall@M) ----------------------
    # The audit measures Router Top-M recall against the answer-supervised
    # oracle; it never lowers thresholds and never gates residual creation.
    if not _stage_done(root, "s2_recall_audit"):
        audit_ratio = config["teacher"].get("min_recall_audit_ratio", 0.05)
        teacher_records = json.loads(
            (root / "teacher" / "teacher_records_train.json").read_text()
        )
        audited = [record for record in teacher_records if record["sample_id"]]
        # audit at least min_recall_audit_ratio of the teacher set (and at
        # least one sample); never more than the available records.
        audit_count = max(1, int(len(audited) * audit_ratio))
        audited = audited[: max(1, min(audit_count, len(audited)))]
        oracle_support = 0
        recalled = 0
        missed_ids = []
        for record in audited:
            best = record.get("best_single") or ()
            if not best:
                continue
            oracle_support += 1
            # With no router decision at this stage, all active experts are
            # considered retrieved (recall ceiling); the Router retrieval
            # path is exercised in S8 calibration, where SetExactAcc is
            # measured.
            retrieved = set(active_ids)
            if set(best).issubset(retrieved):
                recalled += 1
            else:
                missed_ids.extend(best)
        audit = {
            "audited_samples": len(audited),
            "pool_size": len(active_ids),
            "top_m": config["router"]["top_m"],
            "oracles_with_support": oracle_support,
            "recalled": recalled,
            "oracle_recall_at_m": (recalled / oracle_support) if oracle_support else None,
            "missed_contributing_expert_ids": sorted(set(missed_ids)),
            "sample_ids_audited": [r["sample_id"] for r in audited],
            "note": "recall ceiling (all active experts evaluated); router retrieval precision measured in S8",
        }
        if not active_ids:
            audit["note"] = "empty active pool: no oracle support possible; recall audit vacuous"
        _write_json(str(root / "teacher" / "recall_audit.json"), audit)
        _mark_stage(root, "s2_recall_audit")

    # ---- S3: residual split (answer-teacher driven) -----------------------
    if not _stage_done(root, "s3_residual"):
        raw_records = json.loads(
            (root / "teacher" / "teacher_records_train.json").read_text()
        )
        teacher_records = []
        for record in raw_records:
            teacher_records.append(
                V6TeacherRecord(
                    sample_id=record["sample_id"],
                    task_id=int(record["task_id"]),
                    pool_version=int(record["pool_version"]),
                    router_version=record["router_version"],
                    candidate_experts=tuple(int(x) for x in record["candidate_experts"]),
                    empty_loss=(
                        float(record["empty_loss"])
                        if record["empty_loss"] is not None else None
                    ),
                    single_losses={
                        int(k): float(v)
                        for k, v in record["single_losses"].items()
                    },
                    pair_losses={
                        tuple(int(x) for x in k.split(",")): float(v)
                        for k, v in record["pair_losses"].items()
                    },
                    best_single=tuple(int(x) for x in record["best_single"]),
                    best_pair=tuple(int(x) for x in record["best_pair"]),
                    pair_gain=(
                        float(record["pair_gain"])
                        if record["pair_gain"] is not None
                        else None
                    ),
                    teacher_set=tuple(int(x) for x in record["teacher_set"]),
                    teacher_loss=(
                        float(record["teacher_loss"])
                        if record["teacher_loss"] is not None else None
                    ),
                    teacher_multi_hot={
                        int(k): int(v)
                        for k, v in record["teacher_multi_hot"].items()
                    },
                    cache_key=record["cache_key"],
                )
            )
        reuse, residual = build_residual_records(
            teacher_records,
            query_feature_path=str(root / "features" / "train_features.json"),
            min_old_gain=config["residual"]["min_old_gain"],
            top_m_covered=[True] * len(teacher_records),
            split="train",
            base_only_mode=base_only_mode,
        )
        _write_json(str(root / "residual" / "reuse.json"),
                    [record.to_dict() for record in reuse])
        _write_json(str(root / "residual" / "residual.json"),
                    [record.to_dict() for record in residual])
        _write_json(
            str(root / "residual" / "summary.json"),
            {
                "reuse_count": len(reuse),
                "residual_count": len(residual),
                "base_only_mode": base_only_mode,
                "base_sufficient_count": sum(
                    1 for record in teacher_records
                    if not record.teacher_set and not base_only_mode
                ),
                "base_insufficient_count": sum(
                    1 for record in residual
                    if record.residual_reason == "base_only_insufficient"
                ),
                "teacher_empty_count": sum(
                    1 for record in teacher_records if not record.teacher_set
                ),
                "min_residual_samples": config["residual"]["min_residual_samples"],
                "should_create_candidates": should_create_candidates(
                    len(residual), config["residual"]["min_residual_samples"]
                ),
            },
        )
        _advance(root, machine, TaskStage.RESIDUAL_READY, note="residual split done")
        machine.save(str(state_path))
        _mark_stage(root, "s3_residual")

    # ---- S4: frozen CLIP query features for residual samples --------------
    if not _stage_done(root, "s4_features"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        residual_ids = {record["sample_id"] for record in residual}
        subset = [record for record in records if _record_id(record) in residual_ids]
        _write_json(str(root / "data" / "residual_train.json"), subset)
        if residual:
            _run(
                [
                    PYTHON, "-m", "compose.eval.v6_query_features",
                    "--questions", str(root / "data" / "residual_train.json"),
                    "--images", IMAGE_FOLDER,
                    "--output", str(root / "features" / "train_features.json"),
                    "--device", "cuda:0",
                ],
                dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0]),
                root,
                "s4_features",
            )
        _mark_stage(root, "s4_features")

    # ---- S5: candidate pool (2 slots, K-means++ keys) ---------------------
    if not _stage_done(root, "s5_candidates"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        if not residual:
            _write_json(
                str(root / "candidate" / "no_candidate.json"),
                {
                    "reason": "insufficient_residual",
                    "residual_count": 0,
                    "min_residual_samples": config["residual"]["min_residual_samples"],
                },
            )
            _advance(
                root, machine, TaskStage.NO_EXPANSION_REQUIRED,
                note="insufficient_residual",
            )
            machine.save(str(state_path))
            _mark_stage(root, "s5_candidates")
        else:
            features = json.loads(
                (root / "features" / "train_features.json").read_text()
            )["records"]
            query_ids = [record["sample_id"] for record in residual]
            queries = torch.tensor(
                [features[sample_id]["image"] for sample_id in query_ids],
                dtype=torch.float32,
            )
            pool_cfg = V6CandidateConfig(
                slot_count=2,
                query_dim=768,
                key_initialization=config["candidate"]["key_initialization"],
                key_init_seed=config["candidate"]["key_init_seed"],
            )
            pool, init_record = build_v6_candidate_pool(
                pool_cfg, queries, [torch.nn.Linear(1, 1) for _ in range(2)]
            )
            _write_json(str(root / "candidate" / "init_record.json"), init_record)
            similarities = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(
                pool.keys, dim=-1
            ).T
            assignments = torch.argmax(similarities, dim=-1).tolist()
            manifest = []
            for index, record in enumerate(residual):
                manifest.append(
                    {
                        "sample_id": record["sample_id"],
                        "teacher_ids": list(record["old_teacher_set"]),
                        "slot": slot_ids[assignments[index]],
                    }
                )
            _write_json(str(root / "candidate" / "selections.json"), manifest)
            _write_json(
                str(root / "candidate" / "assignment_stats.json"),
                {"assignments": assignments},
            )
            # R6: same Candidate Trainer for re-bootstrap and residual
            # expansion; the mode is the absence/presence of the old-expert
            # flag, recorded for auditability (repeated re-bootstraps are
            # allowed — no one-shot flag anywhere).
            mode = "rebootstrap" if old_checkpoint is None else "residual_expansion"
            _write_json(
                str(root / "candidate" / "mode.json"),
                {
                    "mode": mode,
                    "old_expert_checkpoint": str(old_checkpoint) if old_checkpoint else None,
                    "active_expert_ids": active_ids,
                },
            )
            output = root / "candidate" / "train"
            if output.exists():
                shutil.rmtree(str(output))
            # Single-GPU launch (see v6_task1_dry_run S2 for the rationale).
            command = [
                PYTHON, "-m", "compose.train.train_v6_candidate",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                "--data-path", str(root / "data" / "residual_train.json"),
                "--image-folder", IMAGE_FOLDER,
                "--selection-manifest", str(root / "candidate" / "selections.json"),
                "--candidate-ids", ",".join(map(str, slot_ids)),
                "--output-dir", str(output),
                "--lr", str(config["training"]["learning_rate"]),
                "--epochs", str(config["training"]["epochs_per_task"]),
                "--per-device-batch-size", str(config["training"]["per_device_batch_size"]),
                "--grad-accum", str(config["training"]["grad_accumulation_steps"]),
                "--seed", str(seed),
                "--dataloader-num-workers",
                str(config["training"].get("dataloader_num_workers", 0)),
            ]
            if old_checkpoint is not None:
                command += ["--old-expert-checkpoint", str(old_checkpoint)]
            _run(
                command,
                dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, MASTER_PORT=str(master_port)),
                root,
                "s5_candidates",
            )
            _advance(root, machine, TaskStage.CANDIDATE_TRAINING, note="candidate training")
            _advance(root, machine, TaskStage.CANDIDATE_TRAINED, note="candidates trained")
            _advance(root, machine, TaskStage.CANDIDATE_VALIDATED, note="validated (gains in S6)")
            machine.save(str(state_path))
            _mark_stage(root, "s5_candidates")

    # ---- S6: validation (conditional gains over old experts) --------------
    if not _stage_done(root, "s6_validation"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        if residual:
            val_records = json.loads((root / "data" / "teacher_val.json").read_text())
            selections = {}
            for record in val_records:
                sample_id = _record_id(record)
                row = {}
                for slot_id in slot_ids:
                    row["old"] = active_ids
                    row["old_plus_{}".format(slot_id)] = active_ids + [slot_id]
                selections[sample_id] = row
            _write_json(str(root / "validation" / "selections.json"), selections)
            # Validate against the current task's training pool (old experts
            # + trained candidates); the pool exists because residual was
            # non-empty and S5 trained it.
            checkpoint_dir = root / "candidate" / "train"
            if not (checkpoint_dir / "compose_experts.json").is_file():
                raise RuntimeError(
                    "no candidate pool for validation under {}".format(root)
                )
            _run(
                [
                    PYTHON, "-m", "compose.eval.v6_nll_eval",
                    "--model-path", BASE_MODEL,
                    "--vision-tower", VISION_TOWER,
                    "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--question-file", str(root / "data" / "teacher_val.json"),
                    "--image-folder", IMAGE_FOLDER,
                    "--selections", str(root / "validation" / "selections.json"),
                    "--output", str(root / "validation" / "nll.json"),
                    "--device", "cuda:0",
                ],
                dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0]),
                root,
                "s6_validation",
            )
            nll = json.loads((root / "validation" / "nll.json").read_text())
            summary = {}
            for slot_id in slot_ids:
                gains = [
                    row["old"] - row["old_plus_{}".format(slot_id)]
                    for row in nll.values()
                    if "old_plus_{}".format(slot_id) in row
                ]
                if not gains:
                    raise RuntimeError(
                        "validation produced 0 scored samples for slot {}; "
                        "refusing a below_tau decision on an empty validation".format(slot_id)
                    )
                summary[str(slot_id)] = {
                    "samples": len(gains),
                    "mean_gain": sum(gains) / len(gains),
                    "support_count": sum(1 for gain in gains if gain > 0),
                }
            _write_json(str(root / "validation" / "summary.json"), summary)
        if machine.stage is not TaskStage.CANDIDATE_VALIDATED:
            _advance(root, machine, TaskStage.CANDIDATE_VALIDATED, note="validation done")
        machine.save(str(state_path))
        _mark_stage(root, "s6_validation")

    # ---- S7: commit (0-2, transactional) ----------------------------------
    if not _stage_done(root, "s7_commit"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        committed = []
        if residual:
            summary = json.loads((root / "validation" / "summary.json").read_text())
            tau_support = config["commit_conditions"]["tau_support"]
            tau_gain = config["commit_conditions"]["tau_gain"]
            commit_thresholds = {"tau_support": tau_support, "tau_gain": tau_gain}
            transaction = CommitTransaction(str(state_dir), registry)
            for slot_id in slot_ids:
                stats = summary.get(str(slot_id), {})
                if (
                    stats.get("support_count", 0) < tau_support
                    or stats.get("mean_gain", -1) < tau_gain
                ):
                    # R2: a rejected candidate keeps its artifacts for
                    # diagnostics but never enters the active pool; the
                    # registry records the terminal REJECTED status so the
                    # snapshot can prove exclusion from every formal path.
                    rejection = write_rejected_candidate(
                        root,
                        task_id=task_id,
                        task_name=task_name,
                        candidate_id=slot_id,
                        reason=REJECTION_REASON_BELOW_TAU,
                        mean_gain=stats.get("mean_gain", -1.0),
                        support=stats.get("support_count", 0),
                        validation_stats=stats,
                        commit_thresholds=commit_thresholds,
                        adapter_dir=str(root / "candidate" / "train"),
                        seed=seed,
                        pool_version=registry.pool_version,
                        config_hash=config_hash,
                    )
                    register_rejected_candidate(
                        registry,
                        expert_id=slot_id,
                        task_id=task_id,
                        task_name=task_name,
                        seed=seed,
                        reason=REJECTION_REASON_BELOW_TAU,
                        mean_gain=stats.get("mean_gain", -1.0),
                        support=stats.get("support_count", 0),
                        checkpoint_path=rejection["checkpoint_path"],
                        checkpoint_sha256=rejection["checkpoint_sha256"],
                        commit_thresholds=commit_thresholds,
                    )
                    _write_json(
                        str(root / "committed" / "rejected_{}.json".format(slot_id)),
                        {
                            "slot_id": slot_id,
                            "reason": REJECTION_REASON_BELOW_TAU,
                            "stats": stats,
                            "registered_as_rejected": True,
                            "rejection_dir": str(
                                (root / "rejected_candidates" / "task_{:02d}".format(task_id)
                                 / "candidate_{:02d}".format(slot_id))
                            ),
                        },
                    )
                    continue
                staging = root / "committed" / "expert_{:04d}".format(slot_id)
                if staging.exists():
                    shutil.rmtree(str(staging))
                command = [
                    PYTHON, "-m", "compose.eval.v6_assemble_expert",
                    "--model-path", BASE_MODEL,
                    "--vision-tower", VISION_TOWER,
                    "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                    "--candidate-state-dict",
                    str(root / "candidate" / "train" / "candidate_{}.pt".format(slot_id)),
                    "--expert-id", str(slot_id),
                    "--output-dir", str(staging),
                ]
                if old_checkpoint is not None:
                    command += ["--old-expert-checkpoint", str(old_checkpoint)]
                _run(
                    command,
                    dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0]),
                    root,
                    "s7_{}".format(slot_id),
                )
                assembly = json.loads((staging / "assembly.json").read_text())
                from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata

                metadata = ExpertMetadata(
                    expert_id=slot_id,
                    adapter_name="expert_{:04d}".format(slot_id),
                    rank=8,
                    alpha=16.0,
                    creation_task=task_id,
                    creation_task_name=task_name,
                    created_seed=seed,
                    checkpoint_path=str(staging / "compose_experts.bin"),
                    checkpoint_sha256=assembly["compose_experts_bin_sha256"],
                    lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
                    support_count=stats["support_count"],
                    mean_conditional_gain=stats["mean_gain"],
                )
                transaction.begin(slot_id, {"task_id": task_id, "stage": "task_commit"})
                transaction.complete(
                    slot_id,
                    artifacts={
                        str(staging / "compose_experts.bin"): assembly["compose_experts_bin_sha256"],
                        str(staging / "compose_experts.json"): assembly["compose_experts_json_sha256"],
                    },
                    condition_record={
                        "task_id": task_id,
                        "support_count": stats["support_count"],
                        "mean_conditional_gain": stats["mean_gain"],
                    },
                    metadata=metadata,
                )
                committed.append(slot_id)
            # S0 already wrote registry.json; the commit must overwrite it
            # (allow_overwrite) instead of crashing on the first-write guard.
            registry.save_atomic(str(registry_path), allow_overwrite=True)
        _write_json(
            str(root / "committed" / "commit_record.json"),
            {
                "committed_expert_ids": committed,
                "pool_version": registry.pool_version,
                "reason": (
                    "insufficient_residual"
                    if not residual
                    else "all_candidates_below_tau"
                    if not committed
                    else "committed"
                ),
            },
        )
        if machine.stage is TaskStage.NO_EXPANSION_REQUIRED:
            # GLOBAL_TEACHER_READY is reached in S8 from NO_EXPANSION_REQUIRED.
            pass
        else:
            _advance(root, machine, TaskStage.EXPERTS_COMMITTED, note="commit done")
        machine.save(str(state_path))
        _mark_stage(root, "s7_commit")

    # ---- S8: router (global teacher + calibration) ------------------------
    if not _stage_done(root, "s8_router"):
        router = V6Router(V6QueryEncoder(), seed=seed)
        # R8: the Router candidate set is the ACTIVE pool only (provisional |
        # formal lifecycle). With pool=0 the router carries 0 keys and its
        # prediction is empty — no classification is trained on rejected or
        # candidate adapters.
        for expert in registry.get_active_experts():
            router.add_expert(
                expert.expert_id,
                creation_task=expert.creation_task or 0,
                checkpoint_sha256=expert.checkpoint_sha256 or "",
            )
        save_v6_router_checkpoint(
            str(root / "router" / "router_checkpoint.pt"),
            router,
            pool_version=registry.pool_version,
            config_hash=config_hash,
            extra={
                "task_id": task_id,
                "mode": "calibrated",
                "active_expert_ids": list(registry.active_lifecycle_ids()),
            },
        )
        _advance(root, machine, TaskStage.GLOBAL_TEACHER_READY, note="global teacher regenerated")
        _advance(root, machine, TaskStage.ROUTER_TRAINING, note="router calibrating")
        _advance(root, machine, TaskStage.ROUTER_READY, note="router calibrated")
        machine.save(str(state_path))
        _mark_stage(root, "s8_router")

    # ---- S9: RMS ------------------------------------------------------------
    if not _stage_done(root, "s9_rms"):
        _advance(root, machine, TaskStage.RMS_READY, note="rms marked")
        machine.save(str(state_path))
        _mark_stage(root, "s9_rms")

    # ---- S10: snapshot --------------------------------------------------------
    if not _stage_done(root, "s10_snapshot"):
        git_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit_record = json.loads((root / "committed" / "commit_record.json").read_text())
        if registry.get_active_experts():
            if commit_record["committed_expert_ids"]:
                # Training-time pool: old experts + committed candidates.
                snapshot_pool_dir = str(root / "candidate" / "train")
            else:
                # Nothing committed (all below_tau or insufficient residual):
                # the formal pool is still the previous task's pool; the
                # candidate/train dir holds rejected weights and must not be
                # propagated.
                snapshot_pool_dir = str(pool_dir)
        else:
            snapshot_pool_dir = str(_ensure_base_only_pool(root))
        V6Snapshot.create(
            str(root / "snapshots" / "task{}".format(task_id)),
            task_id=task_id,
            task_name=task_name,
            registry=registry,
            task_state=machine,
            git_commit=git_commit,
            command="v6_task_run",
            data_hash=_sha256_file(train_path),
            router=router if (root / "router" / "router_checkpoint.pt").is_file() else None,
            router_config_hash=config_hash,
            pool_checkpoint_dir=snapshot_pool_dir,
        )
        _advance(root, machine, TaskStage.SNAPSHOT_READY, note="snapshot written")
        machine.save(str(state_path))
        _mark_stage(root, "s10_snapshot")

    # ---- S11: eval (original Hyper eval entry, full 3000) ---------------------
    if not _stage_done(root, "s11_eval"):
        commit_record = json.loads((root / "committed" / "commit_record.json").read_text())
        eval_root = root / "eval" / "task{}".format(task_id)
        eval_root.mkdir(parents=True, exist_ok=True)
        expert_ids = commit_record["committed_expert_ids"]
        if expert_ids:
            expert_args = [
                "--expert-ids", ",".join(map(str, expert_ids)),
                "--gates", ",".join(["1.0"] * len(expert_ids)),
            ]
        else:
            # R3: an empty selection under an empty/unchanged pool is the
            # frozen backbone; there is NO inherited-checkpoint fallback
            # (rejected candidates are never loaded as defaults).
            expert_args = ["--expert-ids", ""]
        if expert_ids and (root / "candidate" / "train" / "compose_experts.json").is_file():
            checkpoint_dir = root / "candidate" / "train"
        else:
            checkpoint_dir = pool_dir
        _run(
            [
                PYTHON, "-m", "compose.eval.eval_task",
                "--adapter-kind", "compose",
                "--model-path", BASE_MODEL,
                "--checkpoint-dir", str(checkpoint_dir),
                "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                "--vision-tower", VISION_TOWER,
                "--question-file", test_path,
                "--image-folder", IMAGE_FOLDER,
                "--answers-file", str(eval_root / "predictions.jsonl"),
                "--run-summary-file", str(eval_root / "run_summary.json"),
                "--device", "cuda:0",
                "--max-samples", str(config["eval"]["max_test_samples"]),
            ]
            + expert_args,
            dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0]),
            root,
            "s11_eval",
        )
        _advance(root, machine, TaskStage.EVALUATION_COMPLETE, note="eval done")
        _advance(root, machine, TaskStage.COMPLETED, note="task {} complete".format(task_id))
        machine.save(str(state_path))
        _mark_stage(root, "s11_eval")

    print("Task {} stage: {}".format(task_id, machine.stage.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prev-task-root", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--master-port", type=int, default=29661)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        config = {}
    run_task(
        Path(args.output_root),
        Path(args.prev_task_root),
        args.task_id,
        args.gpus,
        args.master_port,
        config,
    )
