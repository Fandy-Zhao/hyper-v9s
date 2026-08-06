"""V6 two-task dry run: Task 2 (ArxivQA) staged runner (Stage E12).

Loads the task-1 snapshot, then runs the full continual loop:

  S1 old teacher   -> OLD_TEACHER_READY  (answer-supervised, Top-M recall)
  S2 residual      -> RESIDUAL_READY     (reuse / residual split)
  S3 features      -> (CLIP query features for residual samples)
  S4 candidates    -> CANDIDATE_TRAINED  (2 slots, K-means++ keys)
  S5 validation    -> CANDIDATE_VALIDATED (conditional gains)
  S6 commit        -> EXPERTS_COMMITTED  (0-2 experts, transactional)
  S7 router        -> ROUTER_READY       (global teacher + calibration)
  S8 rms           -> RMS_READY
  S9 snapshot      -> SNAPSHOT_READY
  S10 eval         -> COMPLETED
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction
from compose.experiments.v6_snapshot import V6Snapshot
from compose.expansion.v6_candidate import (
    V6CandidateConfig,
    assignment_statistics,
    build_v6_candidate_pool,
)
from compose.expansion.v6_residual import build_residual_records
from compose.router.v6_router import V6QueryEncoder, V6Router, save_v6_router_checkpoint
from compose.teacher.types import AnswerNLL
from compose.teacher.v6_teacher import V6TeacherRecord

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
DATA_ROOT = "/data/dataset/zhaozhuofan/UCIT"
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"

TASK_NAME = "ArxivQA"
SLOT_IDS = (20, 21)


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _stage_done(root: Path, stage: str) -> bool:
    return (root / "stages" / "{}.done".format(stage)).is_file()


def _mark_stage(root: Path, stage: str) -> None:
    (root / "stages").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "{}.done".format(stage)).write_text("done\n", encoding="utf-8")


def run_task2(root: Path, task1_root: Path, gpus: str, master_port: int, config: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "task_state.json"
    machine = (
        TaskStateMachine.load(str(state_path))
        if state_path.is_file()
        else TaskStateMachine(1, TASK_NAME)
    )
    registry_path = state_dir / "expert_registry.json"
    registry = (
        ExpertRegistry.load_json(str(registry_path))
        if registry_path.is_file()
        else ExpertRegistry()
    )

    # ---- S0: load task-1 snapshot (independent load) ----------------------
    if not _stage_done(root, "s0_snapshot_load"):
        snapshot = V6Snapshot.load(str(task1_root / "snapshots" / "task0"))
        if snapshot.manifest["task_id"] != 0:
            raise ValueError("task-1 snapshot has the wrong task id")
        registry.load_state_dict(snapshot.registry.state_dict())
        registry.save_json(str(registry_path))
        _write_json(
            str(root / "data" / "task1_snapshot_info.json"),
            {
                "pool_version": registry.pool_version,
                "expert_ids": registry.list_all_ids() if hasattr(registry, "list_all_ids") else [e.expert_id for e in registry.list_all()],
                "git_commit": snapshot.manifest["git_commit"],
            },
        )
        machine.advance(TaskStage.DATA_READY, note="task-1 snapshot loaded")
        machine.save(str(state_path))
        _mark_stage(root, "s0_snapshot_load")

    train_path = os.path.join(DATA_ROOT, "instructions", "ArxivQA", "train_4w.json")
    test_path = os.path.join(DATA_ROOT, "instructions", "ArxivQA", "test_3000.json")
    records = json.load(open(train_path, "r", encoding="utf-8"))

    # ---- S1: old-expert teacher search (answer-supervised) ----------------
    if not _stage_done(root, "s1_teacher"):
        teacher_train = records[: config["tasks"][1]["teacher_search_train_samples"]]
        teacher_val = records[
            config["tasks"][1]["teacher_search_train_samples"]:
            config["tasks"][1]["teacher_search_train_samples"]
            + config["tasks"][1]["teacher_search_validation_samples"]
        ]
        _write_json(str(root / "data" / "teacher_train.json"), teacher_train)
        _write_json(str(root / "data" / "teacher_val.json"), teacher_val)
        old_ids = [e.expert_id for e in registry.list_all()]
        for split, subset, tag in (
            ("train", teacher_train, "teacher_train"),
            ("val", teacher_val, "teacher_val"),
        ):
            selections = {}
            for record in subset:
                sample_id = str(record["id"])
                selections[sample_id] = {
                    "empty": [],
                    "single_{}".format(old_ids[0]): [old_ids[0]] if old_ids else [],
                }
            _write_json(str(root / "data" / "{}_selections.json".format(tag)), selections)
            checkpoint_dir = (
                task1_root / "committed" / "expert_{:04d}".format(old_ids[0])
                if old_ids else task1_root / "candidate" / "cold_start"
            )
            command = [
                PYTHON, "-m", "compose.eval.v6_nll_eval",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                "--checkpoint-dir", str(checkpoint_dir),
                "--question-file", str(root / "data" / "{}.json".format(tag)),
                "--image-folder", IMAGE_FOLDER,
                "--selections", str(root / "data" / "{}_selections.json".format(tag)),
                "--output", str(root / "teacher" / "{}_nll.json".format(tag)),
                "--device", "cuda:0",
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            (root / "logs" / "s1_{}_stdout.log".format(tag)).write_text(result.stdout, encoding="utf-8")
            (root / "logs" / "s1_{}_stderr.log".format(tag)).write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError("teacher search {} failed:\n{}".format(tag, result.stderr[-3000:]))
        machine.advance(TaskStage.OLD_TEACHER_RUNNING, note="teacher search running")
        machine.advance(TaskStage.OLD_TEACHER_READY, note="teacher search done")
        machine.save(str(state_path))
        _mark_stage(root, "s1_teacher")

    # ---- S2: residual split (answer teacher driven) -----------------------
    if not _stage_done(root, "s2_residual"):
        teacher_nll = json.loads((root / "teacher" / "teacher_train_nll.json").read_text())
        old_ids = [e.expert_id for e in registry.list_all()]
        teacher_records = []
        for sample_id, rows in sorted(teacher_nll.items()):
            empty_loss = rows["empty"]
            single_loss = rows.get("single_{}".format(old_ids[0])) if old_ids else None
            if single_loss is None:
                teacher_set = ()
                teacher_loss = empty_loss
            elif empty_loss - single_loss >= config["teacher"]["delta_pair_raw"]:
                teacher_set = (old_ids[0],)
                teacher_loss = single_loss
            else:
                teacher_set = ()
                teacher_loss = empty_loss
            teacher_records.append(
                V6TeacherRecord(
                    sample_id=sample_id, task_id=1,
                    pool_version=registry.pool_version,
                    router_version="v6_router_v1",
                    candidate_experts=tuple(old_ids),
                    empty_loss=empty_loss,
                    single_losses={
                        int(k.split("_")[-1]): v
                        for k, v in rows.items() if k != "empty"
                    },
                    pair_losses={},
                    best_single=teacher_set,
                    best_pair=(),
                    pair_gain=None,
                    teacher_set=teacher_set,
                    teacher_loss=teacher_loss,
                    teacher_multi_hot={int(e): 1 for e in teacher_set},
                    cache_key="",
                )
            )
        reuse, residual = build_residual_records(
            teacher_records,
            query_feature_path=str(root / "features" / "train_features.json"),
            min_old_gain=config["residual"]["min_old_gain"],
            top_m_covered=[True] * len(teacher_records),
            split="train",
        )
        _write_json(
            str(root / "residual" / "reuse.json"),
            [record.to_dict() for record in reuse],
        )
        _write_json(
            str(root / "residual" / "residual.json"),
            [record.to_dict() for record in residual],
        )
        _write_json(
            str(root / "residual" / "summary.json"),
            {
                "reuse_count": len(reuse),
                "residual_count": len(residual),
                "teacher_empty_count": sum(1 for r in teacher_records if not r.teacher_set),
            },
        )
        machine.advance(TaskStage.RESIDUAL_READY, note="residual split done")
        machine.save(str(state_path))
        _mark_stage(root, "s2_residual")

    # ---- S3: CLIP query features for residual samples ---------------------
    if not _stage_done(root, "s3_features"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        residual_ids = {record["sample_id"] for record in residual}
        subset = [record for record in records if str(record["id"]) in residual_ids]
        _write_json(str(root / "data" / "residual_train.json"), subset)
        command = [
            PYTHON, "-m", "compose.eval.v6_query_features",
            "--questions", str(root / "data" / "residual_train.json"),
            "--images", IMAGE_FOLDER,
            "--output", str(root / "features" / "train_features.json"),
            "--device", "cuda:0",
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (root / "logs" / "s3_stdout.log").write_text(result.stdout, encoding="utf-8")
        (root / "logs" / "s3_stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("feature extraction failed:\n" + result.stderr[-3000:])
        _mark_stage(root, "s3_features")

    # ---- S4: candidate training (2 slots, conditional residual) -----------
    if not _stage_done(root, "s4_candidates"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        if not residual:
            _write_json(
                str(root / "candidate" / "no_candidate.json"),
                {"reason": "no residual samples; commit_count=0"},
            )
            machine.advance(TaskStage.CANDIDATE_TRAINING, note="no residual -> skip")
            machine.advance(TaskStage.CANDIDATE_TRAINED, note="no residual -> skip")
            machine.advance(TaskStage.CANDIDATE_VALIDATED, note="skip validation")
        else:
            features = json.loads((root / "features" / "train_features.json").read_text())["records"]
            query_ids = [record["sample_id"] for record in residual]
            queries = torch.tensor(
                [features[sample_id]["image"] for sample_id in query_ids],
                dtype=torch.float32,
            )
            config_cand = V6CandidateConfig(
                slot_count=2,
                query_dim=768,
                key_initialization="kmeans_plus_plus",
                key_init_seed=config["candidate"]["key_init_seed"],
            )
            from compose.expansion.v6_candidate import build_v6_candidate_pool

            pool, init_record = build_v6_candidate_pool(
                config_cand, queries, [torch.nn.Linear(1, 1) for _ in range(2)]
            )
            _write_json(str(root / "candidate" / "init_record.json"), init_record)
            # Assignment: selected_slot = argmax cosine(query, key).
            similarities = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(pool.keys, dim=-1).T
            assignments = torch.argmax(similarities, dim=-1).tolist()
            old_ids = [e.expert_id for e in registry.list_all()]
            manifest = []
            for index, record in enumerate(residual):
                manifest.append({
                    "sample_id": record["sample_id"],
                    "teacher_ids": list(record["old_teacher_set"]),
                    "slot": SLOT_IDS[assignments[index]],
                })
            _write_json(str(root / "candidate" / "selections.json"), manifest)
            _write_json(
                str(root / "candidate" / "assignment_stats.json"),
                {"assignments": assignments},
            )
            old_checkpoint = (
                task1_root / "committed" / "expert_{:04d}".format(old_ids[0])
                if old_ids else None
            )
            output = root / "candidate" / "train"
            if output.exists():
                shutil.rmtree(str(output))
            # DDP launch (see v6_task1_dry_run S2 for the rationale):
            # DataParallel cannot split ComposeSelection.
            num_gpus = len([g for g in gpus.split(",") if g.strip()])
            command = [
                PYTHON, "-m", "torch.distributed.run",
                "--nproc_per_node", str(num_gpus),
                "--master_port", str(master_port),
                "-m", "compose.train.train_v6_candidate",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                "--data-path", str(root / "data" / "residual_train.json"),
                "--image-folder", IMAGE_FOLDER,
                "--selection-manifest", str(root / "candidate" / "selections.json"),
                "--candidate-ids", ",".join(map(str, SLOT_IDS)),
                "--old-expert-checkpoint", str(old_checkpoint) if old_checkpoint else "",
                "--output-dir", str(output),
                "--lr", str(config["training"]["learning_rate"]),
                "--epochs", str(config["training"]["epochs_per_task"]),
                "--per-device-batch-size", str(config["training"]["per_device_batch_size"]),
                "--grad-accum", str(config["training"]["grad_accumulation_steps"]),
                "--seed", str(config["data"]["seed"]),
            ]
            if old_checkpoint is None:
                command.remove("--old-expert-checkpoint")
                command.remove("")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, MASTER_PORT=str(master_port))
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            (root / "logs" / "s4_stdout.log").write_text(result.stdout, encoding="utf-8")
            (root / "logs" / "s4_stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError("candidate training failed:\n" + result.stderr[-3000:])
            machine.advance(TaskStage.CANDIDATE_TRAINING, note="candidate training")
            machine.advance(TaskStage.CANDIDATE_TRAINED, note="candidates trained")
            machine.advance(TaskStage.CANDIDATE_VALIDATED, note="validated (gains in S5)")
        machine.save(str(state_path))
        _mark_stage(root, "s4_candidates")

    # ---- S5: validation (conditional gains) --------------------------------
    if not _stage_done(root, "s5_validation"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        if residual:
            val_records = json.loads((root / "data" / "teacher_val.json").read_text())
            old_ids = [e.expert_id for e in registry.list_all()]
            selections = {}
            for record in val_records:
                sample_id = str(record["id"])
                row = {}
                for slot_id in SLOT_IDS:
                    row["old"] = old_ids
                    row["old_plus_{}".format(slot_id)] = old_ids + [slot_id]
                selections[sample_id] = row
            _write_json(str(root / "validation" / "selections.json"), selections)
            command = [
                PYTHON, "-m", "compose.eval.v6_nll_eval",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                "--checkpoint-dir", str(root / "candidate" / "train"),
                "--question-file", str(root / "data" / "teacher_val.json"),
                "--image-folder", IMAGE_FOLDER,
                "--selections", str(root / "validation" / "selections.json"),
                "--output", str(root / "validation" / "nll.json"),
                "--device", "cuda:0",
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            (root / "logs" / "s5_stdout.log").write_text(result.stdout, encoding="utf-8")
            (root / "logs" / "s5_stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError("validation failed:\n" + result.stderr[-3000:])
            nll = json.loads((root / "validation" / "nll.json").read_text())
            summary = {}
            for slot_id in SLOT_IDS:
                gains = [
                    row["old"] - row["old_plus_{}".format(slot_id)]
                    for row in nll.values() if "old_plus_{}".format(slot_id) in row
                ]
                summary[str(slot_id)] = {
                    "samples": len(gains),
                    "mean_gain": sum(gains) / len(gains) if gains else 0.0,
                    "support_count": sum(1 for g in gains if g > 0),
                }
            _write_json(str(root / "validation" / "summary.json"), summary)
        if machine.stage is not TaskStage.CANDIDATE_VALIDATED:
            machine.advance(TaskStage.CANDIDATE_VALIDATED, note="validation done")
        machine.save(str(state_path))
        _mark_stage(root, "s5_validation")

    # ---- S6: commit (0-2, transactional) -----------------------------------
    if not _stage_done(root, "s6_commit"):
        residual = json.loads((root / "residual" / "residual.json").read_text())
        committed = []
        if residual:
            summary = json.loads((root / "validation" / "summary.json").read_text())
            tau_support = config["commit_conditions"]["tau_support"]
            tau_gain = config["commit_conditions"]["tau_gain"]
            transaction = CommitTransaction(str(state_dir), registry)
            for slot_id in SLOT_IDS:
                stats = summary.get(str(slot_id), {})
                if stats.get("support_count", 0) < tau_support or stats.get("mean_gain", -1) < tau_gain:
                    _write_json(
                        str(root / "committed" / "rejected_{}.json".format(slot_id)),
                        {"slot_id": slot_id, "reason": "below_tau", "stats": stats},
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
                    "--old-expert-checkpoint", str(task1_root / "committed" / "expert_0010"),
                    "--output-dir", str(staging),
                ]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                (root / "logs" / "s6_{}_stdout.log".format(slot_id)).write_text(result.stdout, encoding="utf-8")
                (root / "logs" / "s6_{}_stderr.log".format(slot_id)).write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError("assemble {} failed:\n{}".format(slot_id, result.stderr[-3000:]))
                assembly = json.loads((staging / "assembly.json").read_text())
                from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata

                metadata = ExpertMetadata(
                    expert_id=slot_id,
                    adapter_name="expert_{:04d}".format(slot_id),
                    rank=8, alpha=16.0,
                    creation_task=1, creation_task_name=TASK_NAME,
                    created_seed=config["data"]["seed"],
                    checkpoint_path=str(staging / "compose_experts.bin"),
                    checkpoint_sha256=assembly["compose_experts_bin_sha256"],
                    lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
                    support_count=stats["support_count"],
                    mean_conditional_gain=stats["mean_gain"],
                )
                transaction.begin(slot_id, {"task_id": 1, "stage": "task2_commit"})
                transaction.complete(
                    slot_id,
                    artifacts={
                        str(staging / "compose_experts.bin"): assembly["compose_experts_bin_sha256"],
                        str(staging / "compose_experts.json"): assembly["compose_experts_json_sha256"],
                    },
                    condition_record={
                        "task_id": 1,
                        "support_count": stats["support_count"],
                        "mean_conditional_gain": stats["mean_gain"],
                    },
                    metadata=metadata,
                )
                committed.append(slot_id)
            registry.save_json(str(registry_path))
        _write_json(
            str(root / "committed" / "commit_record.json"),
            {"committed_expert_ids": committed, "pool_version": registry.pool_version},
        )
        machine.advance(TaskStage.EXPERTS_COMMITTED, note="commit done")
        machine.save(str(state_path))
        _mark_stage(root, "s6_commit")

    # ---- S7: router (global teacher + calibration) -------------------------
    if not _stage_done(root, "s7_router"):
        old_ids = [e.expert_id for e in registry.list_all()]
        router = V6Router(V6QueryEncoder(), seed=config["data"]["seed"])
        for expert in registry.list_all():
            router.add_expert(
                expert.expert_id, creation_task=expert.creation_task or 0,
                checkpoint_sha256=expert.checkpoint_sha256 or "",
            )
        save_v6_router_checkpoint(
            str(root / "router" / "router_checkpoint.pt"),
            router,
            pool_version=registry.pool_version,
            config_hash=config.get("config_hash", "dry-run"),
            extra={"task_id": 1, "mode": "calibrated"},
        )
        machine.advance(TaskStage.GLOBAL_TEACHER_READY, note="global teacher regenerated")
        machine.advance(TaskStage.ROUTER_TRAINING, note="router calibrating")
        machine.advance(TaskStage.ROUTER_READY, note="router calibrated")
        machine.save(str(state_path))
        _mark_stage(root, "s7_router")

    # ---- S8: RMS ------------------------------------------------------------
    if not _stage_done(root, "s8_rms"):
        machine.advance(TaskStage.RMS_READY, note="rms marked")
        machine.save(str(state_path))
        _mark_stage(root, "s8_rms")

    # ---- S9: snapshot ---------------------------------------------------------
    if not _stage_done(root, "s9_snapshot"):
        git_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        V6Snapshot.create(
            str(root / "snapshots" / "task1"),
            task_id=1, task_name=TASK_NAME,
            registry=registry, task_state=machine,
            git_commit=git_commit, command="v6_task2_dry_run",
            data_hash=hashlib.sha256(
                open(train_path, "rb").read(1 << 20)
            ).hexdigest(),
        )
        machine.advance(TaskStage.SNAPSHOT_READY, note="snapshot written")
        machine.save(str(state_path))
        _mark_stage(root, "s9_snapshot")

    # ---- S10: eval ------------------------------------------------------------
    if not _stage_done(root, "s10_eval"):
        commit_record = json.loads((root / "committed" / "commit_record.json").read_text())
        eval_root = root / "eval" / "task1"
        eval_root.mkdir(parents=True, exist_ok=True)
        expert_ids = commit_record["committed_expert_ids"]
        if expert_ids:
            expert_args = ["--expert-ids", ",".join(map(str, expert_ids)),
                           "--gates", ",".join(["1.0"] * len(expert_ids))]
        else:
            expert_args = ["--expert-ids", ""]
        checkpoint_dir = (
            root / "candidate" / "train"
            if (root / "candidate" / "train" / "compose_experts.json").is_file()
            else task1_root / "committed" / "expert_0010"
        )
        command = [
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
        ] + expert_args
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (root / "logs" / "s10_stdout.log").write_text(result.stdout, encoding="utf-8")
        (root / "logs" / "s10_stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("eval failed:\n" + result.stderr[-3000:])
        machine.advance(TaskStage.EVALUATION_COMPLETE, note="eval done")
        machine.advance(TaskStage.COMPLETED, note="task 2 complete")
        machine.save(str(state_path))
        _mark_stage(root, "s10_eval")

    print("Task 2 stage: {}".format(machine.stage.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task1-root", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--master-port", type=int, default=29671)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        import yaml

        config = yaml.safe_load(handle) if args.config else {}
    run_task2(Path(args.output_root), Path(args.task1_root), args.gpus, args.master_port, config)
