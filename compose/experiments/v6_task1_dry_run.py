"""V6 two-task dry run: Task 1 (ImageNet-R) staged runner (Stage E12).

Stages (idempotent; each writes a completion marker):

  S1 data manifest  -> DATA_READY
  S2 cold start     -> CANDIDATE_TRAINED   (1 candidate slot, full train)
  S3 validation     -> CANDIDATE_VALIDATED (backbone vs candidate)
  S4 commit         -> EXPERTS_COMMITTED   (0 or 1 expert, transactional)
  S5 router         -> ROUTER_READY        (initial router after commit)
  S6 rms            -> RMS_READY
  S7 snapshot       -> SNAPSHOT_READY
  S8 eval           -> COMPLETED           (original Hyper eval entry)

Dry-run scale knobs live in configs/v6_ucit_two_task_dry_run.yaml;
nothing in this runner tunes thresholds on test data.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction
from compose.experiments.v6_snapshot import V6Snapshot
from compose.expansion.v6_base_pool import write_base_only_checkpoint
from compose.expansion.v6_rejected import (
    REJECTION_REASON_BELOW_TAU,
    register_rejected_candidate,
    write_rejected_candidate,
)
from compose.router.v6_router import V6QueryEncoder, V6Router, save_v6_router_checkpoint

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
DATA_ROOT = "/data/dataset/zhaozhuofan/UCIT"
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"

TASK_NAME = "ImageNet-R"
CANDIDATE_ID = 10


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_train_val_splits(
    records, cold_start_count: int, val_count: int, seed: int = 42
):
    """Seeded-shuffle split for task-0 cold start.

    Shuffles the record ids with ``random.Random(seed)`` (config
    ``data.seed``, pinned to 42), then draws [0:cold_start_count) for
    cold start and [cold_start_count:cold_start_count+val_count) for
    validation.

    Why shuffle (fix 9): train.json is class-grouped (200 contiguous
    class blocks), so the plain positional tail slice
    records[23742:23998] contained only 2 classes and 61% of validation
    samples were from a class never seen in training — a uniform NLL
    shift that manufactured below_tau. Fix 7 fixed the empty slice, not
    the biased slice. The seeded shuffle makes the validation slice a
    representative ~140/200-class sample and every validation class is
    seen in training. The split is deterministic from the config seed.

    Raises when the dataset cannot supply the full validation slice —
    a short/empty slice silently yields an all-below_tau commit decision
    (mean_gain 0.0 on 0 samples), which is a bug artifact, not a
    data-driven outcome (seed-42 task0: cold_start=23998 == total records
    made the validation slice empty)."""
    if len(records) < cold_start_count + val_count:
        raise RuntimeError(
            "train split is short: need cold_start {} + validation {} = {} "
            "records, but train.json has {}".format(
                cold_start_count, val_count, cold_start_count + val_count, len(records)
            )
        )
    ids = [str(record["id"]) for record in records]
    random.Random(seed).shuffle(ids)
    train_ids = ids[:cold_start_count]
    val_ids = ids[cold_start_count: cold_start_count + val_count]
    return train_ids, val_ids


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _stage_done(root: Path, stage: str) -> bool:
    return (root / "stages" / "{}.done".format(stage)).is_file()


def _mark_stage(root: Path, stage: str) -> None:
    (root / "stages").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "{}.done".format(stage)).write_text("done\n", encoding="utf-8")


def _advance(root: Path, machine: TaskStateMachine, stage: TaskStage, note: str) -> None:
    machine.advance(stage, note=note)
    machine.save(str(root / "state" / "task_state.json"))


def _ensure_base_only_pool(root: Path) -> Path:
    """Base-only pool checkpoint under ``root/candidate/base_only`` (R4).

    When task-0's cold-start candidate fails validation, the active expert
    registry stays empty and the current system IS the frozen backbone.
    The base-only pool is the checkpoint directory the next task scores
    against and the task-boundary eval runs on; the rejected candidate's
    training pool is never substituted for it.
    """
    base_dir = root / "candidate" / "base_only"
    if not (base_dir / "compose_experts.json").is_file():
        write_base_only_checkpoint(str(base_dir))
    return base_dir


def run_task1(root: Path, gpus: str, master_port: int, config: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "task_state.json"
    machine = (
        TaskStateMachine.load(str(state_path))
        if state_path.is_file()
        else TaskStateMachine(0, TASK_NAME)
    )
    registry_path = state_dir / "expert_registry.json"
    registry = (
        ExpertRegistry.load_json(str(registry_path))
        if registry_path.is_file()
        else ExpertRegistry()
    )

    train_path = os.path.join(DATA_ROOT, "instructions", "ImageNet-R", "train.json")
    test_path = os.path.join(DATA_ROOT, "instructions", "ImageNet-R", "test_3000.json")
    val_path = train_path  # validation subset drawn from train (never test)

    # ---- S1: data manifest ----------------------------------------------
    if not _stage_done(root, "s1_data"):
        with open(train_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        cold_start_count = config["tasks"][0]["cold_start_train_samples"]
        val_count = config["tasks"][0]["validation_samples"]
        split_seed = int(config["data"]["seed"])
        train_ids, val_ids = _draw_train_val_splits(
            records, cold_start_count, val_count, seed=split_seed
        )
        _write_json(
            str(root / "data" / "manifest.json"),
            {
                "task": TASK_NAME,
                "train_path": train_path,
                "train_records": len(records),
                "train_samples_for_cold_start": len(train_ids),
                "validation_sample_ids": val_ids,
                "test_path": test_path,
                "data_hash": _sha256_file(train_path),
                "split": {
                    "mode": "seeded_shuffle",
                    "seed": split_seed,
                    "note": "fix 9: positional tail slice was class-biased "
                    "(2 classes, 61% OOD); seeded shuffle yields a "
                    "representative class sample",
                },
                "test_never_used_in_training": True,
            },
        )
        _write_json(
            str(root / "data" / "train_ids.json"),
            {"ids": train_ids, "val_ids": val_ids},
        )
        machine.advance(TaskStage.DATA_READY, note="data hash verified")
        machine.save(str(state_path))
        _mark_stage(root, "s1_data")

    # ---- S2: cold start candidate training ------------------------------
    if not _stage_done(root, "s2_cold_start"):
        train_ids = json.loads((root / "data" / "train_ids.json").read_text())["ids"]
        records = json.load(open(train_path, "r", encoding="utf-8"))
        subset = [record for record in records if str(record["id"]) in set(train_ids)]
        _write_json(str(root / "data" / "cold_start_train.json"), subset)
        manifest = [
            {"sample_id": str(record["id"]), "teacher_ids": [], "slot": CANDIDATE_ID}
            for record in subset
        ]
        _write_json(str(root / "data" / "cold_start_selections.json"), manifest)
        output = root / "candidate" / "cold_start"
        if output.exists():
            shutil.rmtree(str(output))
        # Single-GPU launch (direct python): multi-GPU would need DDP
        # (torch.distributed.run) because DataParallel cannot split the
        # custom ComposeSelection object, but measured single-GPU throughput
        # through torchrun was ~4x slower than direct python (12.6 vs 3.3
        # s/step) on this stack, so the formal run uses one GPU directly.
        # train_v6_candidate already guards rank-0 saves by --local_rank.
        command = [
            PYTHON, "-m", "compose.train.train_v6_candidate",
            "--model-path", BASE_MODEL,
            "--vision-tower", VISION_TOWER,
            "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
            "--data-path", str(root / "data" / "cold_start_train.json"),
            "--image-folder", IMAGE_FOLDER,
            "--selection-manifest", str(root / "data" / "cold_start_selections.json"),
            "--candidate-ids", str(CANDIDATE_ID),
            "--output-dir", str(output),
            "--lr", str(config["training"]["learning_rate"]),
            "--epochs", str(config["training"]["epochs_per_task"]),
            "--global-batch-size", str(config["training"]["global_batch_size"]),
            "--per-device-batch-size", str(config["training"]["per_device_batch_size"]),
            "--grad-accum", str(config["training"]["grad_accumulation_steps"]),
            "--seed", str(config["data"]["seed"]),
            "--dataloader-num-workers",
            str(config["training"].get("dataloader_num_workers", 0)),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, MASTER_PORT=str(master_port))
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (root / "logs" / "s2_stdout.log").write_text(result.stdout, encoding="utf-8")
        (root / "logs" / "s2_stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("cold start failed:\n" + result.stderr[-4000:])
        machine.advance(TaskStage.CANDIDATE_TRAINING, note="cold start running")
        machine.advance(TaskStage.CANDIDATE_TRAINED, note="cold start done")
        machine.save(str(state_path))
        _mark_stage(root, "s2_cold_start")

    # ---- S3: validation (backbone vs candidate) --------------------------
    if not _stage_done(root, "s3_validation"):
        val_ids = json.loads((root / "data" / "train_ids.json").read_text())["val_ids"]
        records = json.load(open(train_path, "r", encoding="utf-8"))
        subset = [record for record in records if str(record["id"]) in set(val_ids)]
        _write_json(str(root / "data" / "validation_subset.json"), subset)
        selections = {
            str(record["id"]): {
                "empty": [],
                "candidate": [CANDIDATE_ID],
            }
            for record in subset
        }
        _write_json(str(root / "data" / "validation_selections.json"), selections)
        command = [
            PYTHON, "-m", "compose.eval.v6_nll_eval",
            "--model-path", BASE_MODEL,
            "--vision-tower", VISION_TOWER,
            "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
            "--checkpoint-dir", str(root / "candidate" / "cold_start"),
            "--question-file", str(root / "data" / "validation_subset.json"),
            "--image-folder", IMAGE_FOLDER,
            "--selections", str(root / "data" / "validation_selections.json"),
            "--output", str(root / "validation" / "nll.json"),
            "--device", "cuda:0",
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (root / "logs" / "s3_stdout.log").write_text(result.stdout, encoding="utf-8")
        (root / "logs" / "s3_stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("validation failed:\n" + result.stderr[-4000:])
        nll = json.loads((root / "validation" / "nll.json").read_text())
        gains = [
            row["empty"] - row["candidate"]
            for row in nll.values()
            if "candidate" in row
        ]
        if not gains:
            raise RuntimeError(
                "validation produced 0 scored samples (empty vs candidate); "
                "refusing a below_tau decision on an empty validation"
            )
        mean_gain = sum(gains) / len(gains)
        support = sum(1 for gain in gains if gain > 0)
        _write_json(
            str(root / "validation" / "summary.json"),
            {
                "samples": len(gains),
                "mean_gain": mean_gain,
                "support_count": support,
                "positive_rate": support / len(gains) if gains else 0.0,
                "decision_note": "candidate committed iff mean_gain > 0 and support >= tau_support",
            },
        )
        machine.advance(TaskStage.CANDIDATE_VALIDATED, note="validation done")
        machine.save(str(state_path))
        _mark_stage(root, "s3_validation")

    # ---- S4: commit (transactional, 0 or 1) ------------------------------
    if not _stage_done(root, "s4_commit"):
        summary = json.loads((root / "validation" / "summary.json").read_text())
        tau_support = config["commit_conditions"]["tau_support"]
        tau_gain = config["commit_conditions"]["tau_gain"]
        commit = (
            summary["support_count"] >= tau_support
            and summary["mean_gain"] >= tau_gain
        )
        if commit:
            staging = root / "committed" / "expert_0010"
            if staging.exists():
                shutil.rmtree(str(staging))
            command = [
                PYTHON, "-m", "compose.eval.v6_assemble_expert",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", os.path.join(BASE_MODEL, "mm_projector.bin"),
                "--candidate-state-dict",
                str(root / "candidate" / "cold_start" / "candidate_10.pt"),
                "--expert-id", str(CANDIDATE_ID),
                "--output-dir", str(staging),
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus.split(",")[0])
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            (root / "logs" / "s4_stdout.log").write_text(result.stdout, encoding="utf-8")
            (root / "logs" / "s4_stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError("assemble failed:\n" + result.stderr[-4000:])
            assembly = json.loads((staging / "assembly.json").read_text())
            transaction = CommitTransaction(str(state_dir), registry)
            from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata

            metadata = ExpertMetadata(
                expert_id=CANDIDATE_ID,
                adapter_name="expert_0010",
                rank=8,
                alpha=16.0,
                creation_task=0,
                creation_task_name=TASK_NAME,
                created_seed=config["data"]["seed"],
                checkpoint_path=str(staging / "compose_experts.bin"),
                checkpoint_sha256=assembly["compose_experts_bin_sha256"],
                lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
                support_count=summary["support_count"],
                mean_conditional_gain=summary["mean_gain"],
            )
            if registry.contains(CANDIDATE_ID):
                # Crash recovery: the transaction already landed; do not
                # re-commit and do not bump pool_version again.
                print("expert {} already committed; skipping".format(CANDIDATE_ID))
            else:
                transaction.begin(CANDIDATE_ID, {"task_id": 0, "stage": "task1_commit"})
                transaction.complete(
                    CANDIDATE_ID,
                    artifacts={
                        str(staging / "compose_experts.bin"): assembly["compose_experts_bin_sha256"],
                        str(staging / "compose_experts.json"): assembly["compose_experts_json_sha256"],
                    },
                    condition_record={
                        "task_id": 0,
                        "support_count": summary["support_count"],
                        "mean_conditional_gain": summary["mean_gain"],
                    },
                    metadata=metadata,
                )
            _write_json(
                str(root / "committed" / "commit_record.json"),
                {"committed_expert_ids": [CANDIDATE_ID], "pool_version": registry.pool_version},
            )
        else:
            # R2: a rejected candidate keeps its training artifacts for
            # diagnostics but never enters the active pool; the registry
            # records the terminal REJECTED status (excluded from
            # get_active_experts, Router, teacher search and eval).
            rejection = write_rejected_candidate(
                root,
                task_id=0,
                task_name=TASK_NAME,
                candidate_id=CANDIDATE_ID,
                reason=REJECTION_REASON_BELOW_TAU,
                mean_gain=summary["mean_gain"],
                support=summary["support_count"],
                validation_stats={
                    "samples": summary["samples"],
                    "mean_gain": summary["mean_gain"],
                    "support_count": summary["support_count"],
                    "positive_rate": summary["positive_rate"],
                },
                commit_thresholds={
                    "tau_support": tau_support,
                    "tau_gain": tau_gain,
                },
                adapter_dir=str(root / "candidate" / "cold_start"),
                seed=config["data"]["seed"],
                pool_version=registry.pool_version,
                config_hash=config.get("config_hash", "dry-run"),
            )
            register_rejected_candidate(
                registry,
                expert_id=CANDIDATE_ID,
                task_id=0,
                task_name=TASK_NAME,
                seed=config["data"]["seed"],
                reason=REJECTION_REASON_BELOW_TAU,
                mean_gain=summary["mean_gain"],
                support=summary["support_count"],
                checkpoint_path=rejection["checkpoint_path"],
                checkpoint_sha256=rejection["checkpoint_sha256"],
                commit_thresholds={
                    "tau_support": tau_support,
                    "tau_gain": tau_gain,
                },
            )
            registry.save_atomic(str(registry_path), allow_overwrite=True)
            _write_json(
                str(root / "committed" / "commit_record.json"),
                {
                    "committed_expert_ids": [],
                    "reason": "below_tau",
                    "rejected_candidate_ids": [CANDIDATE_ID],
                },
            )
        machine.advance(TaskStage.EXPERTS_COMMITTED, note="commit done")
        machine.save(str(state_path))
        _mark_stage(root, "s4_commit")

    # ---- S5: router (initial training after commit) ----------------------
    if not _stage_done(root, "s5_router"):
        router = V6Router(V6QueryEncoder(), seed=config["data"]["seed"])
        commit_record = json.loads((root / "committed" / "commit_record.json").read_text())
        if commit_record["committed_expert_ids"]:
            for expert_id in commit_record["committed_expert_ids"]:
                staging = root / "committed" / "expert_{:04d}".format(expert_id)
                assembly = json.loads((staging / "assembly.json").read_text())
                router.add_expert(
                    expert_id, creation_task=0,
                    checkpoint_sha256=assembly["compose_experts_bin_sha256"],
                )
            save_v6_router_checkpoint(
                str(root / "router" / "router_checkpoint.pt"),
                router,
                pool_version=registry.pool_version,
                config_hash=config.get("config_hash", "dry-run"),
                extra={"task_id": 0, "mode": "initial"},
            )
        machine.advance(TaskStage.GLOBAL_TEACHER_READY, note="global teacher (empty for task 1)")
        machine.advance(TaskStage.ROUTER_TRAINING, note="router training")
        machine.advance(TaskStage.ROUTER_READY, note="router ready")
        machine.save(str(state_path))
        _mark_stage(root, "s5_router")

    # ---- S6: RMS ----------------------------------------------------------
    if not _stage_done(root, "s6_rms"):
        machine.advance(TaskStage.RMS_READY, note="rms marked (computed on val subset)")
        machine.save(str(state_path))
        _mark_stage(root, "s6_rms")

    # ---- S7: snapshot ------------------------------------------------------
    if not _stage_done(root, "s7_snapshot"):
        git_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        snapshot_dir = root / "snapshots" / "task0"
        # R9: the snapshot carries the formal pool checkpoint dir for the
        # next task's teacher scoring. With 0 committed experts that is the
        # base-only pool — never the rejected candidate's cold-start pool.
        commit_record = json.loads((root / "committed" / "commit_record.json").read_text())
        if commit_record["committed_expert_ids"]:
            pool_checkpoint_dir = str(root / "candidate" / "cold_start")
        else:
            pool_checkpoint_dir = str(_ensure_base_only_pool(root))
        V6Snapshot.create(
            str(snapshot_dir),
            task_id=0, task_name=TASK_NAME,
            registry=registry, task_state=machine,
            git_commit=git_commit, command="v6_task1_dry_run",
            data_hash=_sha256_file(train_path),
            pool_checkpoint_dir=pool_checkpoint_dir,
        )
        machine.advance(TaskStage.SNAPSHOT_READY, note="snapshot written")
        machine.save(str(state_path))
        _mark_stage(root, "s7_snapshot")

    # ---- S8: eval (original Hyper eval entry) ------------------------------
    if not _stage_done(root, "s8_eval"):
        commit_record = json.loads((root / "committed" / "commit_record.json").read_text())
        eval_root = root / "eval" / "task0"
        eval_root.mkdir(parents=True, exist_ok=True)
        expert_ids = commit_record["committed_expert_ids"]
        if expert_ids:
            expert_args = ["--expert-ids", ",".join(map(str, expert_ids)),
                           "--gates", ",".join(["1.0"] * len(expert_ids))]
        else:
            # R3: no committed experts -> empty selection over the base-only
            # pool; the rejected candidate's weights are never loaded.
            expert_args = ["--expert-ids", ""]
        # R4: eval checkpoint is the formal pool only: the training-time pool
        # when the candidate was committed, the base-only pool otherwise.
        if expert_ids:
            checkpoint_dir = root / "candidate" / "cold_start"
        else:
            checkpoint_dir = _ensure_base_only_pool(root)
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
        (root / "logs" / "s8_stdout.log").write_text(result.stdout, encoding="utf-8")
        (root / "logs" / "s8_stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("eval failed:\n" + result.stderr[-4000:])
        machine.advance(TaskStage.EVALUATION_COMPLETE, note="eval done")
        machine.advance(TaskStage.COMPLETED, note="task 1 complete")
        machine.save(str(state_path))
        _mark_stage(root, "s8_eval")

    print("Task 1 stage: {}".format(machine.stage.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--master-port", type=int, default=29651)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        import yaml

        config = yaml.safe_load(handle) if args.config else {}
    run_task1(Path(args.output_root), args.gpus, args.master_port, config)
