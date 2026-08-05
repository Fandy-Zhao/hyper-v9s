"""V6 task-boundary snapshots and resume analysis (Stage E10).

A snapshot is a self-contained directory: it can be loaded independently
without any previous stage's temporary directories. It captures:

- stage manifest (schema, git commit, command, environment, GPU info,
  data hash, config copy, component hashes)
- task id/name, expert registry (+ pool_version), task state machine
- LoRA checkpoint references + hashes
- query encoder / expert keys / router params and thresholds
- RMS statistics
- lifecycle status, optimizer/scheduler (if any), random states
  (Python/NumPy/PyTorch/CUDA)
- teacher cache manifest, residual manifest, candidate validation
- stdout/stderr and the original Hyper eval output (runner fills these)

Resume analysis maps the persisted state machine onto the six recovery
nodes from the task book (old teacher mid-run, candidate epoch mid-run,
candidate validated not committed, candidate committed, router
calibrating, snapshot ready but eval not done) and cleans up any
half-finished commit transactions (CommitTransaction.resume), which is
idempotent: no duplicate commits, no duplicate pool_version bumps, and
old expert hashes stay unchanged.
"""

import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStateMachine, TaskStage
from compose.experts.transaction import CommitTransaction

SNAPSHOT_SCHEMA_VERSION = 1

MANIFEST_NAME = "manifest.json"
REGISTRY_NAME = "expert_registry.json"
TASK_STATE_NAME = "task_state.json"
ROUTER_NAME = "router_checkpoint.pt"
RMS_NAME = "rms_statistics.json"
RANDOM_STATES_NAME = "random_states.pt"
TEACHER_CACHE_MANIFEST_NAME = "teacher_cache_manifest.json"
RESIDUAL_MANIFEST_NAME = "residual_manifest.json"
CANDIDATE_VALIDATION_NAME = "candidate_validation.json"
LORA_DIR_NAME = "lora_checkpoints"
EVAL_OUTPUT_DIR_NAME = "eval_output"
CONFIG_COPY_NAME = "config.yaml"
STDOUT_NAME = "stdout.log"
STDERR_NAME = "stderr.log"

#: Snapshot completeness: files that must exist for a loadable snapshot.
REQUIRED_COMPONENTS = (
    MANIFEST_NAME,
    REGISTRY_NAME,
    TASK_STATE_NAME,
    RANDOM_STATES_NAME,
)

#: Recovery nodes (task book Stage E10 list).
RESUME_NODES = (
    "old_teacher_running",
    "candidate_epoch_running",
    "candidate_validated_not_committed",
    "candidate_committed",
    "router_calibrating",
    "snapshot_ready_eval_pending",
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
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


def _gpu_inventory() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()
        return {"available": True, "devices": output}
    except (OSError, subprocess.SubprocessError):
        return {
            "available": True,
            "devices": [
                "{}: {} MiB".format(index, torch.cuda.get_device_properties(index))
                for index in range(torch.cuda.device_count())
            ],
        }


def _environment() -> Dict[str, Any]:
    import importlib

    def version(name):
        try:
            module = importlib.import_module(name)
            return getattr(module, "__version__", "unknown")
        except ImportError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "deepspeed": version("deepspeed"),
        "numpy": version("numpy"),
        "cuda": torch.version.cuda,
    }


def _random_states() -> Dict[str, Any]:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = [
            torch.cuda.get_rng_state(index).tolist()
            for index in range(torch.cuda.device_count())
        ]
    return states


def _restore_random_states(states: Dict[str, Any]) -> None:
    random.setstate(tuple(states["python"]))
    np.random.set_state(states["numpy"])
    torch.set_rng_state(torch.tensor(states["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in states:
        for index, state in enumerate(states["torch_cuda"]):
            if index < torch.cuda.device_count():
                torch.cuda.set_rng_state(torch.tensor(state, dtype=torch.uint8), index)


@dataclass
class V6Snapshot:
    """A self-contained task-boundary snapshot."""

    directory: str
    manifest: Dict[str, Any]
    registry: ExpertRegistry
    task_state: TaskStateMachine
    random_states: Dict[str, Any]
    # Optional components (None when not yet produced at save time).
    router_path: Optional[str] = None
    rms_path: Optional[str] = None
    teacher_cache_manifest: Optional[Dict[str, Any]] = None
    residual_manifest: Optional[Dict[str, Any]] = None
    candidate_validation: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None

    @property
    def path(self) -> Path:
        return Path(self.directory)

    def registry_path(self) -> Path:
        return self.path / REGISTRY_NAME

    def task_state_path(self) -> Path:
        return self.path / TASK_STATE_NAME

    def lora_checkpoint_dir(self) -> Path:
        return self.path / LORA_DIR_NAME

    def eval_output_dir(self) -> Path:
        return self.path / EVAL_OUTPUT_DIR_NAME

    def verify_complete(self) -> List[str]:
        """Return missing required components; empty means loadable."""
        missing = []
        for name in REQUIRED_COMPONENTS:
            if not (self.path / name).is_file():
                missing.append(name)
        return missing

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        directory: str,
        task_id: int,
        task_name: str,
        registry: ExpertRegistry,
        task_state: TaskStateMachine,
        git_commit: str,
        command: str,
        data_hash: str,
        config_copy_path: Optional[str] = None,
        router: Optional[Any] = None,
        router_config_hash: Optional[str] = None,
        rms_stats: Optional[Any] = None,
        teacher_cache_manifest: Optional[Dict[str, Any]] = None,
        residual_manifest: Optional[Dict[str, Any]] = None,
        candidate_validation: Optional[Dict[str, Any]] = None,
        stdout_text: str = "",
        stderr_text: str = "",
    ) -> "V6Snapshot":
        """Write a complete snapshot atomically (manifest last)."""
        root = Path(directory)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("snapshot directory is not empty: {}".format(root))
        root.mkdir(parents=True, exist_ok=True)

        # Core components.
        registry.save_json(str(root / REGISTRY_NAME))
        task_state.save(str(root / TASK_STATE_NAME))
        torch.save(_random_states(), str(root / RANDOM_STATES_NAME))
        if config_copy_path and Path(config_copy_path).is_file():
            shutil.copyfile(config_copy_path, root / CONFIG_COPY_NAME)
        if router is not None:
            from compose.router.v6_router import save_v6_router_checkpoint

            save_v6_router_checkpoint(
                str(root / ROUTER_NAME), router,
                pool_version=registry.pool_version,
                config_hash=router_config_hash or "",
            )
        if rms_stats is not None:
            rms_stats.save_json(str(root / RMS_NAME))
        if teacher_cache_manifest is not None:
            _atomic_write_json(root / TEACHER_CACHE_MANIFEST_NAME, teacher_cache_manifest)
        if residual_manifest is not None:
            _atomic_write_json(root / RESIDUAL_MANIFEST_NAME, residual_manifest)
        if candidate_validation is not None:
            _atomic_write_json(root / CANDIDATE_VALIDATION_NAME, candidate_validation)
        (root / EVAL_OUTPUT_DIR_NAME).mkdir(exist_ok=True)
        (root / LORA_DIR_NAME).mkdir(exist_ok=True)
        if stdout_text:
            (root / STDOUT_NAME).write_text(stdout_text, encoding="utf-8")
        if stderr_text:
            (root / STDERR_NAME).write_text(stderr_text, encoding="utf-8")

        # Component hashes for integrity checks.
        component_hashes = {}
        for name in (REGISTRY_NAME, TASK_STATE_NAME, RANDOM_STATES_NAME):
            component_hashes[name] = file_sha256(str(root / name))

        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "task_id": int(task_id),
            "task_name": str(task_name),
            "pool_version": registry.pool_version,
            "git_commit": str(git_commit),
            "command": str(command),
            "environment": _environment(),
            "gpu_inventory": _gpu_inventory(),
            "data_hash": str(data_hash),
            "component_hashes": component_hashes,
            "has_router": router is not None,
            "has_rms": rms_stats is not None,
            "has_teacher_cache_manifest": teacher_cache_manifest is not None,
            "has_residual_manifest": residual_manifest is not None,
            "has_candidate_validation": candidate_validation is not None,
        }
        _atomic_write_json(root / MANIFEST_NAME, manifest)
        return cls.load(str(root))

    # ------------------------------------------------------------------
    # Load (independent of any previous stage directories)
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, directory: str) -> "V6Snapshot":
        root = Path(directory)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError("snapshot manifest missing: {}".format(manifest_path))
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("schema_version", -1)) != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported snapshot schema_version: {}".format(
                    manifest.get("schema_version")
                )
            )
        # Integrity first: component hashes must match before any parsing.
        for name, expected in manifest.get("component_hashes", {}).items():
            path = root / name
            if not path.is_file():
                continue
            actual = file_sha256(str(path))
            if actual != expected:
                raise ValueError(
                    "snapshot component hash mismatch for {}: expected {}, got {}".format(
                        name, expected, actual
                    )
                )
        missing = [
            name for name in REQUIRED_COMPONENTS if not (root / name).is_file()
        ]
        if missing:
            raise ValueError(
                "snapshot is incomplete (missing {}): {}".format(missing, root)
            )
        registry = ExpertRegistry.load_json(str(root / REGISTRY_NAME))
        task_state = TaskStateMachine.load(str(root / TASK_STATE_NAME))
        random_states = torch.load(
            str(root / RANDOM_STATES_NAME), map_location="cpu", weights_only=False
        )
        return cls(
            directory=str(root),
            manifest=manifest,
            registry=registry,
            task_state=task_state,
            random_states=random_states,
            router_path=str(root / ROUTER_NAME) if (root / ROUTER_NAME).is_file() else None,
            rms_path=str(root / RMS_NAME) if (root / RMS_NAME).is_file() else None,
            teacher_cache_manifest=(
                json.loads((root / TEACHER_CACHE_MANIFEST_NAME).read_text(encoding="utf-8"))
                if (root / TEACHER_CACHE_MANIFEST_NAME).is_file() else None
            ),
            residual_manifest=(
                json.loads((root / RESIDUAL_MANIFEST_NAME).read_text(encoding="utf-8"))
                if (root / RESIDUAL_MANIFEST_NAME).is_file() else None
            ),
            candidate_validation=(
                json.loads((root / CANDIDATE_VALIDATION_NAME).read_text(encoding="utf-8"))
                if (root / CANDIDATE_VALIDATION_NAME).is_file() else None
            ),
            config_path=str(root / CONFIG_COPY_NAME) if (root / CONFIG_COPY_NAME).is_file() else None,
        )

    def load_router(self):
        """Load the router checkpoint lazily (only when present)."""
        if self.router_path is None:
            return None
        from compose.router.v6_router import (
            V6QueryEncoder,
            V6Router,
            load_v6_router_checkpoint,
        )

        router = V6Router(V6QueryEncoder(), seed=0)
        extra = load_v6_router_checkpoint(self.router_path, router)
        router.validate_pool_version(self.registry.pool_version)
        return router, extra

    def load_rms(self, expected_provenance: Mapping[str, Any]):
        """Load RMS statistics; provenance must be supplied by the runner
        (checkpoint/dataset/config hashes are not stored inside the snapshot
        registry)."""
        from compose.lora.statistics import RMSStatistics

        if self.rms_path is None:
            return None
        return RMSStatistics.load_json(
            self.rms_path, expected_provenance=expected_provenance
        )

    def verify_expert_hashes_unchanged(self, expected: Mapping[int, str]) -> List[str]:
        """Cross-check registered expert checkpoint hashes against ``expected``;
        returns any mismatches (recovery must never change old experts)."""
        mismatches = []
        for expert_id, expected_hash in expected.items():
            metadata = self.registry.get(expert_id)
            if str(metadata.checkpoint_sha256) != str(expected_hash):
                mismatches.append(
                    "expert {} hash changed: {} != {}".format(
                        expert_id, metadata.checkpoint_sha256, expected_hash
                    )
                )
        return mismatches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directory": self.directory,
            "task_id": self.manifest["task_id"],
            "task_name": self.manifest["task_name"],
            "pool_version": self.manifest["pool_version"],
            "git_commit": self.manifest["git_commit"],
            "stage": self.task_state.stage.value,
        }


def analyze_resume(
    snapshot_dir: str,
    registry_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Map snapshot state onto the six recovery nodes and clean pending
    commit transactions idempotently.

    Returns ``{"stage", "resume_node", "pending_transactions", "cleanups",
    "can_resume"}``. ``registry_dir`` defaults to the snapshot directory
    (transactions live next to the registry).
    """
    snapshot = V6Snapshot.load(snapshot_dir)
    transaction_dir = registry_dir or snapshot_dir
    registry, incomplete, completed = CommitTransaction.resume(transaction_dir)
    stage = snapshot.task_state.stage
    resume_node = None
    if stage in (TaskStage.OLD_TEACHER_RUNNING,):
        resume_node = "old_teacher_running"
    elif stage in (TaskStage.CANDIDATE_TRAINING, TaskStage.CANDIDATE_TRAINED):
        resume_node = "candidate_epoch_running"
    elif stage is TaskStage.CANDIDATE_VALIDATED:
        resume_node = "candidate_validated_not_committed"
    elif stage in (TaskStage.EXPERTS_COMMITTED,):
        resume_node = "candidate_committed"
    elif stage in (TaskStage.ROUTER_TRAINING, TaskStage.ROUTER_READY,
                   TaskStage.RMS_READY, TaskStage.GLOBAL_TEACHER_READY):
        resume_node = "router_calibrating"
    elif stage in (TaskStage.SNAPSHOT_READY, TaskStage.EVALUATION_COMPLETE):
        resume_node = "snapshot_ready_eval_pending"
    elif stage is TaskStage.COMPLETED:
        resume_node = "completed"
    return {
        "stage": stage.value,
        "resume_node": resume_node,
        "pending_transactions": {str(key): value for key, value in incomplete.items()},
        "completed_transactions_cleaned": [str(key) for key in completed],
        "pool_version": snapshot.registry.pool_version,
        "can_resume": resume_node not in (None, "completed"),
    }


def write_stdout_stderr(snapshot_dir: str, stdout: str, stderr: str) -> None:
    """Runner helper: persist captured output into the snapshot."""
    root = Path(snapshot_dir)
    if stdout:
        (root / STDOUT_NAME).write_text(stdout, encoding="utf-8")
    if stderr:
        (root / STDERR_NAME).write_text(stderr, encoding="utf-8")
