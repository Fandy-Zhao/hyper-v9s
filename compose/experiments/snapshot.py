"""Compose task-boundary snapshots and resume analysis.

A snapshot is a self-contained directory: it can be loaded independently
without any previous stage's temporary directories. It captures:

- stage manifest (schema, git commit, command, environment, GPU info,
  data hash, component hashes)
- task id/name, expert registry (+ pool_version), task state machine
- LoRA checkpoint references + hashes
- functional query encoder (task-0 provenance; later tasks reuse it)
- expert keys / router checkpoint, RMS statistics, runtime kappa
  calibration
- random states (Python/NumPy/PyTorch/CUDA)

Resume analysis maps the persisted state machine onto the Compose
recovery nodes (cluster LoRA training running, key training running,
keys ready not committed, experts committed, RMS ready, snapshot ready
but eval pending) and cleans up half-finished commit transactions
(``CommitTransaction.resume``), which is idempotent: no duplicate
commits, no duplicate pool_version bumps, old expert hashes unchanged.
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
from dataclasses import dataclass, field
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
QUERY_ENCODER_NAME = "query_encoder.pt"
ROUTER_NAME = "router_checkpoint.pt"
RMS_NAME = "rms_statistics.json"
CALIBRATION_NAME = "rms_calibration.json"
RANDOM_STATES_NAME = "random_states.pt"
RESIDUAL_MANIFEST_NAME = "residual_manifest.json"
FORMATION_MANIFEST_NAME = "expert_formation.json"
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

#: Compose recovery nodes.
RESUME_NODES = (
    "cluster_training_running",
    "keys_training_running",
    "keys_ready_not_committed",
    "experts_committed",
    "rms_ready",
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


@dataclass
class ComposeSnapshot:
    """A self-contained task-boundary snapshot."""

    directory: str
    manifest: Dict[str, Any]
    registry: ExpertRegistry
    task_state: TaskStateMachine
    random_states: Dict[str, Any]
    # Optional components (None when not yet produced at save time).
    query_encoder_path: Optional[str] = None
    router_path: Optional[str] = None
    rms_path: Optional[str] = None
    calibration_path: Optional[str] = None
    residual_manifest: Optional[Dict[str, Any]] = None
    formation_manifest: Optional[Dict[str, Any]] = None
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
        query_encoder: Optional[Any] = None,
        router: Optional[Any] = None,
        router_config_hash: Optional[str] = None,
        rms_stats: Optional[Any] = None,
        calibration: Optional[Dict[str, Any]] = None,
        residual_manifest: Optional[Dict[str, Any]] = None,
        formation_manifest: Optional[Dict[str, Any]] = None,
        stdout_text: str = "",
        stderr_text: str = "",
        pool_checkpoint_dir: Optional[str] = None,
    ) -> "ComposeSnapshot":
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
        if query_encoder is not None:
            from compose.router.functional_query import save_query_encoder_checkpoint

            save_query_encoder_checkpoint(str(root / QUERY_ENCODER_NAME), query_encoder)
        if router is not None:
            from compose.router.router import save_compose_router_checkpoint

            save_compose_router_checkpoint(
                str(root / ROUTER_NAME), router,
                pool_version=registry.pool_version,
                config_hash=router_config_hash or "",
                extra={"pool_version": registry.pool_version},
            )
        if rms_stats is not None:
            rms_stats.save_json(str(root / RMS_NAME))
        if calibration is not None:
            _atomic_write_json(root / CALIBRATION_NAME, calibration)
        if residual_manifest is not None:
            _atomic_write_json(root / RESIDUAL_MANIFEST_NAME, residual_manifest)
        if formation_manifest is not None:
            _atomic_write_json(root / FORMATION_MANIFEST_NAME, formation_manifest)
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

        active_ids = [e.expert_id for e in registry.get_active_experts()]
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
            "has_query_encoder": query_encoder is not None,
            "has_router": router is not None,
            "has_rms": rms_stats is not None,
            "has_calibration": calibration is not None,
            "has_residual_manifest": residual_manifest is not None,
            "has_formation_manifest": formation_manifest is not None,
            "active_expert_ids": list(active_ids),
            "pool_checkpoint_dir": (
                str(pool_checkpoint_dir) if pool_checkpoint_dir is not None else None
            ),
        }
        _atomic_write_json(root / MANIFEST_NAME, manifest)
        return cls.load(str(root))

    # ------------------------------------------------------------------
    # Load (independent of any previous stage directories)
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, directory: str) -> "ComposeSnapshot":
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
            query_encoder_path=str(root / QUERY_ENCODER_NAME)
            if (root / QUERY_ENCODER_NAME).is_file() else None,
            router_path=str(root / ROUTER_NAME) if (root / ROUTER_NAME).is_file() else None,
            rms_path=str(root / RMS_NAME) if (root / RMS_NAME).is_file() else None,
            calibration_path=str(root / CALIBRATION_NAME)
            if (root / CALIBRATION_NAME).is_file() else None,
            residual_manifest=(
                json.loads((root / RESIDUAL_MANIFEST_NAME).read_text(encoding="utf-8"))
                if (root / RESIDUAL_MANIFEST_NAME).is_file() else None
            ),
            formation_manifest=(
                json.loads((root / FORMATION_MANIFEST_NAME).read_text(encoding="utf-8"))
                if (root / FORMATION_MANIFEST_NAME).is_file() else None
            ),
            config_path=str(root / CONFIG_COPY_NAME) if (root / CONFIG_COPY_NAME).is_file() else None,
        )

    def load_query_encoder(self):
        """Load the functional query encoder lazily (when present)."""
        if self.query_encoder_path is None:
            return None
        from compose.router.functional_query import (
            ComposeQueryEncoder,
            load_query_encoder_checkpoint,
        )

        info = load_query_encoder_checkpoint(self.query_encoder_path)
        encoder = ComposeQueryEncoder(
            visual_dim=int(info["visual_dim"]),
            text_dim=int(info["text_dim"]),
            query_dim=int(info["query_dim"]),
            seed=int(info["init_seed"]),
            initialize=True,
        )
        load_query_encoder_checkpoint(self.query_encoder_path, encoder)
        return encoder, info

    def load_router(self):
        """Load the router checkpoint lazily (only when present)."""
        if self.router_path is None:
            return None
        from compose.router.router import (
            ComposeRouter,
            load_compose_router_checkpoint,
        )

        router = ComposeRouter()
        extra = load_compose_router_checkpoint(self.router_path, router)
        router.validate_pool_version(self.registry.pool_version)
        return router, extra

    def load_calibration(self) -> Optional[Dict[str, Any]]:
        """Load the persisted runtime kappa calibration (when present)."""
        if self.calibration_path is None:
            return None
        return json.loads(Path(self.calibration_path).read_text(encoding="utf-8"))

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
    """Map snapshot state onto the Compose recovery nodes and clean pending
    commit transactions idempotently.

    Returns ``{"stage", "resume_node", "pending_transactions", "cleanups",
    "can_resume"}``. ``registry_dir`` defaults to the snapshot directory
    (transactions live next to the registry).
    """
    snapshot = ComposeSnapshot.load(snapshot_dir)
    transaction_dir = registry_dir or snapshot_dir
    registry, incomplete, completed = CommitTransaction.resume(transaction_dir)
    stage = snapshot.task_state.stage
    resume_node = None
    if stage in (TaskStage.CLUSTER_EXPERTS_TRAINING,):
        resume_node = "cluster_training_running"
    elif stage in (TaskStage.CLUSTER_EXPERTS_TRAINED, TaskStage.KEYS_TRAINING):
        resume_node = "keys_training_running"
    elif stage is TaskStage.KEYS_READY:
        resume_node = "keys_ready_not_committed"
    elif stage is TaskStage.EXPERTS_COMMITTED:
        resume_node = "experts_committed"
    elif stage in (TaskStage.RMS_READY,):
        resume_node = "rms_ready"
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
