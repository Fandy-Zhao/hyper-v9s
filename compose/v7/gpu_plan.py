"""GPU-count-adaptive execution planning for V7 (spec 0903).

The planner is a pure, deterministic function of the *available* GPU ids:
it decides stage GPU assignment (query / training / RMS / pruning /
evaluation) and the S3 recipe (world size, gradient accumulation) so that
the declared formal recipe is preserved exactly:

    global_batch = per_device_batch * training_world_size * grad_accum
    strict:      effective_global_batch == target_global_batch (recipe exact)
    throughput:  only when the user explicitly opts in; the closest integer
                 global batch is used and ``recipe_exact`` is False.

GPU count only changes *scheduling*.  It never changes the V7 method or
the formal recipe: per_device_batch stays 1 (v1 adaptive), learning rate,
epochs, scheduler, warmup, seed, routing, RMS, pruning and evaluation
protocols are untouched.  Remaining GPUs (available but not used by S3 in
strict mode) are simply left idle for future-task precompute; they are
never shared with the current S3 DDP job.

Resolution order for the available GPU set:

    CLI --gpus > env V7_GPUS > env CUDA_VISIBLE_DEVICES > idle-GPU probe

The probe only selects devices that are *already idle* (used memory below
``idle_memory_threshold_mib`` and zero utilization); it never steals a GPU
that another user is using, never kills processes and never resets devices.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

GPU_PLAN_SCHEMA_VERSION = 1

DEFAULT_TARGET_GLOBAL_BATCH = 64
DEFAULT_PER_DEVICE_BATCH = 1


class GpuPlanError(ValueError):
    """A fail-closed planner error (invalid ids, unresolvable recipe...)."""


# ---------------------------------------------------------------------------
# Ids and GPU probing
# ---------------------------------------------------------------------------


def parse_gpu_ids(value: Optional[str]) -> List[int]:
    """Parse a comma-separated GPU id string into distinct, ordered ints."""
    if value is None:
        return []
    raw = [item.strip() for item in str(value).split(",") if item.strip()]
    if not raw:
        raise GpuPlanError("empty GPU id list")
    ids: List[int] = []
    for item in raw:
        try:
            gpu_id = int(item)
        except ValueError:
            raise GpuPlanError("invalid GPU id {!r}".format(item)) from None
        if gpu_id < 0:
            raise GpuPlanError("GPU ids must be non-negative, got {}".format(gpu_id))
        if gpu_id not in ids:
            ids.append(gpu_id)
    return ids


def probe_gpu_state() -> Dict[int, Tuple[int, int]]:
    """``{physical_id: (memory_used_mib, utilization_percent)}``.

    Raises ``GpuPlanError`` when nvidia-smi is unavailable or unparsable so
    automatic detection fails closed instead of silently misplanning.
    """
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GpuPlanError(
            "cannot probe GPU state (nvidia-smi failed); pass an explicit "
            "--gpus / V7_GPUS list instead"
        ) from error
    state: Dict[int, Tuple[int, int]] = {}
    for row in rows.splitlines():
        columns = [item.strip() for item in row.split(",")]
        if len(columns) != 3:
            raise GpuPlanError("unparsable nvidia-smi row: {!r}".format(row))
        try:
            index, memory, utilization = (int(value) for value in columns)
        except ValueError:
            raise GpuPlanError("unparsable nvidia-smi row: {!r}".format(row)) from None
        state[index] = (memory, utilization)
    if not state:
        raise GpuPlanError("nvidia-smi reported no GPUs")
    return state


def detect_idle_gpu_ids(
    idle_memory_threshold_mib: int = 1024,
) -> List[int]:
    """Deterministically select idle physical GPUs (ascending id order).

    A GPU is idle only when another user is clearly not using it: used
    memory below ``idle_memory_threshold_mib`` AND zero utilization.  This
    probe never touches busy devices.
    """
    state = probe_gpu_state()
    idle = [
        index for index, (memory, utilization) in sorted(state.items())
        if memory < idle_memory_threshold_mib and utilization == 0
    ]
    if not idle:
        raise GpuPlanError(
            "no idle GPU found (all devices busy); pass an explicit "
            "--gpus / V7_GPUS list"
        )
    return idle


def resolve_available_gpu_ids(
    cli_ids: Optional[str] = None,
    env_var: str = "V7_GPUS",
    allow_auto_detect: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> List[int]:
    """Resolution: CLI --gpus > $V7_GPUS > $CUDA_VISIBLE_DEVICES > probe."""
    environment = dict(os.environ if env is None else env)
    cli_value = cli_ids if cli_ids is not None else None
    env_value = environment.get(env_var)
    cuda_visible = environment.get("CUDA_VISIBLE_DEVICES")
    for label, value in (
        ("--gpus", cli_value),
        ("${}".format(env_var), env_value),
        ("CUDA_VISIBLE_DEVICES", cuda_visible),
    ):
        if value is None:
            continue
        parsed = parse_gpu_ids(value)
        return parsed
    if not allow_auto_detect:
        raise GpuPlanError(
            "no explicit GPU ids; automatic detection is disabled for this entry"
        )
    return detect_idle_gpu_ids()


# ---------------------------------------------------------------------------
# Recipe planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V7RecipePlan:
    """The S3 recipe implied by ``available_gpu_ids`` and ``recipe_mode``."""

    recipe_mode: str  # "strict" | "throughput"
    per_device_batch: int
    target_global_batch: int
    training_world_size: int
    gradient_accumulation_steps: int
    effective_global_batch: int

    @property
    def recipe_exact(self) -> bool:
        if self.recipe_mode != "strict":
            return False
        return self.effective_global_batch == self.target_global_batch

    @property
    def global_batch_delta(self) -> int:
        return self.effective_global_batch - self.target_global_batch

    @property
    def relative_difference(self) -> float:
        return self.global_batch_delta / float(self.target_global_batch)

    @property
    def description(self) -> str:
        return (
            "{}-rank DDP, per-device batch {}, gradient accumulation {}, "
            "global batch {}, recipe exact: {}".format(
                self.training_world_size, self.per_device_batch,
                self.gradient_accumulation_steps, self.effective_global_batch,
                "YES" if self.recipe_exact else "NO",
            )
        )


def plan_training_recipe(
    available_gpu_count: int,
    target_global_batch: int = DEFAULT_TARGET_GLOBAL_BATCH,
    per_device_batch: int = DEFAULT_PER_DEVICE_BATCH,
    recipe_mode: str = "strict",
) -> V7RecipePlan:
    """Choose ``(world_size, gradient_accumulation)`` for the GPU count.

    strict (default, formal): the largest world size that (a) does not
    exceed ``available_gpu_count`` and (b) keeps the global batch exactly
    at the target.  With per_device_batch=1 and target 64 this selects
    powers of two: 1->1/64, 2->2/32, 3->2/32, 4->4/16, 5..7->4/16,
    8->8/8.  Training never silently changes the recipe to 63/60/70.

    throughput (opt-in only): uses every GPU and rounds the accumulation
    to the nearest integer global batch; ``recipe_exact`` is False and the
    relative difference is recorded for the report.
    """
    if recipe_mode not in ("strict", "throughput"):
        raise GpuPlanError("recipe_mode must be 'strict' or 'throughput'")
    if available_gpu_count < 1:
        raise GpuPlanError("at least one GPU is required")
    if target_global_batch < 1 or per_device_batch < 1:
        raise GpuPlanError("global batch and per-device batch must be positive")
    if target_global_batch % per_device_batch != 0:
        raise GpuPlanError(
            "per_device_batch {} does not divide target_global_batch {}".format(
                per_device_batch, target_global_batch
            )
        )
    if recipe_mode == "strict":
        divisors = [
            world_size
            for world_size in range(1, available_gpu_count + 1)
            if target_global_batch % (per_device_batch * world_size) == 0
        ]
        if not divisors:
            raise GpuPlanError(
                "no world size in [1, {}] keeps global batch {} exact; "
                "use recipe_mode=throughput for an approximate recipe".format(
                    available_gpu_count, target_global_batch
                )
            )
        world_size = max(divisors)
        accumulation = target_global_batch // (per_device_batch * world_size)
        effective = target_global_batch
    else:
        world_size = available_gpu_count
        per_window = per_device_batch * world_size
        accumulation = max(1, int(round(target_global_batch / per_window)))
        effective = per_window * accumulation
    return V7RecipePlan(
        recipe_mode=recipe_mode,
        per_device_batch=per_device_batch,
        target_global_batch=target_global_batch,
        training_world_size=world_size,
        gradient_accumulation_steps=accumulation,
        effective_global_batch=effective,
    )


# ---------------------------------------------------------------------------
# Stage allocation and the full plan
# ---------------------------------------------------------------------------


def allocate_stage_gpus(
    available_gpu_ids: Sequence[int], training_world_size: int
) -> Dict[str, List[int]]:
    """Default stage allocation (spec 0903 §15).

    Query / RMS / pruning / evaluation use all available GPUs.  Training
    (strict recipe) uses the largest recipe-exact subset; the remaining
    GPUs stay idle during S3 (or are used by future-task precompute) and
    are never shared with the DDP job.
    """
    available = sorted(int(value) for value in available_gpu_ids)
    if not available:
        raise GpuPlanError("no available GPU ids to allocate")
    if training_world_size > len(available):
        raise GpuPlanError(
            "training world size {} exceeds available GPUs {}".format(
                training_world_size, len(available)
            )
        )
    training = available[:training_world_size]
    return {
        "query": list(available),
        "training": training,
        "rms": list(available),
        "pruning": list(available),
        "evaluation": list(available),
    }


@dataclass(frozen=True)
class V7GPUPlan:
    """Complete deterministic GPU execution plan for one V7 task stage set."""

    available_gpu_ids: Tuple[int, ...]
    recipe: V7RecipePlan
    query_gpu_ids: Tuple[int, ...] = field(default_factory=tuple)
    training_gpu_ids: Tuple[int, ...] = field(default_factory=tuple)
    rms_gpu_ids: Tuple[int, ...] = field(default_factory=tuple)
    pruning_gpu_ids: Tuple[int, ...] = field(default_factory=tuple)
    evaluation_gpu_ids: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def build(
        cls,
        available_gpu_ids: Sequence[int],
        target_global_batch: int = DEFAULT_TARGET_GLOBAL_BATCH,
        per_device_batch: int = DEFAULT_PER_DEVICE_BATCH,
        recipe_mode: str = "strict",
    ) -> "V7GPUPlan":
        available = sorted(int(value) for value in available_gpu_ids)
        recipe = plan_training_recipe(
            len(available),
            target_global_batch=target_global_batch,
            per_device_batch=per_device_batch,
            recipe_mode=recipe_mode,
        )
        stages = allocate_stage_gpus(available, recipe.training_world_size)
        return cls(
            available_gpu_ids=tuple(available),
            recipe=recipe,
            query_gpu_ids=tuple(stages["query"]),
            training_gpu_ids=tuple(stages["training"]),
            rms_gpu_ids=tuple(stages["rms"]),
            pruning_gpu_ids=tuple(stages["pruning"]),
            evaluation_gpu_ids=tuple(stages["evaluation"]),
        )

    @property
    def training_world_size(self) -> int:
        return self.recipe.training_world_size

    @property
    def query_world_size(self) -> int:
        return len(self.query_gpu_ids)

    @property
    def rms_world_size(self) -> int:
        return len(self.rms_gpu_ids)

    @property
    def pruning_world_size(self) -> int:
        return len(self.pruning_gpu_ids)

    @property
    def evaluation_world_size(self) -> int:
        return len(self.evaluation_gpu_ids)

    @property
    def recipe_exact(self) -> bool:
        return self.recipe.recipe_exact

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": GPU_PLAN_SCHEMA_VERSION,
            "available_gpu_ids": list(self.available_gpu_ids),
            "query_gpu_ids": list(self.query_gpu_ids),
            "query_world_size": self.query_world_size,
            "training_gpu_ids": list(self.training_gpu_ids),
            "training_world_size": self.recipe.training_world_size,
            "rms_gpu_ids": list(self.rms_gpu_ids),
            "rms_world_size": self.rms_world_size,
            "pruning_gpu_ids": list(self.pruning_gpu_ids),
            "pruning_world_size": self.pruning_world_size,
            "evaluation_gpu_ids": list(self.evaluation_gpu_ids),
            "evaluation_world_size": self.evaluation_world_size,
            "recipe_mode": self.recipe.recipe_mode,
            "training_per_device_batch": self.recipe.per_device_batch,
            "training_gradient_accumulation_steps": (
                self.recipe.gradient_accumulation_steps
            ),
            "effective_global_batch": self.recipe.effective_global_batch,
            "target_global_batch": self.recipe.target_global_batch,
            "global_batch_delta": self.recipe.global_batch_delta,
            "recipe_exact": self.recipe_exact,
        }

    def render_plan_block(self) -> str:
        """Human-readable GPU plan banner (printed at task start)."""
        recipe = self.recipe
        return "\n".join(
            [
                "===============================",
                "V7 Adaptive GPU Execution Plan",
                "===============================",
                "Available GPUs:",
                ",".join(str(value) for value in self.available_gpu_ids),
                "",
                "Query workers:",
                str(self.query_world_size),
                "",
                "Training:",
                "{}-rank DDP".format(recipe.training_world_size),
                "per-device batch: {}".format(recipe.per_device_batch),
                "gradient accumulation: {}".format(
                    recipe.gradient_accumulation_steps
                ),
                "global batch: {}".format(recipe.effective_global_batch),
                "recipe exact: {}".format(
                    "YES" if recipe.recipe_exact else "NO"
                ),
                "",
                "RMS workers:",
                str(self.rms_world_size),
                "",
                "Pruning workers:",
                str(self.pruning_world_size),
                "",
                "Evaluation workers:",
                str(self.evaluation_world_size),
                "===============================",
            ]
        )


def json_dumps_gpu_plan(plan: V7GPUPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# GPU ownership lease (single-node worker assignment bookkeeping)
# ---------------------------------------------------------------------------


class GPULeasePool:
    """Deterministic round-robin GPU assignment plus usage bookkeeping.

    The lease only guarantees single-owner assignment inside this
    orchestrator (GPU id -> current job).  It never kills processes, never
    resets devices and never acquires a GPU outside ``gpu_ids``.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        usage_log_path: Optional[str] = None,
    ) -> None:
        self.gpu_ids = tuple(int(value) for value in gpu_ids)
        if not self.gpu_ids:
            raise GpuPlanError("lease pool requires at least one GPU")
        self._available: List[int] = list(self.gpu_ids)
        self._lock = threading.Lock()
        self.usage_log_path = usage_log_path

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise GpuPlanError("no GPU available in the lease pool")
            # Rotate so a long queue does not pin the same device.
            gpu_id = self._available.pop(0)
            self._available.append(gpu_id)
            self._record(gpu_id, "acquire")
            return gpu_id

    def release(self, gpu_id: int) -> None:
        with self._lock:
            self._record(gpu_id, "release")

    def _record(self, gpu_id: int, action: str) -> None:
        if not self.usage_log_path:
            return
        try:
            # The lock above serializes appends; O_APPEND keeps each record
            # atomic in practice. Bookkeeping must never break execution.
            with open(self.usage_log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"gpu_id": gpu_id, "action": action},
                        sort_keys=True,
                    )
                    + "\n"
                )
        except OSError:
            pass
