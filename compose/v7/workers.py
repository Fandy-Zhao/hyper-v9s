"""Orchestrator-side worker pool primitives for adaptive GPU execution.

The V7 orchestrator is a single process (WORLD_SIZE = 1).  Every heavy
stage runs as a fleet of *subprocess workers*, one physical GPU each;
this module gives the orchestrator deterministic, failure-raising
subprocess management:

- ``Deferred``: a lazily resolved job result that starts executing
  immediately and blocks only when the orchestrator first asks for its
  value.  This lets the pruning loop submit an entire remove-and-reroute
  iteration (the ``full`` hypothesis plus every ``minus-candidate``
  hypothesis) before it consumes any score, so candidate evaluations run
  on separate GPUs concurrently while the serial iteration logic keeps
  its exact trajectory.
- ``PooledJobRunner``: fixed-size job pool over a GPU lease list; every
  job owns one GPU for its whole duration (CUDA_VISIBLE_DEVICES set at
  execution), and GPU ownership is recorded in the usage jsonl.
- ``run_worker_batch``: launch N independent worker subprocesses (query
  shards, evaluation chunks), wait for all, fail loudly on the first
  error with its log path.

Workers write only *their own* partial outputs; the orchestrator merges
and atomically renames.  No formal artifact is ever written by two
processes.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .gpu_plan import GpuPlanError


# ---------------------------------------------------------------------------
# Deferred job result
# ---------------------------------------------------------------------------


class Deferred:
    """A value that is being computed on a background worker.

    ``__float__``/``__index__``/``resolve`` block until the worker
    finishes; JSON serialization resolves the value first (the
    orchestrator passes ``default=resolve_json_default``).
    """

    __slots__ = ("_future", "_kind", "_lock")

    def __init__(self, future: Future, kind: str = "value") -> None:
        self._future = future
        self._kind = kind
        self._lock = threading.Lock()

    def _result(self) -> Any:
        with self._lock:
            return self._future.result()

    def _select(self, value: Any) -> Any:
        if isinstance(value, dict):
            return value[self._kind]
        return value

    def resolve(self) -> Any:
        return self._select(self._result())

    def __float__(self) -> float:
        return float(self._select(self._result()))

    def __index__(self) -> int:
        return int(self._select(self._result()))

    def __repr__(self) -> str:
        return "<Deferred {} pending>".format(self._kind)


def resolve_json_default(value: Any) -> Any:
    """json.dumps(default=...) that resolves nested :class:`Deferred`."""
    if isinstance(value, Deferred):
        return value.resolve()
    raise TypeError(
        "object of type {} is not JSON serializable".format(type(value).__name__)
    )


def dump_json_with_deferred(payload: Dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      default=resolve_json_default)
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


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_job_logged(
    command: Sequence[str],
    env: Dict[str, str],
    log_path: str,
) -> None:
    """Run one worker command; stdout/stderr append to ``log_path``."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "worker failed with exit code {}; see {}".format(
                completed.returncode, log_path
            )
        )


def make_worker_env(base_env: Dict[str, str], gpu_id: int) -> Dict[str, str]:
    """One worker sees exactly one physical GPU (as cuda:0)."""
    env = dict(base_env)
    # A worker owns a physical GPU, but must never inherit the orchestrator's
    # multi-GPU teacher plan.  Otherwise a subprocess that re-enters a task
    # runner can recursively launch another fleet of teacher workers.
    env.pop("V8_TEACHER_GPUS", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def run_worker_batch(
    jobs: Sequence[Dict[str, Any]],
) -> None:
    """Run independent workers concurrently; raise on the first failure.

    Each job: ``{"command": [...], "env": {...}, "log": "path"}``.
    ``env`` must already restrict the worker to its own GPU.
    """
    if not jobs:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = []
        for job in jobs:
            futures.append(
                pool.submit(
                    run_job_logged, job["command"], job["env"], job["log"]
                )
            )
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - surface the first
                errors.append(str(error))
        if errors:
            raise RuntimeError("parallel worker batch failed:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# Pooled job runner (candidate evaluations, one GPU per job)
# ---------------------------------------------------------------------------


class PooledJobRunner:
    """Execute ``body(gpu_id, env)`` jobs on a fixed GPU pool.

    Each body runs on exactly one GPU for its whole duration; the pool
    serializes jobs when the candidate count exceeds the GPU count.  GPU
    ownership is recorded in ``usage_log_path`` (jsonl) on acquire and
    release.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        usage_log_path: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.gpu_ids = tuple(int(value) for value in gpu_ids)
        if not self.gpu_ids:
            raise GpuPlanError("job runner requires at least one GPU")
        worker_count = (
            len(self.gpu_ids) if max_workers is None else int(max_workers)
        )
        if worker_count < 1:
            raise GpuPlanError("job runner worker count must be positive")
        self.usage_log_path = usage_log_path
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="v7-job"
        )
        self._lock = threading.Lock()
        self._free: List[int] = list(self.gpu_ids)

    def submit(self, body: Callable[[int, Dict[str, str]], Any]) -> Future:
        return self._pool.submit(self._run_body, body)

    def _run_body(self, body: Callable[[int, Dict[str, str]], Any]) -> Any:
        gpu_id = self._acquire()
        try:
            env = {"V7_WORKER_GPU": str(gpu_id)}
            return body(gpu_id, env)
        finally:
            self._release(gpu_id)

    def _acquire(self) -> int:
        with self._lock:
            if not self._free:
                raise GpuPlanError("no GPU available in the job pool")
            gpu_id = self._free.pop(0)
            self._record(gpu_id, "acquire")
            return gpu_id

    def _release(self, gpu_id: int) -> None:
        with self._lock:
            self._free.append(gpu_id)
            self._free.sort()
            self._record(gpu_id, "release")

    def _record(self, gpu_id: int, action: str) -> None:
        if not self.usage_log_path:
            return
        try:
            with open(self.usage_log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"gpu_id": gpu_id, "action": action}, sort_keys=True
                    )
                    + "\n"
                )
        except OSError:
            pass

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def deferred_from(future: Future, kind: str = "value") -> Deferred:
    return Deferred(future, kind)


def deep_resolve(value: Any) -> Any:
    """Materialize every nested :class:`Deferred` inside a result tree.

    Used once, after the pruning trajectory finished, before the result is
    handed to plain-JSON writers (e.g. ``commit_retained_candidates``); for
    legacy synchronous scorers it is a structural no-op.
    """
    if isinstance(value, Deferred):
        return deep_resolve(value.resolve())
    if isinstance(value, dict):
        return {key: deep_resolve(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(deep_resolve(item) for item in value)
    if isinstance(value, list):
        return [deep_resolve(item) for item in value]
    return value
