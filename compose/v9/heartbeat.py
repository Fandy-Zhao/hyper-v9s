"""A liveness file a supervisor can read instead of guessing.

The watchdog has to tell *this stage is running* apart from *this stage died*,
and it cannot do that from the step metrics: calibration, the task-end audit and
the evaluation pass write no per-step metrics at all, so a healthy forty-minute
calibration and a hang look identical from file timestamps alone.  A watchdog
that judged by timestamps would report a stall on every task, and one that
learned to ignore those minutes would stop noticing real stalls.

So each long stage announces itself here before it starts and the watchdog
treats a fresh beat as proof of life for a stage that has no other output.  The
file is written atomically -- a reader must never catch a half-written beat and
read it as a timestamp of zero, which would look like an infinitely old beat.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def heartbeat_path(run_root: str | Path) -> Path:
    return Path(run_root) / "status" / "heartbeat.json"


def beat(run_root: str | Path, stage: str, task: Optional[int] = None, **extra: Any) -> Dict[str, Any]:
    """Record that ``stage`` of ``task`` is running now."""
    path = heartbeat_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "task": None if task is None else int(task),
        "timestamp": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "pid": os.getpid(),
    }
    payload.update(extra)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def read(run_root: str | Path) -> Optional[Dict[str, Any]]:
    path = heartbeat_path(run_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def age(run_root: str | Path, now: Optional[float] = None) -> Optional[float]:
    """Seconds since the last beat, or ``None`` if there has never been one.

    ``None`` and ``0.0`` are deliberately different answers: "no stage has
    announced itself" is not the same as "a stage announced itself just now",
    and a caller that needs liveness has to know which it is looking at.
    """
    payload = read(run_root)
    if payload is None:
        return None
    stamp = payload.get("timestamp")
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
        return None
    return max(0.0, (time.time() if now is None else float(now)) - float(stamp))
