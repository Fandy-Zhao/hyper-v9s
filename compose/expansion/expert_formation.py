"""Cluster -> expert formation.

Each valid cluster becomes exactly one new Compose expert:

- global expert ids come from ``ExpertRegistry.next_expert_id()``
  (monotonic, collision-free across snapshots/resume — never the implicit
  ``task_id * 10 + slot`` namespace);
- the expert key is initialized from the cluster centroid:
  ``e_m = normalize(mean(q_i for i in cluster m))`` (centroid is the
  initialization only; the key stays a learnable parameter);
- LoRA rank is the configuration rank (8);
- the training manifest maps every residual sample to
  ``{teacher_ids, cluster_expert_id, cluster_id}`` so cluster-wise
  conditional-residual LoRA training can run as one ComposeSelection-based
  training job.

Cluster membership is frozen here: LoRA and key training never re-decide
which sample belongs to which cluster.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .query_clustering import ClusterResult

EXPERT_FORMATION_VERSION = 1


@dataclass(frozen=True)
class FormedExpert:
    expert_id: int
    cluster_id: int
    sample_ids: Tuple[str, ...]
    size: int
    centroid: Tuple[float, ...]  # normalized key initialization
    key_mode: str  # "prototype" | "learnable"
    creation_task: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "expert_id": self.expert_id,
            "cluster_id": self.cluster_id,
            "sample_ids": list(self.sample_ids),
            "size": self.size,
            "centroid": list(self.centroid),
            "key_mode": self.key_mode,
            "creation_task": self.creation_task,
        }


def form_cluster_experts(
    result: ClusterResult,
    registry,
    creation_task: int,
    key_mode: str = "learnable",
) -> Tuple[List[FormedExpert], List[str]]:
    """Allocate one global expert id per valid cluster.

    Returns ``(formed, noise_sample_ids)``. Cluster ids are the assignment
    labels; expert ids are allocated monotonically from the registry.
    """
    if key_mode not in ("prototype", "learnable"):
        raise ValueError("key_mode must be one of prototype, learnable")
    formed = []
    for cluster in result.clusters:
        expert_id = registry.next_expert_id()
        formed.append(
            FormedExpert(
                expert_id=int(expert_id),
                cluster_id=int(cluster.cluster_id),
                sample_ids=tuple(cluster.sample_ids),
                size=int(cluster.size),
                centroid=tuple(float(value) for value in cluster.centroid),
                key_mode=str(key_mode),
                creation_task=int(creation_task),
            )
        )
    return formed, list(result.noise_sample_ids)


def build_training_manifest(
    formed: Sequence[FormedExpert],
    residual_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-sample training selections: ``old_teacher_set + new cluster
    expert`` (unions, deduplicated, sorted).

    ``residual_records`` is the residual split (list of dicts with
    ``sample_id`` and ``old_teacher_set``).
    """
    sample_to_expert = {}
    for expert in formed:
        for sample_id in expert.sample_ids:
            sample_to_expert.setdefault(sample_id, []).append(expert.expert_id)
    manifest = []
    seen = set()
    for record in residual_records:
        sample_id = str(record["sample_id"])
        new_experts = sample_to_expert.get(sample_id, [])
        if not new_experts:
            continue
        if sample_id in seen:
            raise ValueError("duplicate sample {} in training manifest".format(sample_id))
        seen.add(sample_id)
        teacher_ids = tuple(sorted(int(value) for value in record.get("old_teacher_set", ())))
        manifest.append(
            {
                "sample_id": sample_id,
                "teacher_ids": list(teacher_ids),
                "expert_ids": sorted(set(teacher_ids) | set(new_experts)),
                "new_expert_ids": sorted(set(new_experts)),
                "cluster_expert_id": int(new_experts[0]),
            }
        )
    return manifest


def write_expert_formation(
    formed: Sequence[FormedExpert],
    manifest: Sequence[Dict[str, Any]],
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the formation manifest; returns its hash."""
    import hashlib
    import os
    import tempfile
    from pathlib import Path

    payload = {
        "schema_version": EXPERT_FORMATION_VERSION,
        "formed_experts": [expert.to_dict() for expert in formed],
        "training_manifest": list(manifest),
    }
    payload.update(extra or {})
    target = Path(path)
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_expert_formation(path: str) -> Dict[str, Any]:
    """Load a formation manifest; validates the schema version."""
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("formation manifest does not exist: {}".format(target))
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != EXPERT_FORMATION_VERSION:
        raise ValueError(
            "unsupported formation manifest schema_version: {}".format(
                payload.get("schema_version")
            )
        )
    return payload
