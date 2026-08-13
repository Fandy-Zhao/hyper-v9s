"""Residual query clustering (spherical K-Means + cosine silhouette).

Inputs are the 128-D normalized functional queries of residual samples.
Distance is ``1 - cosine(q_i, q_j)``.

Procedure:

1. For K = 1 .. K_max run spherical K-Means with multiple deterministic
   K-means++ initializations and retain the lowest cosine-inertia solution.
2. For K >= 2 compute the standard cosine silhouette (mean intra-cluster
   distance versus the minimum mean distance to another cluster); for K = 1 silhouette is
   treated as optimal (a single cluster always fits).
3. ``K_star = argmax_K silhouette(K)``; if ``best_silhouette`` falls below
   ``silhouette_threshold`` the assignment degrades to K = 1.
4. Clusters with ``size < min_cluster_samples`` are marked noise and never
   trained as experts.
5. The cluster manifest (assignments, centroids, silhouette trace, seed)
   is persisted; assignment is frozen from that point on — LoRA training
   and key training never re-decide cluster membership.

Deterministic subsampling bounds the silhouette cost on large residual
sets (recorded in the manifest with its seed and sample count).
"""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

CLUSTER_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ComposeClusteringConfig:
    algorithm: str = "spherical_kmeans"
    max_clusters: int = 4
    silhouette_threshold: float = 0.15
    min_cluster_samples: int = 8
    random_seed: int = 42
    max_iterations: int = 100
    silhouette_sample_size: int = 2000
    n_init: int = 20

    def __post_init__(self) -> None:
        if self.algorithm != "spherical_kmeans":
            raise ValueError(
                "unknown clustering algorithm: {!r}".format(self.algorithm)
            )
        if self.max_clusters < 1:
            raise ValueError("max_clusters must be at least 1")
        if not 0.0 <= self.silhouette_threshold <= 1.0:
            raise ValueError("silhouette_threshold must lie in [0, 1]")
        if self.min_cluster_samples < 1:
            raise ValueError("min_cluster_samples must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.silhouette_sample_size < 0:
            raise ValueError("silhouette_sample_size must be non-negative")
        if self.n_init < 1:
            raise ValueError("n_init must be positive")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClusterAssignment:
    """One valid cluster (not noise)."""

    cluster_id: int
    sample_ids: List[str]
    size: int
    centroid: List[float]  # normalized 128-D centroid

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "sample_ids": list(self.sample_ids),
            "size": self.size,
            "centroid": list(self.centroid),
        }


@dataclass
class ClusterResult:
    task_id: int
    selected_k: int
    selected_silhouette: Optional[float]
    silhouette_by_k: Dict[int, Optional[float]]
    clusters: List[ClusterAssignment]
    noise_sample_ids: List[str]
    seed: int
    query_hash: str
    n_init: int = 20
    effective_cluster_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "selected_k": self.selected_k,
            "selected_silhouette": self.selected_silhouette,
            "silhouette_by_k": {
                str(k): value for k, value in self.silhouette_by_k.items()
            },
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "noise_sample_ids": list(self.noise_sample_ids),
            "seed": self.seed,
            "query_hash": self.query_hash,
            "n_init": self.n_init,
            "effective_cluster_count": self.effective_cluster_count,
        }


def _kmeans_plus_plus_centers(
    queries: Tensor, k: int, generator: torch.Generator
) -> Tensor:
    """K-means++ initialization over L2-normalized rows."""
    n = queries.shape[0]
    first = int(torch.randint(n, (1,), generator=generator).item())
    centers = [queries[first].clone()]
    closest = (1.0 - queries @ queries[first]).square()
    for _ in range(k - 1):
        weights = closest.clamp_min(1e-12)
        if weights.sum().item() <= 0:
            weights = torch.ones(n, dtype=queries.dtype)
        chosen = int(
            torch.multinomial(weights, 1, generator=generator).item()
        )
        center = queries[chosen]
        centers.append(center.clone())
        closest = torch.minimum(
            closest, (1.0 - queries @ center).square()
        )
    return torch.stack([F.normalize(c, dim=0) for c in centers])


def spherical_kmeans(
    queries: Tensor,
    k: int,
    max_iterations: int = 100,
    seed: int = 42,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Spherical K-Means; returns ``(assignments, centers)``.

    ``queries`` must be [N, D] normalized rows. ``assignments`` is a
    LongTensor of length N in [0, k).
    """
    if queries.ndim != 2 or queries.shape[0] == 0:
        raise ValueError("queries must be a non-empty [samples, dim] tensor")
    if k < 1 or k > queries.shape[0]:
        raise ValueError("k must lie in [1, sample count]")
    if queries.shape[1] != 128:
        raise ValueError("functional queries must be 128-D")
    if k == 1:
        center = F.normalize(queries.float().mean(dim=0), dim=0)
        return torch.zeros(queries.shape[0], dtype=torch.long), center.unsqueeze(0)
    generator = torch.Generator().manual_seed(int(seed))
    centers = _kmeans_plus_plus_centers(queries, k, generator)
    assignments = torch.zeros(queries.shape[0], dtype=torch.long)
    for _ in range(max_iterations):
        similarities = queries @ centers.T  # [N, k]
        new_assignments = similarities.argmax(dim=1)
        new_centers = []
        for cluster in range(k):
            members = queries[new_assignments == cluster]
            if members.shape[0] == 0:
                # Re-seed an empty center from the least-represented query.
                counts = torch.bincount(new_assignments, minlength=k).float()
                least = int(counts.argmin().item())
                fallback_index = int(torch.randint(queries.shape[0], (1,), generator=generator).item())
                fallback = queries[fallback_index]
                new_centers.append(F.normalize(fallback, dim=0))
                new_assignments = new_assignments.clone()
                new_assignments[fallback_index] = least
                continue
            new_centers.append(F.normalize(members.float().mean(dim=0), dim=0))
        centers = torch.stack(new_centers)
        converged = bool(torch.all(new_assignments == assignments))
        assignments = new_assignments
        if converged:
            break
    return assignments, centers


def cosine_silhouette(
    queries: Tensor,
    assignments: Tensor,
    k: int,
    seed: int = 42,
    sample_size: int = 0,
) -> Optional[float]:
    """Cosine silhouette over a deterministic subsample.

    ``sample_size <= 0`` means "use every query"; otherwise a seeded
    subsample of at most ``sample_size`` rows is drawn and the silhouette
    is computed on it (bounded memory for large residual sets).
    """
    if k == 1:
        return None
    if queries.ndim != 2 or queries.shape[0] != assignments.shape[0]:
        raise ValueError("queries and assignments must align")
    if sample_size > 0 and queries.shape[0] > sample_size:
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(queries.shape[0], generator=generator)[
            :sample_size
        ]
        queries = queries[indices]
        assignments = assignments[indices]
    if queries.shape[0] < 2:
        return None
    distances = (1.0 - queries @ queries.T).clamp_(0.0, 2.0)
    values = []
    for index in range(queries.shape[0]):
        own = int(assignments[index])
        own_mask = assignments == own
        own_mask[index] = False
        if int(own_mask.sum()) == 0:
            values.append(0.0)
            continue
        a = distances[index, own_mask].mean()
        other_means = []
        for cluster in range(k):
            if cluster == own:
                continue
            mask = assignments == cluster
            if int(mask.sum()) > 0:
                other_means.append(distances[index, mask].mean())
        if not other_means:
            values.append(0.0)
            continue
        b = torch.stack(other_means).min()
        denominator = torch.maximum(a, b)
        values.append(float((b - a) / denominator) if float(denominator) >= 1e-12 else 0.0)
    if not values:
        return None
    return float(sum(values) / len(values))


def _cosine_inertia(queries: Tensor, assignments: Tensor, centers: Tensor) -> float:
    selected = centers[assignments]
    return float((1.0 - (queries * selected).sum(dim=1)).clamp_min(0.0).sum())


def spherical_kmeans_multi_init(
    queries: Tensor, k: int, max_iterations: int, seed: int, n_init: int
) -> Tuple[Tensor, Tensor]:
    best = None
    for init_index in range(int(n_init)):
        init_seed = int(seed) + init_index * 104729
        assignments, centers = spherical_kmeans(
            queries, k, max_iterations=max_iterations, seed=init_seed
        )
        inertia = _cosine_inertia(queries, assignments, centers)
        signature = tuple(int(value) for value in assignments.tolist())
        candidate = (inertia, signature, assignments, centers)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3]


def cluster_residual_queries(
    queries: Tensor,
    sample_ids: Sequence[str],
    config: ComposeClusteringConfig,
    task_id: int = 0,
    query_hash: str = "",
) -> ClusterResult:
    """Run the dynamic-K spherical clustering and build the manifest.

    ``queries`` rows are 128-D normalized functional queries; they must
    align 1:1 with ``sample_ids``.
    """
    queries = queries.detach().float()
    if queries.ndim != 2 or queries.shape[0] != len(sample_ids):
        raise ValueError("queries and sample_ids must align")
    if queries.shape[0] == 0:
        raise ValueError("no residual queries to cluster")
    if queries.shape[1] != 128:
        raise ValueError("functional queries must be 128-D")

    k_max = min(config.max_clusters, queries.shape[0])
    assignments_by_k = {}
    silhouette_by_k = {}
    for k in range(1, k_max + 1):
        assignments, _ = spherical_kmeans_multi_init(
            queries,
            k,
            max_iterations=config.max_iterations,
            seed=config.random_seed,
            n_init=config.n_init,
        )
        assignments_by_k[k] = assignments
        silhouette_by_k[k] = cosine_silhouette(
            queries,
            assignments,
            k,
            seed=config.random_seed,
            sample_size=config.silhouette_sample_size,
        )

    # K=1 is always admissible (silhouette None -> treated as optimal).
    candidates = [k for k in range(1, k_max + 1)]
    if k_max > 1:
        def _score(k: int) -> float:
            value = silhouette_by_k[k]
            return value if value is not None else -1.0
        best = max(candidates[1:], key=_score)
        if _score(best) < config.silhouette_threshold:
            selected_k = 1
            selected_silhouette = None
        else:
            selected_k = best
            selected_silhouette = silhouette_by_k[best]
    else:
        selected_k = 1
        selected_silhouette = None

    assignments = assignments_by_k[selected_k]
    clusters = []
    noise_ids = []
    for cluster in range(selected_k):
        member_indices = (assignments == cluster).nonzero(as_tuple=False).flatten()
        if member_indices.numel() < config.min_cluster_samples:
            noise_ids.extend(
                str(sample_ids[int(index)]) for index in member_indices.tolist()
            )
            continue
        member_queries = queries[member_indices]
        centroid = F.normalize(member_queries.mean(dim=0), dim=0)
        clusters.append(
            ClusterAssignment(
                cluster_id=int(cluster),
                sample_ids=[
                    str(sample_ids[int(index)]) for index in member_indices.tolist()
                ],
                size=int(member_indices.numel()),
                centroid=centroid.detach().cpu().tolist(),
            )
        )
    return ClusterResult(
        task_id=int(task_id),
        selected_k=int(selected_k),
        selected_silhouette=selected_silhouette,
        silhouette_by_k={
            int(k): value for k, value in silhouette_by_k.items()
        },
        clusters=clusters,
        noise_sample_ids=sorted(set(noise_ids)),
        seed=int(config.random_seed),
        query_hash=str(query_hash),
        n_init=int(config.n_init),
        effective_cluster_count=len(clusters),
    )


def build_single_bootstrap_cluster(
    queries: Tensor,
    sample_ids: Sequence[str],
    task_id: int = 0,
    query_hash: str = "",
    seed: int = 42,
    n_init: int = 20,
) -> ClusterResult:
    """Create the Task0 bootstrap cluster without residual model selection."""
    queries = queries.detach().float()
    if task_id != 0:
        raise ValueError("single bootstrap is restricted to task 0")
    if queries.ndim != 2 or queries.shape != (len(sample_ids), 128):
        raise ValueError("queries and sample_ids must align as [N, 128]")
    if not sample_ids:
        raise ValueError("bootstrap requires at least one sample")
    centroid = F.normalize(queries.mean(dim=0), dim=0)
    cluster = ClusterAssignment(
        cluster_id=0,
        sample_ids=[str(value) for value in sample_ids],
        size=len(sample_ids),
        centroid=centroid.detach().cpu().tolist(),
    )
    return ClusterResult(
        task_id=0,
        selected_k=1,
        selected_silhouette=None,
        silhouette_by_k={1: None},
        clusters=[cluster],
        noise_sample_ids=[],
        seed=int(seed),
        query_hash=str(query_hash),
        n_init=int(n_init),
        effective_cluster_count=1,
    )


def cluster_assignment_map(result: ClusterResult) -> Dict[str, int]:
    """sample_id -> cluster_id for every non-noise cluster member."""
    mapping = {}
    for cluster in result.clusters:
        for sample_id in cluster.sample_ids:
            mapping[sample_id] = cluster.cluster_id
    return mapping


def write_cluster_manifest(
    result: ClusterResult, path: str, extra: Optional[Dict[str, Any]] = None
) -> str:
    """Persist the cluster manifest; returns the manifest hash."""
    payload = {
        "schema_version": CLUSTER_MANIFEST_VERSION,
        "result": result.to_dict(),
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
    return stable_hash_manifest(payload)


def stable_hash_manifest(payload: Dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_cluster_manifest(path: str) -> Dict[str, Any]:
    """Load a cluster manifest; validates the schema version."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("cluster manifest does not exist: {}".format(target))
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != CLUSTER_MANIFEST_VERSION:
        raise ValueError(
            "unsupported cluster manifest schema_version: {}".format(
                payload.get("schema_version")
            )
        )
    return payload
