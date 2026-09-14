"""V7 fixed-query binary cache: contract, sharding, merge and audit helpers.

The mathematical definition of the Fixed Query is owned by
``compose.v7.query.FixedMultimodalQuery``; this module never redefines it.
It implements the persistent-cache layer around it:

- deterministic ``index % world_size`` shard planning for a declared split,
- a self-contained ``queries.pt`` per split (``sample_ids`` + ``queries``),
- a ``metadata.json`` provenance/audit sidecar (full contract in §0903 spec),
- tmp -> fsync -> atomic-rename writes,
- contract-bound resume and stale-cache invalidation,
- merge audits (dimension, dtype, duplicate/missing/unknown ids, finite,
  norm, count equality) and value hashes.

Row order inside every artifact is the *declared split file order*; the
sample id is always the primary key.  All public functions are pure CPU
code so the audit logic is unit-testable without GPUs.
"""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .pool import tensor_checksum  # sha256 of the raw float32 bytes

QUERY_SCHEMA_VERSION = 1
QUERY_MODE = "v7_fixed"
QUERY_DIM = 1536
QUERY_DTYPE = "float32"
SHARD_ALGORITHM = "index%world"
CACHE_KIND = "v7_fixed_query_split_cache"
TASK_CENTER_KIND = "v7_full_train_task_center"

_MISSING = object()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_payload_hash(payload: Mapping[str, object]) -> str:
    """Deterministic canonical-JSON sha256 (matches repo conventions)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sample_id_of(record: Mapping[str, object]) -> str:
    """Canonical sample id: ``id`` when present, else ``question_id``.

    Train splits carry a collision-safe ``id`` (v7_t{task}_train_{i}); the
    validation and test splits carry ``question_id``.  This mirrors
    ``compose.eval.query_features.shard_expected_ids``.
    """
    value = record.get("id", record.get("question_id"))
    if value is None:
        raise ValueError("declared record has neither 'id' nor 'question_id'")
    return str(value)


def sample_id_set_hash(sample_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(value) for value in sample_ids), separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def ordered_id_hash(sample_ids: Sequence[str]) -> str:
    """Hash over ids in declared order (row-order contract of the cache)."""
    return stable_payload_hash({"ordered_ids": [str(value) for value in sample_ids]})


def split_content_hash(records: Sequence[Mapping[str, object]]) -> str:
    """Semantic identity of a declared split: image + question text only.

    The query depends on exactly these two fields, so this hash invalidates
    a cache when content changes even if the JSON bytes also differ.
    """
    from compose.data.records import question_text

    rows = []
    for record in records:
        rows.append(
            {
                "image": str(record.get("image", "")),
                "question": question_text(record).strip(),
            }
        )
    return stable_payload_hash({"rows": rows})


def rank_indexes(total: int, world_size: int, rank: int) -> List[int]:
    """Deterministic parity plan: index % world_size == rank.

    No padding, no drop_last:  |D0| + |D1| == |D| and every worker's slice
    is exactly the declared-file order restricted to its residue class.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError("rank {} out of world_size {}".format(rank, world_size))
    return [index for index in range(int(total)) if index % world_size == rank]


def rank_sample_slice(sample_ids: Sequence[str], world_size: int, rank: int) -> List[str]:
    return [str(sample_ids[index]) for index in rank_indexes(len(sample_ids), world_size, rank)]


def _canonical_json_lines(sample_ids: Sequence[str], queries: Tensor) -> str:
    """Deterministic value fingerprint independent of file serialization."""
    return json.dumps(
        {
            "ids": [str(value) for value in sample_ids],
            "checksum": tensor_checksum(queries),
            "dtype": QUERY_DTYPE,
            "query_dim": QUERY_DIM,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def query_tensor_hash(sample_ids: Sequence[str], queries: Tensor) -> str:
    return hashlib.sha256(_canonical_json_lines(sample_ids, queries).encode()).hexdigest()


@dataclass(frozen=True)
class SplitContract:
    """Everything that binds a cache (or shard) to the inputs that made it."""

    schema_version: int
    git_sha: str
    git_branch: str
    task_index: int
    task_name: str
    split: str
    source_dataset_path: str
    source_dataset_sha256: str
    source_content_hash: str
    image_folder: str
    backbone_name: str
    backbone_path: str
    backbone_hash: str
    query_mode: str
    query_impl_hash: str
    query_dim: int
    query_dtype: str
    world_size: int
    shard_algorithm: str
    batch_size: int
    runtime_clip_dtype: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
            "task_index": self.task_index,
            "task_name": self.task_name,
            "split": self.split,
            "source_dataset_path": str(Path(self.source_dataset_path).expanduser().resolve()),
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_content_hash": self.source_content_hash,
            "image_folder": str(Path(self.image_folder).expanduser().resolve()),
            "query_backbone_name": self.backbone_name,
            "query_backbone_path": str(Path(self.backbone_path).expanduser().resolve()),
            "query_backbone_hash": self.backbone_hash,
            "query_mode": self.query_mode,
            "query_impl_hash": self.query_impl_hash,
            "query_dim": self.query_dim,
            "query_dtype": self.query_dtype,
            "world_size": self.world_size,
            "shard_algorithm": self.shard_algorithm,
            "batch_size": self.batch_size,
            "runtime_clip_dtype": self.runtime_clip_dtype,
        }

    def contract_hash(self) -> str:
        return stable_payload_hash(self.to_dict())


def build_split_contract(
    *,
    git_sha: str,
    git_branch: str,
    task_index: int,
    task_name: str,
    split: str,
    source_dataset_path: str,
    records: Sequence[Mapping[str, object]],
    image_folder: str,
    backbone_name: str,
    backbone_path: str,
    backbone_hash: str,
    query_impl_hash: str,
    world_size: int,
    batch_size: int,
    runtime_clip_dtype: str = "float16",
) -> SplitContract:
    return SplitContract(
        schema_version=QUERY_SCHEMA_VERSION,
        git_sha=git_sha,
        git_branch=git_branch,
        task_index=int(task_index),
        task_name=str(task_name),
        split=str(split),
        source_dataset_path=source_dataset_path,
        source_dataset_sha256=sha256_file(source_dataset_path),
        source_content_hash=split_content_hash(records),
        image_folder=image_folder,
        backbone_name=backbone_name,
        backbone_path=backbone_path,
        backbone_hash=backbone_hash,
        query_mode=QUERY_MODE,
        query_impl_hash=query_impl_hash,
        query_dim=QUERY_DIM,
        query_dtype=QUERY_DTYPE,
        world_size=int(world_size),
        shard_algorithm=SHARD_ALGORITHM,
        batch_size=int(batch_size),
        runtime_clip_dtype=runtime_clip_dtype,
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            pass
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_partial(
    path: str,
    *,
    rank: int,
    world_size: int,
    contract_hash: str,
    task_index: int,
    task_name: str,
    split: str,
    sample_ids: Sequence[str],
    queries: Tensor,
    stats: Mapping[str, object],
) -> None:
    """Worker-owned shard file (never the official cache)."""
    queries = _as_clean_queries(sample_ids, queries)
    payload = {
        "schema_version": QUERY_SCHEMA_VERSION,
        "kind": "v7_query_shard_partial",
        "task_index": int(task_index),
        "task_name": str(task_name),
        "split": str(split),
        "rank": int(rank),
        "world_size": int(world_size),
        "shard_algorithm": SHARD_ALGORITHM,
        "contract_hash": str(contract_hash),
        "query_dim": QUERY_DIM,
        "dtype": QUERY_DTYPE,
        "sample_ids": [str(value) for value in sample_ids],
        "queries": queries,
        "query_tensor_hash": query_tensor_hash(sample_ids, queries),
        "stats": dict(stats),
    }
    _atomic_torch_save(payload, Path(path))
    _atomic_write_bytes(
        Path(path + ".json"),
        (json.dumps({"kind": "partial_sidecar", "sample_ids": payload["sample_ids"],
                     "contract_hash": payload["contract_hash"],
                     "rank": payload["rank"], "world_size": payload["world_size"],
                     "query_tensor_hash": payload["query_tensor_hash"],
                     "stats": payload["stats"]}, indent=2, sort_keys=True) + "\n").encode(),
    )


def _as_clean_queries(sample_ids: Sequence[str], queries: Tensor) -> Tensor:
    if queries.ndim != 2 or queries.shape[1] != QUERY_DIM:
        raise ValueError("queries must have shape [N, {}]".format(QUERY_DIM))
    if queries.shape[0] != len(sample_ids):
        raise ValueError("{} sample ids but {} query rows".format(len(sample_ids), queries.shape[0]))
    queries = queries.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(queries).all()):
        raise ValueError("queries contain non-finite values")
    return queries


def load_partial(path: str) -> Tuple[List[str], Tensor, Dict[str, object]]:
    """(sample_ids, queries, meta) with structural corruption checks."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("partial {} is not a dict payload".format(path))
    if payload.get("kind") != "v7_query_shard_partial":
        raise ValueError("partial {} has wrong kind".format(path))
    if payload.get("query_dim") != QUERY_DIM or payload.get("dtype") != QUERY_DTYPE:
        raise ValueError("partial {} violates the dimension/dtype contract".format(path))
    sample_ids = [str(value) for value in payload["sample_ids"]]
    queries = payload["queries"]
    if not isinstance(queries, Tensor):
        raise ValueError("partial {} has no query tensor".format(path))
    queries = _as_clean_queries(sample_ids, queries)
    return sample_ids, queries, dict(payload)


def partial_valid_for(path: str, contract_hash: str, expected_ids: Sequence[str]) -> bool:
    """Resume gate for one shard: contract-bound and covering its exact slice."""
    target = Path(path)
    if not target.is_file():
        return False
    try:
        sample_ids, _queries, meta = load_partial(path)
    except Exception:
        return False
    if meta.get("contract_hash") != contract_hash:
        return False
    return list(sample_ids) == [str(value) for value in expected_ids]


def merge_partials(
    partial_paths: Sequence[str],
    *,
    contract_hash: str,
    declared_ids: Sequence[str],
    declared_count: int,
) -> Tuple[List[str], Tensor, Dict[str, object]]:
    """Deterministically merge rank partials into one full-split cache row set.

    Returns (declared_order_ids, queries, audit).  Raises on any of the
    §9 merge audit failures so an interrupted/invalid run can never become
    an official cache.

    Rank r must cover exactly ``index % world_size == r`` over the declared
    order; rows are reassembled into the declared order.
    """
    if not partial_paths:
        raise ValueError("merge requires at least one partial")
    declared = [str(value) for value in declared_ids]
    if len(declared) != int(declared_count):
        raise ValueError("declared count mismatch: {} ids vs {} declared".format(
            len(declared), declared_count))
    if len(set(declared)) != len(declared):
        raise ValueError("declared sample ids are not unique")

    partials = {}
    for path in partial_paths:
        sample_ids, queries, meta = load_partial(path)
        if meta.get("contract_hash") != contract_hash:
            raise ValueError("partial {} has a stale contract ({} != {})".format(
                path, meta.get("contract_hash"), contract_hash))
        rank = int(meta["rank"])
        world = int(meta["world_size"])
        if rank in partials:
            raise ValueError("duplicate partial for rank {}".format(rank))
        partials[rank] = {"ids": sample_ids, "queries": queries, "world": world}

    ranks = sorted(partials)
    world_sizes = {partials[rank]["world"] for rank in ranks}
    if len(world_sizes) != 1:
        raise ValueError("partials disagree on world_size: {}".format(sorted(world_sizes)))
    world = world_sizes.pop()
    if ranks != list(range(world)):
        raise ValueError("missing partial ranks: have {}, need 0..{}".format(ranks, world - 1))

    merged_ids: List[str] = []
    audit = {}
    for rank in ranks:
        ids = partials[rank]["ids"]
        audit["rank{}_sample_count".format(rank)] = len(ids)
        merged_ids.extend(ids)
    # Coverage first: the union of all rank partials must equal the declared
    # sample set exactly (missing / duplicate / unknown samples surface here).
    if sorted(merged_ids) != sorted(declared):
        missing = sorted(set(declared) - set(merged_ids))
        foreign = sorted(set(merged_ids) - set(declared))
        raise ValueError("merge coverage mismatch: {} missing, {} foreign".format(
            len(missing), len(foreign)))
    # Then per-rank determinism: rank r must cover exactly index % world == r
    # over the declared order, in ascending declared order.
    for rank in ranks:
        expected = rank_sample_slice(declared, world, rank)
        ids = partials[rank]["ids"]
        if list(ids) != expected:
            raise ValueError(
                "rank {} does not cover its deterministic slice "
                "(missing/foreign/out-of-order samples)".format(rank)
            )

    # Reassemble into the declared order: global index % world == rank, and
    # each rank emitted its rows in ascending global order, so row j of rank
    # r belongs to global index (j * world + r).
    by_rank = {rank: partials[rank]["queries"] for rank in ranks}
    row_position = [0] * world
    full_rows = []
    for global_index in range(len(declared)):
        rank = global_index % world
        local = row_position[rank]
        row_position[rank] += 1
        full_rows.append(by_rank[rank][local])
    queries = torch.stack(full_rows).to(torch.float32).contiguous()
    if queries.shape != (len(declared), QUERY_DIM):
        raise ValueError("merged query shape is {}".format(queries.shape))

    audit.update({
        "merged_count": len(merged_ids),
        "unique_merged_ids": len(set(merged_ids)),
        "declared_count": int(declared_count),
        "duplicate_sample_ids": 0,
        "missing_sample_ids": 0,
        "unknown_sample_ids": 0,
        "query_dim": QUERY_DIM,
        "query_dtype": QUERY_DTYPE,
        "all_finite": bool(torch.isfinite(queries).all()),
    })
    norms = queries.float().norm(dim=-1)
    audit["norm_min"] = float(norms.min())
    audit["norm_max"] = float(norms.max())
    audit["norm_mean"] = float(norms.mean())
    audit["count_equality"] = bool(len(merged_ids) == int(declared_count))
    return list(declared), queries, audit


def write_split_cache(
    directory: str,
    *,
    contract: SplitContract,
    sample_ids: Sequence[str],
    queries: Tensor,
    merge_audit: Mapping[str, object],
    worker_stats: Mapping[str, object],
    runtime: Mapping[str, object],
    created_at: str,
) -> Tuple[Path, Dict[str, object]]:
    """Official split artifact: queries.pt + metadata.json (atomic writes)."""
    sample_ids = [str(value) for value in sample_ids]
    queries = _as_clean_queries(sample_ids, queries)
    folder = Path(directory)
    queries_path = folder / "queries.pt"
    metadata_path = folder / "metadata.json"
    query_hash = query_tensor_hash(sample_ids, queries)
    payload = {
        "schema_version": QUERY_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "task_index": contract.task_index,
        "task_name": contract.task_name,
        "split": contract.split,
        "query_dim": QUERY_DIM,
        "dtype": QUERY_DTYPE,
        "sample_ids": sample_ids,
        "queries": queries,
        "query_tensor_hash": query_hash,
        "contract_hash": contract.contract_hash(),
    }
    _atomic_torch_save(payload, queries_path)
    metadata = {
        "schema_version": QUERY_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash(),
        "num_declared_samples": len(sample_ids),
        "num_saved_queries": int(queries.shape[0]),
        "num_unique_sample_ids": len(set(sample_ids)),
        "ordered_sample_ids": sample_ids,
        "sample_id_set_hash": sample_id_set_hash(sample_ids),
        "query_tensor_hash": query_hash,
        "queries_file": str(queries_path.resolve()),
        "merge_audit": {key: value for key, value in merge_audit.items()},
        "worker_stats": dict(worker_stats),
        "runtime": dict(runtime),
        "created_at": created_at,
    }
    _atomic_write_bytes(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return queries_path, metadata


def split_cache_valid(directory: str, contract_hash: str) -> bool:
    """Official cache is present, contract-current and internally audited."""
    folder = Path(directory)
    queries_path = folder / "queries.pt"
    metadata_path = folder / "metadata.json"
    if not queries_path.is_file() or not metadata_path.is_file():
        return False
    try:
        payload = torch.load(queries_path, map_location="cpu")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict) or not isinstance(meta, dict):
        return False
    if payload.get("kind") != CACHE_KIND:
        return False
    if payload.get("contract_hash") != contract_hash or meta.get("contract_hash") != contract_hash:
        return False
    return True


def read_split_cache(directory: str) -> Tuple[Tuple[str, ...], Tensor]:
    """(sample_ids, queries) for cache reuse (trainer / RMS / eval / pruning).

    Raises on structural corruption so a stale or tampered cache can never
    be silently consumed.
    """
    folder = Path(directory)
    payload = torch.load(folder / "queries.pt", map_location="cpu")
    if not isinstance(payload, dict) or payload.get("kind") != CACHE_KIND:
        raise ValueError("not a V7 fixed-query split cache: {}".format(directory))
    if payload.get("query_dim") != QUERY_DIM or payload.get("dtype") != QUERY_DTYPE:
        raise ValueError("split cache violates the 1536-D float32 contract")
    sample_ids = tuple(str(value) for value in payload["sample_ids"])
    queries = payload["queries"]
    if not isinstance(queries, Tensor):
        raise ValueError("split cache has no query tensor")
    queries = _as_clean_queries(sample_ids, queries)
    return sample_ids, queries


def load_split_cache_for_training(
    directory: str,
    *,
    expected_contract_hash: Optional[str] = None,
    expected_ids: Optional[Sequence[str]] = None,
    verify_value_hash: bool = True,
    mmap: bool = True,
) -> Tuple[Tensor, Dict[str, int], str, Dict[str, object]]:
    """Load ``queries.pt`` for direct consumption by the training dataset.

    The training path wants one ``[N, QUERY_DIM]`` tensor plus an id -> row
    index, not the pipeline's per-sample JSON document (``train.json`` is
    ~1.3 GB of text and costs ~25 s and ~3.5 GiB RSS to parse for a 40 k
    split, for data that is bit-identical to this tensor).  This is the
    existence-detecting, fingerprint-checked loader for that path:

    * ``queries.pt`` and its ``metadata.json`` sidecar must both exist, else
      :class:`FileNotFoundError` -- the caller decides whether to fall back.
    * structural contract checks (cache kind, 1536-D float32, one row per id,
      unique ids) raise :class:`ValueError`;
    * ``verify_value_hash`` recomputes :func:`query_tensor_hash` and demands it
      equal the value recorded both inside the payload and in the sidecar, so
      a corrupted or truncated file can never be consumed silently
      (~0.8 s for 40 k x 1536 float32, i.e. 244 MB);
    * ``expected_contract_hash`` is the *invalidation* check: when the caller
      knows which runtime contract produced this training run, a mismatch
      means the cache is stale and must be regenerated;
    * ``expected_ids`` is the *coverage* check: a superset of the dataset ids
      is required, and the count must match exactly.

    Returns ``(queries, rows_by_sample_id, value_hash, metadata)``.

    ``mmap=True`` maps the tensor instead of reading it (0.011 s vs 0.307 s
    for 244 MB); the pages fault in on first touch, and every downstream
    consumer copies rows into collated batches, so nothing mutates in place.
    """
    folder = Path(directory)
    queries_path = folder / "queries.pt"
    metadata_path = folder / "metadata.json"
    if not queries_path.is_file():
        raise FileNotFoundError("V7 query tensor not found: {}".format(queries_path))
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "V7 query tensor sidecar not found: {}".format(metadata_path)
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("invalid V7 query metadata: {}".format(metadata_path))
    payload = torch.load(
        queries_path, map_location="cpu", mmap=mmap, weights_only=False
    )
    if not isinstance(payload, dict) or payload.get("kind") != CACHE_KIND:
        raise ValueError("not a V7 fixed-query split cache: {}".format(queries_path))
    if payload.get("query_dim") != QUERY_DIM or payload.get("dtype") != QUERY_DTYPE:
        raise ValueError("split cache violates the 1536-D float32 contract")
    sample_ids = [str(value) for value in payload["sample_ids"]]
    queries = payload["queries"]
    if not isinstance(queries, Tensor):
        raise ValueError("split cache has no query tensor")
    if queries.shape[0] != len(sample_ids):
        raise ValueError(
            "{} sample ids but {} query rows".format(len(sample_ids), queries.shape[0])
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("split cache has duplicate sample ids")
    if int(metadata.get("num_saved_queries", len(sample_ids))) != len(sample_ids):
        raise ValueError(
            "split cache count mismatch: payload={}, metadata={}".format(
                len(sample_ids), metadata.get("num_saved_queries")
            )
        )
    if str(metadata.get("query_tensor_hash")) != str(payload.get("query_tensor_hash")):
        raise ValueError(
            "split cache metadata/payload fingerprint mismatch: {} vs {}".format(
                metadata.get("query_tensor_hash"), payload.get("query_tensor_hash")
            )
        )
    if expected_contract_hash is not None:
        for source, value in (
            ("payload", payload.get("contract_hash")),
            ("metadata", metadata.get("contract_hash")),
        ):
            if str(value) != str(expected_contract_hash):
                raise ValueError(
                    "stale V7 query cache ({} contract {} != {}); regenerate it "
                    "with compose.eval.precompute_v7_queries".format(
                        source, value, expected_contract_hash
                    )
                )
    value_hash = str(payload.get("query_tensor_hash"))
    if verify_value_hash:
        recomputed = query_tensor_hash(sample_ids, queries)
        if recomputed != value_hash:
            raise ValueError(
                "V7 query cache value fingerprint mismatch: recorded {}, "
                "recomputed {}".format(value_hash, recomputed)
            )
    ordered_ids = metadata.get("ordered_sample_ids")
    if ordered_ids is not None and [str(v) for v in ordered_ids] != sample_ids:
        raise ValueError(
            "split cache id order disagrees with its metadata sidecar: the "
            "tensor and {} were written from different runs".format(metadata_path)
        )
    rows = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if expected_ids is not None:
        missing = [str(value) for value in expected_ids if str(value) not in rows]
        if missing:
            raise ValueError(
                "V7 query tensor misses {} train samples (first: {})".format(
                    len(missing), missing[:3]
                )
            )
        if len(expected_ids) != len(rows):
            raise ValueError(
                "V7 full-data query coverage mismatch: train={}, tensor={}".format(
                    len(expected_ids), len(rows)
                )
            )
    return queries, rows, value_hash, metadata


def write_task_center(
    directory: str,
    *,
    contract: SplitContract,
    queries: Tensor,
    source_query_cache_hash: str,
    created_at: str,
) -> Tuple[Path, Dict[str, object]]:
    """Full-train task center: mu_t = L2Norm(mean_i(q_i)) over the *entire*
    declared train split (math delegated to compose.v7.query)."""
    from .query import full_train_task_center

    num_train = int(queries.shape[0])
    center, coverage = full_train_task_center(queries, num_train)
    payload = {
        "schema_version": QUERY_SCHEMA_VERSION,
        "kind": TASK_CENTER_KIND,
        "task_index": contract.task_index,
        "task_name": contract.task_name,
        "center": center.detach().cpu().to(torch.float32).contiguous(),
        "num_queries_used_for_center": int(coverage["num_queries_used_for_center"]),
        "num_declared_train_samples": num_train,
        "source_query_cache_hash": source_query_cache_hash,
        "contract_hash": contract.contract_hash(),
    }
    target = Path(directory)
    queries_path = target / "task_center.pt"
    _atomic_torch_save(payload, queries_path)
    _atomic_write_bytes(
        target / "task_center.json",
        (json.dumps(
            {key: value for key, value in payload.items() if key != "center"},
            indent=2, sort_keys=True,
        ) + "\n").encode("utf-8"),
    )
    return queries_path, dict(payload)


def load_task_center(path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("kind") != TASK_CENTER_KIND:
        raise ValueError("not a V7 task-center file: {}".format(path))
    return payload


def compare_by_sample_id(
    a_ids: Sequence[str], a: Tensor, b_ids: Sequence[str], b: Tensor
) -> Dict[str, object]:
    """Per-sample numerical comparison of two query sets (same-id alignment)."""
    a_ids = [str(value) for value in a_ids]
    b_ids = [str(value) for value in b_ids]
    if len(a_ids) != len(set(a_ids)) or len(b_ids) != len(set(b_ids)):
        raise ValueError("compare requires unique ids")
    a_index = {value: position for position, value in enumerate(a_ids)}
    b_index = {value: position for position, value in enumerate(b_ids)}
    common = [value for value in a_ids if value in b_index]
    if len(common) != len(a_ids):
        raise ValueError("comparison id sets are not identical")
    a = a.float().cpu()
    b = b.float().cpu()
    diff = (a - b).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    eps = 1.0e-8
    relative = (diff / a.abs().clamp(min=eps)).max(dim=-1).values
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {
        "common_samples": len(common),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_relative_diff": float(relative.max()),
        "mean_relative_diff": float(relative.mean()),
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "exact_bit_equal": bool(torch.equal(a, b)),
    }


def norm_stats(queries: Tensor) -> Dict[str, object]:
    """L2-norm audit of the cached query rows (L2-normalized contract)."""
    q = queries.detach().float()
    norms = q.norm(dim=-1)
    return {
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
    }


def verify_route_agreement(
    pool_state_path: str,
    a_ids: Sequence[str],
    a_queries: Tensor,
    b_ids: Sequence[str],
    b_queries: Tensor,
    exclude: Sequence[int] = (),
) -> Dict[str, object]:
    """Global Top-2 equivalence of two query sets against one expert pool.

    The pool is a V7 key state (candidate or committed); this is a routing
    *consistency* smoke, not a performance experiment.
    """
    from .pool import V7ExpertKeyPool
    from .routing import GlobalTop2Router

    if list(a_ids) != list(b_ids):
        raise ValueError("route agreement requires aligned sample ids")
    state = torch.load(pool_state_path, map_location="cpu", weights_only=False)
    pool = V7ExpertKeyPool.from_state(state)
    router = GlobalTop2Router(pool).eval()
    with torch.no_grad():
        result_a = router(a_queries.detach().float(), excluded=exclude)
        result_b = router(b_queries.detach().float(), excluded=exclude)
    top2_a = result_a.expert_ids.detach().cpu()
    top2_b = result_b.expert_ids.detach().cpu()
    score_a = result_a.similarities.detach().cpu()
    score_b = result_b.similarities.detach().cpu()
    row_agreement = (top2_a == top2_b).all(dim=-1)
    disagreements = [
        {"sample_id": str(a_ids[index]), "single": top2_a[index].tolist(),
         "multi": top2_b[index].tolist()}
        for index in range(len(a_ids))
        if not bool(row_agreement[index])
    ]
    return {
        "samples": int(len(a_ids)),
        "top2_agreement_rate": float(row_agreement.float().mean()),
        "agreement_exact": bool(row_agreement.all()),
        "disagreements": disagreements[:20],
        "max_abs_score_diff": float((score_a - score_b).abs().max()),
        "pool_path": str(pool_state_path),
        "visible_expert_ids": list(pool.selectable_ids(exclude)),
    }


# ---------------------------------------------------------------------------
# Runtime cache readers (downstream stages: S1 adapter, S3, RMS, pruning, eval)
#
# One shared, read-only, sample_id-keyed component is the only cache entry
# point for every downstream stage.  Tensors stay on CPU storage and are
# moved to the caller's device on demand; every returned row is detached,
# requires_grad=False and never enters the query encoder backward graph.
# A cache miss ALWAYS raises QueryCacheMissError (fail closed) - the formal
# recipe never recomputes a query from a live encoder.
# ---------------------------------------------------------------------------

class QueryCacheMissError(KeyError):
    """Raised when a requested sample id is not in the split cache.

    Formal stages must treat this as a hard failure (fail closed): a miss
    means the sample was not part of the declared, contract-bound cache and
    re-computing it live would break the single-contract guarantee.
    """


class V7QuerySplitReader:
    """Read-only access to one split cache, keyed by sample_id.

    ``queries`` stay in CPU memory (fp32 storage) and ``get``/``get_batch``
    return fresh detached CPU/device tensors; no gradient edge is ever
    created through the cache.
    """

    def __init__(
        self,
        directory: str,
        *,
        expected_count: Optional[int] = None,
    ) -> None:
        self.directory = str(Path(directory).expanduser().resolve())
        folder = Path(self.directory)
        payload = torch.load(folder / "queries.pt", map_location="cpu")
        if not isinstance(payload, dict) or payload.get("kind") != CACHE_KIND:
            raise ValueError("not a V7 fixed-query split cache: {}".format(self.directory))
        if payload.get("query_dim") != QUERY_DIM or payload.get("dtype") != QUERY_DTYPE:
            raise ValueError("split cache violates the 1536-D float32 contract")
        sample_ids = tuple(str(value) for value in payload["sample_ids"])
        queries = _as_clean_queries(sample_ids, payload["queries"])
        if len(sample_ids) != queries.shape[0]:
            raise ValueError("split cache id/query row mismatch")
        if expected_count is not None and len(sample_ids) != int(expected_count):
            raise ValueError(
                "split cache has {} samples, expected {}".format(
                    len(sample_ids), int(expected_count)
                )
            )
        self.sample_ids: Tuple[str, ...] = sample_ids
        self.queries: Tensor = queries
        self.n: int = len(sample_ids)
        self._index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
        if len(self._index) != self.n:
            raise ValueError("split cache contains duplicate sample ids")
        self.metadata: Dict[str, object] = {}
        metadata_path = folder / "metadata.json"
        if metadata_path.is_file():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # -- core lookups -------------------------------------------------------

    def contains(self, sample_id: str) -> bool:
        return str(sample_id) in self._index

    def get(self, sample_id: str, device: Optional[str] = None) -> Tensor:
        """One row q_i (1536,) fp32, detached; miss raises QueryCacheMissError."""
        key = str(sample_id)
        if key not in self._index:
            raise QueryCacheMissError(
                "query cache miss for sample {} in {}".format(key, self.directory)
            )
        return self.queries[self._index[key]].detach().to(device=device)

    def get_batch(self, sample_ids: Sequence[str], device: Optional[str] = None) -> Tensor:
        """Rows [B,1536] fp32 in the requested id order (order-preserving).

        Any unknown id raises QueryCacheMissError listing the misses, so a
        formal batch can fail closed before any routing happens.
        """
        keys = [str(value) for value in sample_ids]
        missing = [value for value in keys if value not in self._index]
        if missing:
            raise QueryCacheMissError(
                "query cache misses {} of {} samples in {}: {}".format(
                    len(missing), len(keys), self.directory, missing[:8]
                )
            )
        rows = self.queries[[self._index[value] for value in keys]]
        return rows.detach().to(device=device)

    # -- audit --------------------------------------------------------------

    def validate_split(self) -> Dict[str, object]:
        """Independent re-audit of the on-disk split (dim, dtype, ids, norms)."""
        q = self.queries
        audit = {
            "kind": CACHE_KIND,
            "saved_count": int(self.n),
            "unique_sample_id_count": len(self._index),
            "duplicate_sample_ids": int(self.n - len(self._index)),
            "query_dim": int(q.shape[1]) if q.ndim == 2 else -1,
            "query_dtype": str(q.dtype).split(".", 1)[-1],  # "torch.float32" -> "float32"
            "all_finite": bool(torch.isfinite(q).all()),
        }
        if q.ndim == 2 and q.shape[1] == QUERY_DIM:
            stats = norm_stats(q.detach().cpu())
            audit["norm_mean"] = stats["norm_mean"]
            audit["norm_min"] = stats["norm_min"]
            audit["norm_max"] = stats["norm_max"]
            audit["norm_in_tolerance"] = bool(
                (1.0 - stats["norm_min"]) <= 1e-6 and (stats["norm_max"] - 1.0) <= 1e-6
            )
        meta = self.metadata
        if meta:
            audit["metadata_kind"] = meta.get("kind")
            audit["num_declared_samples"] = meta.get("num_declared_samples")
            audit["num_saved_queries"] = meta.get("num_saved_queries")
            audit["num_unique_sample_ids"] = meta.get("num_unique_sample_ids")
            audit["contract_hash"] = meta.get("contract_hash")
            audit["query_tensor_hash"] = meta.get("query_tensor_hash")
        return audit

    def rehash(self) -> Dict[str, str]:
        """Byte-level rehash of ids and tensor (tamper detection)."""
        return {
            "sample_id_set_hash": sample_id_set_hash(self.sample_ids),
            "query_tensor_hash": query_tensor_hash(list(self.sample_ids), self.queries),
        }


class V7CacheManifest:
    """Validated handle over a full precomputed query-cache manifest.

    Holds the six-task (train/val/test) layout, binds every split through
    its contract hash, and verifies at construction that the manifest's
    runtime contract matches the *current* process contract (git head,
    query backbone content hash, query implementation hash, schema).
    """

    def __init__(self, manifest_path: str) -> None:
        self.path = str(Path(manifest_path).expanduser().resolve())
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        if payload.get("kind") != "v7_fixed_query_cache_manifest":
            raise ValueError("not a V7 query-cache manifest: {}".format(self.path))
        self.payload = payload
        self.cache_root = str(payload["cache_root"])
        self.query_mode = str(payload["query_mode"])
        self.schema_version = int(payload["schema_version"])
        self.query_dim = int(payload["query_dim"])
        self.dtype = str(payload["dtype"])
        self.tasks: Dict[str, Dict[str, Dict[str, object]]] = payload["tasks"]
        # The manifest keys tasks by *name*; the only index binding is the
        # ``/task{N}/`` directory inside each split's artifact path.
        self._task_to_index: Dict[str, int] = {}
        for task_name, splits in payload["tasks"].items():
            probe = next(iter(splits.values()))["path"]
            match = re.search(r"/task(\d+)/", str(probe))
            if not match:
                raise ValueError(
                    "cannot infer task index from split path {}".format(probe)
                )
            self._task_to_index[str(task_name)] = int(match.group(1))
        self._index_to_task = {
            index: name for name, index in self._task_to_index.items()
        }

    @staticmethod
    def locate(cache_root_or_manifest: str) -> "V7CacheManifest":
        """Accept a run root (dir containing query_cache_manifest.json) or the
        manifest file itself, resolving to the manifest either way."""
        candidate = Path(cache_root_or_manifest)
        if candidate.is_file():
            return V7CacheManifest(str(candidate))
        if (candidate / "query_cache_manifest.json").is_file():
            return V7CacheManifest(str(candidate / "query_cache_manifest.json"))
        raise ValueError("no query-cache manifest at {}".format(cache_root_or_manifest))

    def manifest_sha256(self) -> str:
        return sha256_file(self.path)

    # -- split location -----------------------------------------------------

    def split_dir(self, task_index: int, split: str) -> str:
        return str(Path(self.cache_root) / "task{}".format(int(task_index)) / split)

    def split_info(self, task_index: int, split: str) -> Dict[str, object]:
        return dict(self.payload["tasks"][str(self.task_name(task_index))][split])

    def task_index(self, task_name: str) -> int:
        if str(task_name) not in self._task_to_index:
            raise ValueError("unknown task in manifest: {}".format(task_name))
        return self._task_to_index[str(task_name)]

    def task_name(self, task_index: int) -> str:
        if int(task_index) not in self._index_to_task:
            raise ValueError(
                "manifest has no task with index {}".format(task_index)
            )
        return self._index_to_task[int(task_index)]

    def reader(self, task_index: int, split: str) -> V7QuerySplitReader:
        """Split reader bound to the manifest-declared count for that split."""
        info = self.split_info(task_index, split)
        reader = V7QuerySplitReader(self.split_dir(task_index, split))
        if int(info["sample_count"]) != reader.n:
            raise ValueError(
                "manifest declares {} samples for task{}.{} but cache has {}".format(
                    int(info["sample_count"]), int(task_index), split, reader.n
                )
            )
        return reader

    def split_contract_hash(self, task_index: int, split: str) -> str:
        return str(self.split_info(task_index, split)["contract_hash"])

    # -- runtime contract ---------------------------------------------------

    def validate_runtime_contract(
        self,
        *,
        git_sha: Optional[str] = None,
        git_branch: Optional[str] = None,
        backbone_hash: Optional[str] = None,
        impl_hash: Optional[str] = None,
        required: Sequence[str] = ("train", "val"),
    ) -> Dict[str, object]:
        """Bind the manifest to the current runtime process, fail closed.

        Checks, cheaply (sidecar metadata only, no tensor loads):

        - manifest schema/dim/dtype/query-mode match the module contract;
        - ``git_sha``/``git_branch`` (when given) match the manifest and the
          per-split ``metadata.json`` contract the manifest entry binds to
          (``contract_hash`` equality is the tamper check);
        - ``backbone_hash``/``impl_hash`` (when given) match every required
          split's metadata contract (the producer identity chain);
        - every ``required`` split exists in the manifest and on disk with
          the declared sample count.

        Returns a machine-readable verdict and raises ValueError on the
        first problem so callers fail closed before any routing happens.
        """
        problems: List[str] = []
        checks: Dict[str, object] = {}
        if self.schema_version != QUERY_SCHEMA_VERSION:
            problems.append("manifest schema {} != cache schema {}".format(
                self.schema_version, QUERY_SCHEMA_VERSION))
        if self.query_dim != QUERY_DIM or self.dtype != QUERY_DTYPE:
            problems.append("manifest dim/dtype mismatch: {}/{}".format(
                self.query_dim, self.dtype))
        if self.query_mode != QUERY_MODE:
            problems.append("manifest query mode {} != {}".format(
                self.query_mode, QUERY_MODE))
        if git_sha is not None and str(self.payload.get("git_sha")) != str(git_sha):
            problems.append("manifest git {} != runtime {}".format(
                self.payload.get("git_sha"), git_sha))
        if git_branch is not None and str(self.payload.get("git_branch")) != str(git_branch):
            problems.append("manifest branch {} != runtime {}".format(
                self.payload.get("git_branch"), git_branch))
        for task_index in sorted(self._task_to_index.values()):
            task_name = self.task_name(task_index)
            splits = self.tasks[task_name]
            for split in required:
                entry = "task{}.{}".format(task_index, split)
                if split not in splits:
                    problems.append("{} is missing from the manifest".format(entry))
                    continue
                info = splits[split]
                metadata_path = Path(self.split_dir(task_index, split)) / "metadata.json"
                if not metadata_path.is_file():
                    problems.append("{} metadata.json is missing".format(entry))
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if str(metadata.get("contract_hash")) != str(info["contract_hash"]):
                    problems.append(
                        "{} metadata contract_hash disagrees with the manifest".format(entry)
                    )
                if int(metadata.get("num_declared_samples", -1)) != int(info["sample_count"]):
                    problems.append(
                        "{} declared sample count disagrees with the manifest".format(entry)
                    )
                if metadata.get("kind") != CACHE_KIND:
                    problems.append("{} is not a V7 split cache".format(entry))
                contract = metadata.get("contract") or {}
                for field, runtime_value in (
                    ("git_sha", git_sha),
                    ("query_backbone_hash", backbone_hash),
                    ("query_impl_hash", impl_hash),
                ):
                    if runtime_value is not None and str(
                        contract.get(field, "")
                    ) != str(runtime_value):
                        problems.append(
                            "{} contract {} {} != runtime {}".format(
                                entry, field, contract.get(field), runtime_value
                            )
                        )
                checks[entry] = {
                    "sample_count": int(info["sample_count"]),
                    "contract_hash": str(info["contract_hash"]),
                    "contract_bound_to_manifest": bool(
                        str(metadata.get("contract_hash")) == str(info["contract_hash"])
                    ),
                    "contract_git_sha": contract.get("git_sha"),
                    "contract_backbone_hash": contract.get("query_backbone_hash"),
                    "contract_impl_hash": contract.get("query_impl_hash"),
                }
        verdict = {
            "manifest_path": self.path,
            "manifest_sha256": self.manifest_sha256(),
            "schema_match": bool(not problems),
            "problems": problems,
            "runtime_checks": checks,
        }
        if problems:
            raise ValueError(
                "query-cache runtime contract mismatch: {}".format("; ".join(problems))
            )
        return verdict


def validate_split(
    directory: str,
    *,
    expected_count: Optional[int] = None,
) -> Dict[str, object]:
    """Standalone split audit: dim/dtype, id uniqueness, finite, L2 norms,
    and metadata/declared-count agreement (spec §5 ``cache.validate_split``)."""
    return V7QuerySplitReader(directory, expected_count=expected_count).validate_split()


get_task_center = load_task_center  # spec §5 ``cache.get_task_center``
