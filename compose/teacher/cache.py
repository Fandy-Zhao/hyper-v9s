"""Versioned, atomic, sharded Oracle cache with strict invalidation."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .types import stable_hash
from .validity import validate_cache_provenance


CACHE_SCHEMA_VERSION = 1


def cache_key(provenance: Mapping[str, Any]) -> str:
    validate_cache_provenance(provenance)
    return stable_hash(dict(provenance))


def _records_checksum(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".pid{}-".format(os.getpid()), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_shard(path, records: Iterable[Mapping[str, Any]], provenance: Mapping[str, Any], rank: int) -> Dict[str, Any]:
    validate_cache_provenance(provenance)
    target = Path(path)
    rows = [dict(row) for row in records]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Oracle shard contains duplicate sample IDs")
    envelope = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "rank": int(rank),
        "provenance": dict(provenance),
        "cache_key": cache_key(provenance),
        "sample_count": len(rows),
        "sample_ids_sha256": stable_hash({"sample_ids": sorted(sample_ids)}),
        "records_sha256": _records_checksum(rows),
        "records": rows,
    }
    _atomic_json(target, envelope)
    return envelope


def load_shard(path, expected_provenance: Mapping[str, Any]) -> Dict[str, Any]:
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        envelope = json.load(handle)
    if int(envelope.get("cache_schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported or corrupt Oracle cache schema")
    if dict(envelope.get("provenance", {})) != dict(expected_provenance):
        raise ValueError("Oracle cache invalidated by provenance change")
    if envelope.get("cache_key") != cache_key(expected_provenance):
        raise ValueError("Oracle cache key mismatch")
    records = envelope.get("records")
    if not isinstance(records, list) or envelope.get("records_sha256") != _records_checksum(records):
        raise ValueError("Oracle cache record checksum mismatch")
    if int(envelope.get("sample_count", -1)) != len(records):
        raise ValueError("Oracle cache sample count mismatch")
    return envelope


def resume_sample_ids(path, expected_provenance: Mapping[str, Any]) -> Tuple[str, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    envelope = load_shard(target, expected_provenance)
    return tuple(str(row["sample_id"]) for row in envelope["records"])


def merge_shards(paths, output_path, expected_provenance: Mapping[str, Any], expected_sample_ids: Iterable[str]) -> Dict[str, Any]:
    rows = []
    ranks = []
    for path in paths:
        envelope = load_shard(path, expected_provenance)
        ranks.append(int(envelope["rank"]))
        rows.extend(envelope["records"])
    if len(ranks) != len(set(ranks)):
        raise ValueError("duplicate Oracle cache rank")
    identifiers = [str(row["sample_id"]) for row in rows]
    duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
    if duplicates:
        raise ValueError("duplicate Oracle samples across shards: {}".format(duplicates))
    expected = set(map(str, expected_sample_ids))
    actual = set(identifiers)
    if expected != actual:
        raise ValueError("Oracle shard merge missing={} unexpected={}".format(sorted(expected - actual), sorted(actual - expected)))
    rows.sort(key=lambda row: str(row["sample_id"]))
    return write_shard(output_path, rows, expected_provenance, rank=-1)
