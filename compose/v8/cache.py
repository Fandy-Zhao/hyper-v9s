"""Persistent teacher cache.

The teacher's verdicts are expensive (they cost generations), so they are
written once and reloaded by the key-learning and training stages.  Two
properties are non-negotiable and are enforced here rather than trusted:

**No ground truth.**  The cache records *decisions* -- teacher state, selected
experts, metric values, key targets -- and never the answer string that produced
them.  ``_forbid_answer_fields`` scans every payload on write and on read, so a
future edit that tries to smuggle answers into the training cache fails loudly
instead of silently turning the training signal into a leak.

**Determinism.**  Writing and reloading must reproduce a byte-identical
canonical digest, otherwise resume would silently train on different labels.
:func:`teacher_result_checksum` is the digest, and :func:`assert_roundtrip`
proves the property.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from compose.v8.teacher import TeacherResult, TeacherSampleRecord


class TeacherCacheError(RuntimeError):
    """Raised when a cache would carry ground truth or fail to round-trip."""


#: Field names that would mean raw supervision leaked into a training artifact.
_FORBIDDEN_FIELD_TOKENS = (
    "ground_truth",
    "groundtruth",
    "gt_answer",
    "answer_text",
    "gold_answer",
    "reference_answer",
    "raw_answer",
    "label_text",
)

CACHE_VERSION = 1


def _forbid_answer_fields(payload: Any, path: str = "$") -> None:
    """Reject any mapping key that would carry a supervision string."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            token = str(key).lower()
            for forbidden in _FORBIDDEN_FIELD_TOKENS:
                if forbidden in token:
                    raise TeacherCacheError(
                        f"teacher cache must not store supervision: {path}.{key} "
                        f"matches {forbidden!r}"
                    )
            _forbid_answer_fields(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _forbid_answer_fields(value, f"{path}[{index}]")


def record_to_payload(record: TeacherSampleRecord) -> Dict[str, Any]:
    payload = record.to_dict()
    _forbid_answer_fields(payload)
    return payload


def record_from_payload(payload: Mapping[str, Any]) -> TeacherSampleRecord:
    _forbid_answer_fields(payload)
    return TeacherSampleRecord(
        sample_id=str(payload["sample_id"]),
        state=str(payload["state"]),
        selected_experts=[int(value) for value in payload["selected_experts"]],
        base_solved=bool(payload["base_solved"]),
        base_value=float(payload["base_value"]),
        recall=[int(value) for value in payload["recall"]],
        single_values={str(k): float(v) for k, v in payload["single_values"].items()},
        pair_values={str(k): float(v) for k, v in payload["pair_values"].items()},
        tested_singles=[int(value) for value in payload["tested_singles"]],
        tested_pairs=[[int(v) for v in pair] for pair in payload["tested_pairs"]],
        achieved_value=float(payload["achieved_value"]),
        achieved_nll=(None if payload["achieved_nll"] is None
                      else float(payload["achieved_nll"])),
        delta_nll=(None if payload["delta_nll"] is None else float(payload["delta_nll"])),
        teacher_gain=float(payload["teacher_gain"]),
        key_targets={str(k): str(v) for k, v in payload["key_targets"].items()},
        decision_reason=str(payload["decision_reason"]),
        # A resume must keep the residual context: it is the historical
        # composition a Residual sample is trained with.  Defaulted so a cache
        # written before this field existed still loads.
        residual_context=[int(value) for value in payload.get("residual_context", [])],
        solved_threshold=float(payload.get("solved_threshold", 0.0)),
    )


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def teacher_result_checksum(result: TeacherResult) -> str:
    """Canonical digest of a teacher result (order-independent).

    Covers exactly what a reload reproduces: the task id and the per-sample
    records.  ``config`` is **excluded on purpose** -- it is run provenance, not
    label identity, and :func:`read_teacher_result` cannot rebuild the live
    config object.  Including it would make every cache fail its own integrity
    check the moment it was reopened.  The manifest keeps the config and its own
    ``config_sha256`` so a resume can still compare the run contract.
    """
    records = sorted((record_to_payload(record) for record in result.records),
                     key=lambda item: item["sample_id"])
    payload = {
        "task_id": int(result.task_id),
        "records": records,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def write_teacher_result(
    root: str | Path,
    result: TeacherResult,
    provenance: Optional[Mapping[str, Any]] = None,
    write_route_scores: bool = True,
) -> Dict[str, Any]:
    """Write a teacher result plus provenance; returns the manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _forbid_answer_fields(dict(provenance or {}))
    digest = teacher_result_checksum(result)

    records_path = root / "teacher_records.jsonl"
    tmp = records_path.with_name(records_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in sorted(result.records, key=lambda item: item.sample_id):
            handle.write(json.dumps(record_to_payload(record), sort_keys=True,
                                    ensure_ascii=False) + "\n")
    tmp.replace(records_path)

    route_path = None
    if write_route_scores:
        route_path = root / "teacher_route_scores.jsonl"
        tmp = route_path.with_name(route_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in result.scored_routes:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        tmp.replace(route_path)

    manifest = {
        "cache_version": CACHE_VERSION,
        "task_id": int(result.task_id),
        "samples": len(result.records),
        "teacher_result_sha256": digest,
        "config": result.config,
        "config_sha256": hashlib.sha256(
            _canonical(result.config).encode("utf-8")
        ).hexdigest(),
        "state_counts": result.state_counts(),
        "positive_counts": {str(k): v for k, v in sorted(result.positive_counts().items())},
        "records_file": records_path.name,
        "route_scores_file": route_path.name if route_path else None,
        "provenance": dict(provenance or {}),
        "stores_ground_truth": False,
    }
    manifest_path = root / "teacher_manifest.json"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest


def read_teacher_result(root: str | Path) -> TeacherResult:
    """Reload a cached teacher result and verify it against its manifest."""
    root = Path(root)
    manifest_path = root / "teacher_manifest.json"
    if not manifest_path.exists():
        raise TeacherCacheError(f"no teacher manifest under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stores_ground_truth") is not False:
        raise TeacherCacheError("teacher cache declares ground truth storage")
    records_path = root / str(manifest["records_file"])
    records: List[TeacherSampleRecord] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(record_from_payload(json.loads(line)))
    result = TeacherResult(
        task_id=int(manifest["task_id"]),
        records=records,
        config={"restored_from_cache": True, "cache_version": manifest["cache_version"]},
        scored_routes=[],
    )
    digest = teacher_result_checksum(result)
    if manifest.get("teacher_result_sha256") not in (None, digest):
        raise TeacherCacheError(
            "cached teacher records do not reproduce their recorded digest: "
            f"{digest} != {manifest['teacher_result_sha256']}"
        )
    route_path = manifest.get("route_scores_file")
    if route_path and (root / str(route_path)).exists():
        with (root / str(route_path)).open("r", encoding="utf-8") as handle:
            result.scored_routes = [json.loads(line) for line in handle if line.strip()]
    return result


def assert_roundtrip(result: TeacherResult, root: str | Path) -> Dict[str, Any]:
    """Write, reload and prove the reload is digest-identical."""
    manifest = write_teacher_result(root, result)
    reloaded = read_teacher_result(root)
    before = teacher_result_checksum(result)
    after = teacher_result_checksum(reloaded)
    if before != after:
        raise TeacherCacheError(
            f"teacher cache is not deterministic: {before} != {after}"
        )
    if [record.sample_id for record in reloaded.records] != sorted(
        record.sample_id for record in result.records
    ):
        raise TeacherCacheError("reloaded teacher records are not in canonical order")
    return {
        "deterministic": True,
        "teacher_result_sha256": before,
        "samples": len(reloaded.records),
        "manifest": manifest,
    }


def merge_state_maps(*mappings: Mapping[str, str]) -> Dict[str, str]:
    """Combine per-shard state maps, refusing silent conflicts."""
    merged: Dict[str, str] = {}
    for mapping in mappings:
        for sample_id, state in mapping.items():
            sample_id = str(sample_id)
            if sample_id in merged and merged[sample_id] != state:
                raise TeacherCacheError(
                    f"sample {sample_id} has conflicting states "
                    f"{merged[sample_id]!r} and {state!r}"
                )
            merged[sample_id] = str(state)
    return merged


__all__ = [
    "CACHE_VERSION",
    "TeacherCacheError",
    "assert_roundtrip",
    "merge_state_maps",
    "read_teacher_result",
    "record_from_payload",
    "record_to_payload",
    "teacher_result_checksum",
    "write_teacher_result",
]
