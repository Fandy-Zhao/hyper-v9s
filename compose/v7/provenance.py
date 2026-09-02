"""Formal V7 split isolation and multimodal runtime provenance."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from compose.data.records import answer_text, question_text


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(values: Iterable[str]) -> str:
    encoded = json.dumps(sorted(str(value) for value in values), separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _record_identity(record: Mapping[str, object]) -> str:
    payload = {
        "image": str(record.get("image", "")),
        "question": question_text(record).strip(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _split_details(path: str) -> Dict[str, object]:
    normalized = str(Path(path).expanduser().resolve())
    records = json.loads(Path(normalized).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("split is empty or invalid: {}".format(normalized))
    sample_ids = [
        str(record.get("id", record.get("question_id", "source_index:{}".format(index))))
        for index, record in enumerate(records)
    ]
    identities = [_record_identity(record) for record in records]
    images = [str(record.get("image", "")) for record in records]
    questions = [question_text(record).strip() for record in records]
    answers = [answer_text(record).strip() for record in records]
    return {
        "normalized_source_path": normalized,
        "file_sha256": sha256_file(normalized),
        "sample_count": len(records),
        "sample_id_hash": _stable_hash(sample_ids),
        "source_record_identity_hash": _stable_hash(identities),
        "image_hash": _stable_hash(images),
        "question_hash": _stable_hash(questions),
        "answer_hash": _stable_hash(answers),
        "_sample_ids": set(sample_ids),
        "_identities": set(identities),
        "_images": set(images),
        "_questions": set(questions),
    }


def audit_split_isolation(
    train_file: str, val_file: str, test_file: Optional[str] = None
) -> Dict[str, object]:
    paths = {"train": train_file, "validation": val_file}
    if test_file:
        paths["test"] = test_file
    details = {name: _split_details(path) for name, path in paths.items()}
    names = list(details)
    overlaps = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pair = "{}_vs_{}".format(left, right)
            a, b = details[left], details[right]
            if a["normalized_source_path"] == b["normalized_source_path"]:
                raise ValueError("split leakage: {} use the same normalized path".format(pair))
            if a["file_sha256"] == b["file_sha256"]:
                raise ValueError("split leakage: {} have identical file hashes".format(pair))
            record_overlap = a["_identities"] & b["_identities"]
            id_overlap = a["_sample_ids"] & b["_sample_ids"]
            overlaps[pair] = {
                "sample_id_overlap": len(id_overlap),
                "source_record_overlap": len(record_overlap),
                "image_overlap": len(a["_images"] & b["_images"]),
                "question_overlap": len(a["_questions"] & b["_questions"]),
            }
            if id_overlap or record_overlap:
                raise ValueError(
                    "split leakage: {} sample-id/record overlap {}".format(
                        pair, overlaps[pair]
                    )
                )
    public = {
        name: {key: value for key, value in values.items() if not key.startswith("_")}
        for name, values in details.items()
    }
    return {
        "splits": public,
        "overlap_checks": overlaps,
        "test_data_used_for_training": False,
        "test_data_used_for_pruning": False,
    }


def build_runtime_contract(
    *, image_aspect_ratio: str, vision_tower: str, mm_vision_select_layer: int,
    mm_vision_select_feature: str, mm_projector_type: str, projector_path: str,
) -> Dict[str, object]:
    return {
        "image_aspect_ratio": str(image_aspect_ratio),
        "vision_tower": str(Path(vision_tower).expanduser().resolve()),
        "mm_vision_select_layer": int(mm_vision_select_layer),
        "mm_vision_select_feature": str(mm_vision_select_feature),
        "mm_projector_type": str(mm_projector_type),
        "projector_path": str(Path(projector_path).expanduser().resolve()),
        "projector_sha256": sha256_file(projector_path),
    }


def validate_runtime_contract(
    expected: Mapping[str, object], actual: Mapping[str, object], stage: str
) -> None:
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise ValueError("{} runtime preprocessing mismatch: {}".format(stage, mismatches))


def load_runtime_contract(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
