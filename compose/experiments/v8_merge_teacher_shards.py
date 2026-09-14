"""Merge deterministic V8 teacher shards into one audited training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(args) -> dict:
    train_path = Path(args.train_json).resolve()
    train = _read(train_path)
    expected_ids = {
        str(record.get("id", record.get("question_id"))) for record in train
    }
    if len(expected_ids) != len(train):
        raise ValueError("training JSON has missing or duplicate sample IDs")
    sample_manifest = None
    if args.teacher_sample_manifest is not None:
        sample_manifest = _read(Path(args.teacher_sample_manifest).resolve())
        expected_ids = {str(value) for value in sample_manifest["sample_ids"]}
        if not expected_ids.issubset({
            str(record.get("id", record.get("question_id"))) for record in train
        }):
            raise ValueError("teacher sample manifest contains IDs outside train split")
    elif args.limit is not None:
        expected_ids = set(sorted(expected_ids)[: int(args.limit)])

    shard_dirs = [Path(value).resolve() for value in args.shard_dir]
    configs = []
    payloads = []
    shard_indexes = set()
    shard_count = None
    all_records = {}
    provenance = []
    reference = None
    for directory in shard_dirs:
        complete = directory / "COMPLETE.json"
        teacher_path = directory / "teacher_result.json"
        config_path = directory / "run_config.json"
        if not complete.is_file():
            raise FileNotFoundError(f"incomplete shard: {directory}")
        payload = _read(teacher_path)
        config = _read(config_path)
        index = int(config["shard_index"])
        count = int(config["shard_count"])
        if shard_count is None:
            shard_count = count
        if count != shard_count or index in shard_indexes:
            raise ValueError("inconsistent or duplicate shard index/count")
        shard_indexes.add(index)
        invariant = {
            "task_id": payload["task_id"],
            "schema_version": payload["schema_version"],
            "teacher_search_mode": payload["teacher_search_mode"],
            "historical_experts_visible": payload["historical_experts_visible"],
            "config": payload["config"],
            "query_contract_hash": config["query_contract_hash"],
            "frozen_hashes": config["frozen_hashes"],
        }
        if reference is None:
            reference = invariant
        elif invariant != reference:
            raise ValueError(f"shard contract mismatch: {directory}")
        ids = []
        for record in payload["records"]:
            sample_id = str(record["sample_id"])
            if sample_id in all_records:
                raise ValueError(f"duplicate sample across shards: {sample_id}")
            all_records[sample_id] = record
            ids.append(sample_id)
        ordered_expected = sorted(expected_ids)
        expected_shard = set(ordered_expected[index::count])
        if set(ids) != expected_shard:
            raise ValueError(f"shard {index} does not match deterministic ID partition")
        configs.append(config)
        payloads.append(payload)
        provenance.append({
            "shard_index": index,
            "directory": str(directory),
            "samples": len(ids),
            "teacher_result_sha256": _sha256(teacher_path),
        })

    if shard_count is None or shard_indexes != set(range(shard_count)):
        raise ValueError(f"missing shards: got {sorted(shard_indexes)}, expected 0..{shard_count - 1}")
    actual_ids = set(all_records)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"merged coverage mismatch; missing={missing[:8]}, extra={extra[:8]}")

    records = [all_records[sample_id] for sample_id in sorted(all_records)]
    states = Counter(str(record["state"]) for record in records)
    for state in ("BaseOnly", "Reuse1", "Reuse2", "Residual"):
        states.setdefault(state, 0)
    visible = list(reference["historical_experts_visible"])
    searched = [record for record in records if not bool(record["base_solved"])]
    tested_sets = {tuple(record["historical_experts_tested"]) for record in searched}
    coverage = {
        "teacher_search_mode": reference["teacher_search_mode"],
        "historical_experts_visible": visible,
        "historical_experts_tested": visible,
        "never_tested": [],
        "num_visible_experts": len(visible),
        "base_only_samples": states["BaseOnly"],
        "searched_samples": len(searched),
        "fully_covered_samples": sum(
            set(record["historical_experts_tested"]) == set(visible)
            for record in searched
        ),
        "tested_set_sizes": sorted({len(values) for values in tested_sets}),
        "full_coverage": all(set(values) == set(visible) for values in tested_sets),
    }
    if not coverage["full_coverage"]:
        raise ValueError("merged teacher is not a full-history oracle")
    result = {
        "task_id": reference["task_id"],
        "schema_version": reference["schema_version"],
        "teacher_search_mode": reference["teacher_search_mode"],
        "historical_experts_visible": visible,
        "config": reference["config"],
        "states": dict(sorted(states.items())),
        "state_rates": {
            state: count / len(records) for state, count in sorted(states.items())
        },
        "coverage": coverage,
        "records": records,
    }
    output = Path(args.output).resolve()
    _write(output, result)
    audit = {
        "status": "COMPLETE",
        "task_id": reference["task_id"],
        "train_json": str(train_path),
        "train_json_sha256": _sha256(train_path),
        "expected_samples": len(expected_ids),
        "teacher_budget": args.limit,
        "teacher_sampling": sample_manifest,
        "merged_samples": len(records),
        "unique_samples": len(all_records),
        "shard_count": shard_count,
        "shards": sorted(provenance, key=lambda value: value["shard_index"]),
        "teacher_result": str(output),
        "teacher_result_sha256": _sha256(output),
        "states": result["states"],
        "coverage": coverage,
    }
    _write(output.with_name("MERGE_COMPLETE.json"), audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--teacher-sample-manifest", default=None)
    print(json.dumps(merge(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
