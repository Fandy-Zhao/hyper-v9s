"""Build the immutable task-level reusable historical expert artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from compose.v8.screening import aggregate_reusable_experts, teacher_router_recall


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-result", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--key-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-teacher-support", type=int, required=True)
    parser.add_argument("--min-teacher-usage-rate", type=float, required=True)
    args = parser.parse_args()
    teacher = _read(args.teacher_result)
    result = aggregate_reusable_experts(
        teacher,
        min_teacher_support=args.min_teacher_support,
        min_teacher_usage_rate=args.min_teacher_usage_rate,
    )
    query_payload = torch.load(args.query_cache, map_location="cpu")
    key_state = torch.load(args.key_state, map_location="cpu")
    result["teacher_router_agreement"] = teacher_router_recall(
        teacher, result, query_payload, key_state
    )
    result["fingerprint"] = {
        "teacher_result": str(Path(args.teacher_result).resolve()),
        "teacher_result_sha256": _sha256(args.teacher_result),
        "query_cache": str(Path(args.query_cache).resolve()),
        "query_contract_hash": query_payload.get("contract_hash"),
        "key_state": str(Path(args.key_state).resolve()),
        "key_state_sha256": _sha256(args.key_state),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
