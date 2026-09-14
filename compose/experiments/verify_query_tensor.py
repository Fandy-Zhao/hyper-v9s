"""Numeric-equivalence audit: ``features/<split>.json`` vs ``queries.pt``.

The V8 acceleration replaces the legacy per-sample query JSON with the V7
pipeline's own ``queries.pt`` split artefact.  Both are supposed to hold the
same fixed 1536-D queries keyed by ``sample_id``; the training path may only
consume the tensor if that is *proved*, not assumed.

This script loads both, aligns them by ``sample_id`` and reports:

- id set equality, id order, and the ids missing from either side,
- per-sample bit equality (``torch.equal``), max/mean absolute difference,
  and the count of differing elements,
- dtype, shape and row-norm agreement,
- the tensor's own value fingerprint against its ``metadata.json`` sidecar.

Exit code is 0 only when the two sources agree exactly.  Usage::

    python -m compose.experiments.verify_query_tensor \
        --features-json <run>/task4/features/train.json \
        --query-tensor <run>/query_cache/task4/train/queries.pt \
        --report <out>/query_tensor_equivalence.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

from compose.v7.query_cache import load_split_cache_for_training


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-json", required=True)
    parser.add_argument("--query-tensor", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--skip-value-hash",
        action="store_true",
        help="skip the (0.8 s) tensor fingerprint recomputation",
    )
    return parser.parse_args(argv)


def _load_features(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", payload)
    ids: List[str] = []
    rows: List[List[float]] = []
    for sample_id, value in records.items():
        query = value.get("query", value) if isinstance(value, dict) else value
        ids.append(str(sample_id))
        rows.append(query)
    tensor = torch.tensor(rows, dtype=torch.float32)
    return ids, tensor


def verify(
    features_json: str, query_tensor: str, skip_value_hash: bool = False
) -> Dict[str, object]:
    started = time.perf_counter()
    # Accept either the split directory or the ``queries.pt`` inside it.  The
    # loader (``load_split_cache_for_training``) and ``--compose_v7_query_tensor``
    # both take the directory, so accepting only a file here made the two
    # disagree and reported every split as missing.
    if os.path.isdir(query_tensor):
        tensor_dir = query_tensor
    else:
        tensor_dir = os.path.dirname(query_tensor) or "."
    queries, rows, value_hash, metadata = load_split_cache_for_training(
        tensor_dir, verify_value_hash=not skip_value_hash
    )
    json_ids, json_queries = _load_features(features_json)

    tensor_ids = [None] * len(rows)
    for sample_id, row in rows.items():
        tensor_ids[row] = sample_id

    report: Dict[str, object] = {
        "features_json": features_json,
        "query_tensor": query_tensor,
        "query_tensor_value_hash": value_hash,
        "query_tensor_contract_hash": metadata.get("contract_hash"),
        "json_count": len(json_ids),
        "tensor_count": len(tensor_ids),
        "json_shape": list(json_queries.shape),
        "tensor_shape": list(queries.shape),
        "json_dtype": str(json_queries.dtype),
        "tensor_dtype": str(queries.dtype),
    }

    json_set, tensor_set = set(json_ids), set(rows)
    report["missing_from_tensor"] = sorted(json_set - tensor_set)[:20]
    report["missing_from_json"] = sorted(tensor_set - json_set)[:20]
    report["num_missing_from_tensor"] = len(json_set - tensor_set)
    report["num_missing_from_json"] = len(tensor_set - json_set)
    report["id_order_identical"] = json_ids == tensor_ids
    report["count_equal"] = len(json_ids) == len(tensor_ids)

    if not (
        report["count_equal"]
        and not report["num_missing_from_tensor"]
        and not report["num_missing_from_json"]
        and tuple(json_queries.shape) == tuple(queries.shape)
    ):
        report["verdict"] = "MISMATCH"
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
        return report

    # Truth is the JSON's per-sample dictionary order; the tensor is indexed by
    # its own sample_ids, so align explicitly rather than assuming an order.
    index = rows
    order = torch.tensor([index[sample_id] for sample_id in json_ids], dtype=torch.long)
    aligned = queries.index_select(0, order)

    difference = (aligned.float() - json_queries.float()).abs()
    report["bit_identical"] = bool(torch.equal(aligned, json_queries))
    report["max_abs_diff"] = float(difference.max().item())
    report["mean_abs_diff"] = float(difference.mean().item())
    report["num_differing_elements"] = int((aligned != json_queries).sum().item())
    report["nonzero_rows"] = int((difference.amax(dim=1) > 0).sum().item())
    report["json_all_finite"] = bool(torch.isfinite(json_queries).all().item())
    report["tensor_all_finite"] = bool(torch.isfinite(queries).all().item())
    json_norm = json_queries.norm(dim=1)
    tensor_norm = aligned.norm(dim=1)
    report["norm_max_abs_diff"] = float((json_norm - tensor_norm).abs().max().item())
    report["id_order_identical"] = json_ids == tensor_ids
    report["verdict"] = "MATCH" if report["bit_identical"] else "MISMATCH"
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    report = verify(args.features_json, args.query_tensor, args.skip_value_hash)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if report.get("verdict") == "MATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
