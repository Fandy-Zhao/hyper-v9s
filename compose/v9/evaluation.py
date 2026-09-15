"""V9-S committed-pool-only fixed-query selection manifests.

This preprocessing stage has no model forward and no answer input.  It maps the
fixed test query cache to V9 global retained-key Top-1/Top-2 selections, which
the existing generation harness consumes later on a GPU.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .inference import V9InferenceRouter
from .keys import V9KeyPool
from .data import resolve_split_query_source

class V9EvaluationError(RuntimeError):
    pass

def load_fixed_queries(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("query_mode") != "v7_fixed":
        raise V9EvaluationError("evaluation requires a v7_fixed query cache")
    records = payload.get("records", {})
    ids = sorted(str(value) for value in records)
    queries = torch.tensor([records[value]["query"] for value in ids], dtype=torch.float32)
    if queries.ndim != 2 or queries.shape[1] != 1536:
        raise V9EvaluationError("fixed queries must be [N,1536]")
    return ids, queries

def build_selection_manifest(key_state: str | Path, query_cache: str | Path | None = None, top_k: int = 2,
                             *, query_cache_manifest: str | Path | None = None,
                             task_index: int | None = None, split: str = "test",
                             query_cache_root: str | Path | None = None):
    state = torch.load(key_state, map_location="cpu", weights_only=False)
    pool = V9KeyPool.from_state(state, current_task=None)
    if pool.current_ids:
        raise V9EvaluationError("committed-pool inference refuses temporary candidates")
    if query_cache_manifest is not None:
        if query_cache is not None or task_index is None:
            raise V9EvaluationError("binary manifest routing needs task_index and no legacy query cache")
        source = resolve_split_query_source(str(query_cache_manifest), task_index=int(task_index),
                                            split=split, query_cache_root=(None if query_cache_root is None else str(query_cache_root)))
        ids, queries = list(source.sample_ids), source.queries
        query_source = source.contract_record()
    elif query_cache is not None:
        ids, queries = load_fixed_queries(query_cache)
        query_source = {"kind": "legacy_json_query_cache", "path": str(query_cache)}
    else:
        raise V9EvaluationError("formal selection requires a precomputed query manifest")
    if not pool.historical_ids:
        raise V9EvaluationError("committed pool has no deployable expert")
    router = V9InferenceRouter(pool, top_k=min(int(top_k), len(pool.historical_ids))).eval()
    route = router.route_policy(queries)
    rows = {}
    for sample_id, row in zip(ids, route["rows"]):
        rows[sample_id] = {
            "global_top2": [int(value) for value in row["expert_ids"]],
            "scores": [float(value) for value in row["scores"]],
            "key_ids": row["key_ids"],
            "source": "v9_committed_global_multi_key_fixed_query",
        }
    return {
        "method": "v9s", "query_cache": (None if query_cache is None else str(query_cache)), "key_state": str(key_state),
        "query_source": query_source, "query_encoder_calls": 0,
        "top_k": min(int(top_k), len(pool.historical_ids)), "answer_features_used": False,
        "task_id_used": False, "training_retrieval_cache_used": False, "rows": rows,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-state", required=True)
    parser.add_argument("--output", required=True)
    # Exactly one query source, and which one is not a free choice: a manifest
    # carries the sample ids the committed pool was routed against, a legacy
    # JSON cache does not.  ``--query-cache`` used to be declared twice here
    # (once ``required``), so argparse raised before either could be read and
    # this entry point could not run at all; the exclusivity that was implied
    # is now stated.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-cache")
    source.add_argument("--query-cache-manifest")
    parser.add_argument("--query-cache-root")
    parser.add_argument("--query-cache-root")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    payload = build_selection_manifest(args.key_state, args.query_cache, args.top_k,
                                       query_cache_manifest=args.query_cache_manifest, task_index=args.task_index,
                                       split=args.split, query_cache_root=args.query_cache_root)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload["rows"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)

if __name__ == "__main__":
    main()
