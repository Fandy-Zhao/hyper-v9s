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

def build_selection_manifest(key_state: str | Path, query_cache: str | Path, top_k: int = 2):
    state = torch.load(key_state, map_location="cpu", weights_only=False)
    pool = V9KeyPool.from_state(state, current_task=None)
    if pool.current_ids:
        raise V9EvaluationError("committed-pool inference refuses temporary candidates")
    ids, queries = load_fixed_queries(query_cache)
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
        "method": "v9s", "query_cache": str(query_cache), "key_state": str(key_state),
        "top_k": min(int(top_k), len(pool.historical_ids)), "answer_features_used": False,
        "task_id_used": False, "training_retrieval_cache_used": False, "rows": rows,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    payload = build_selection_manifest(args.key_state, args.query_cache, args.top_k)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload["rows"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)

if __name__ == "__main__":
    main()
