"""Does a current-task alias key raise recall of the experts that already solve?

Causal-chain Q3, answered without training.  The V8-A runs leave the pool with
one key per expert, so the multi-key machinery never gets exercised by them; but
the teacher's verdicts say which *historical* experts already solve which
samples, and that is exactly the supervision alias keys are built from.  So this
script takes a finished run's ``teacher_result.json``, creates alias keys for the
experts the teacher found solving -- using the specified initialisation
``K_init(k, t) = Normalize(mean(q_i for i in P(k, t)))`` and nothing else, no
optimisation -- and measures the recall of a solving expert before and after.

Two honest readings of the result:

* It is an **upper bound** on what a *trained* alias key could do at this support
  size, because the centroid is computed from the same split it is evaluated on.
  The comparison to draw is therefore "is there retrieval headroom at all?", not
  "alias keys deliver +X".
* It is a **lower bound** on what the router can be made to see, because it uses
  only the frozen query and no gradient.

No GPU is required: routing is a matrix product over the frozen query cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.experiments.v8_task_run import (  # noqa: E402
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_FORMAL_ROOT,
    DEFAULT_QUERY_CACHE,
    _read_json,
    _write_json,
)
from compose.v8.cache import read_teacher_result  # noqa: E402
from compose.v8.config import V8Config  # noqa: E402
from compose.v8.key_learning import create_alias_keys  # noqa: E402
from compose.v8.pool import MultiKeyExpertPool  # noqa: E402
from compose.v8.routing import MultiKeyRouter  # noqa: E402


def _recall_at_k(
    per_expert: torch.Tensor,
    pool_expert_ids: Sequence[int],
    positives: Mapping[str, List[int]],
    sample_ids: Sequence[str],
    k: int,
) -> Dict[str, Any]:
    """Fraction of samples with >=1 solving expert inside the router's Top-K."""
    solving_samples = [s for s in sample_ids if positives.get(s)]
    if not solving_samples:
        return {"k": k, "samples": 0, "hits": 0, "recall": None}
    index = {int(expert): col for col, expert in enumerate(pool_expert_ids)}
    hits = 0
    for row, sample_id in enumerate(sample_ids):
        if sample_id not in positives:
            continue
        order = torch.argsort(per_expert[row], descending=True, stable=True).tolist()
        top = {int(pool_expert_ids[int(col)]) for col in order[:k]}
        if top & {int(value) for value in positives[sample_id]}:
            hits += 1
    return {"k": k, "samples": len(solving_samples), "hits": hits,
            "recall": hits / len(solving_samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--formal-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--query-cache", default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--teacher-cache", required=True,
                        help="canonical teacher cache dir (see "
                             "compose/experiments/v8a_export_teacher_cache.py)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    formal = Path(args.formal_root)
    checkpoint_dir = Path(args.checkpoint_dir or (formal / "task5" / "committed"))
    run_dir = Path(args.run_root) / "task{}".format(args.task)

    teacher = read_teacher_result(Path(args.teacher_cache))
    recall = _read_json(run_dir / "recall.json")
    sample_ids = sorted(str(value) for value in recall["recall"].keys()) or \
        sorted(str(record.sample_id) for record in teacher.records)
    excluded = [int(value) for value in recall.get("excluded_expert_ids", [])]

    payload = torch.load(
        Path(args.query_cache) / "query_cache" / "task{}".format(args.task)
        / "val" / "queries.pt",
        map_location="cpu",
    )
    cache_ids = [str(value) for value in payload["sample_ids"]]
    index = {sample_id: row for row, sample_id in enumerate(cache_ids)}
    queries_by_sample = {
        sample_id: payload["queries"][index[sample_id]].float() for sample_id in sample_ids
    }
    queries = torch.stack([queries_by_sample[sample_id] for sample_id in sample_ids])

    v7_state = torch.load(checkpoint_dir / "v7_keys.pt", map_location="cpu")
    manifest = _read_json(checkpoint_dir / "compose_experts.json")
    pool = MultiKeyExpertPool.load_v7_pool(v7_state, manifest, frozen=True)
    pool.validate()

    router = MultiKeyRouter(V8Config().routing)
    before = router.score_matrix(queries, pool, excluded_experts=excluded)
    origin_ids = [int(value) for value in before[1].tolist()]

    positives = teacher.positives_by_sample()
    positives = {str(k): sorted(int(v) for v in values) for k, values in positives.items()}

    created = create_alias_keys(
        teacher, pool, args.task, queries_by_sample,
        config=V8Config().pruning, trainable=False,
    )
    pool.validate()
    after = router.score_matrix(queries, pool, excluded_experts=excluded, return_keys=True)
    alias_ids = [int(value) for value in after[1].tolist()]
    if alias_ids != origin_ids:
        raise SystemExit(
            "alias keys changed the expert column order: {} != {}".format(
                alias_ids, origin_ids)
        )

    curves_before = [_recall_at_k(before[0], origin_ids, positives, sample_ids, k)
                     for k in (1, 2, 4, 8)]
    curves_after = [_recall_at_k(after[0], origin_ids, positives, sample_ids, k)
                    for k in (1, 2, 4, 8)]

    # Which key made each recallable expert reachable: the alias, or the origin?
    key_ids = after[3]
    key_scores = after[2]
    key_expert = [pool.key_records[key_id]["expert_id"] for key_id in key_ids]
    key_is_alias = [pool.key_records[key_id]["key_type"] == "task_alias" for key_id in key_ids]
    alias_similarity = {}
    for key_id, record in pool.key_records.items():
        if record["key_type"] != "task_alias":
            continue
        origin = pool.origin_key_id(record["expert_id"])
        alias_similarity[key_id] = float(F.cosine_similarity(
            pool.normalized([key_id]), pool.normalized([origin]), dim=-1).item())

    wins = {"origin": 0, "alias": 0}
    for row, sample_id in enumerate(sample_ids):
        for expert in positives.get(sample_id, []):
            columns = [i for i, value in enumerate(key_expert) if value == expert]
            if not columns:
                continue
            best = max(columns, key=lambda i: float(key_scores[row, i]))
            wins["alias" if key_is_alias[best] else "origin"] += 1

    result = {
        "task": args.task,
        "scope_run": str(args.run_root),
        "excluded_expert_ids": excluded,
        "samples": len(sample_ids),
        "samples_with_a_solving_expert": sum(1 for s in sample_ids if positives.get(s)),
        "recall_before_aliases": curves_before,
        "recall_after_aliases": curves_after,
        "delta_at_k": {
            str(b["k"]): (
                None if b["recall"] is None or a["recall"] is None
                else a["recall"] - b["recall"]
            )
            for b, a in zip(curves_before, curves_after)
        },
        "aliases": {
            "num_created": created["num_created"],
            "num_skipped": created["num_skipped"],
            "experts_with_support": created["experts_with_support"],
            "created": created["created"],
            "skipped": created["skipped"],
        },
        "support_histogram": {
            str(size): sum(1 for row in created["created"].values() if row["support"] == size)
            for size in sorted({row["support"] for row in created["created"].values()})
        },
        "alias_origin_cosine": alias_similarity,
        "winning_key": wins,
        "caveat": (
            "Upper bound on a trained alias key at this support: the centroid is "
            "built from the same split it is evaluated on.  Read it as 'is there "
            "retrieval headroom', not as a trained-key result."
        ),
    }
    _write_json(Path(args.out), result)
    print(json.dumps({
        "samples": result["samples"],
        "solving": result["samples_with_a_solving_expert"],
        "aliases_created": created["num_created"],
        "recall_before": [row["recall"] for row in curves_before],
        "recall_after": [row["recall"] for row in curves_after],
        "wins": wins,
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
