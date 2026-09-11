"""Task 0 parity: does the V8 router + engine reproduce V7 exactly?

Task 0 is the one place where V7 and V8 are *supposed* to agree.  There is no
history to reuse, so the V8 candidate set is the four task-0 experts -- the same
four V7's own router scored when it committed them -- and V8's multi-key
aggregation degenerates to one key per expert, i.e. V7's plain inner product.
If the two differ here, they differ for a reason that has nothing to do with
multi-key routing, and every later comparison is unsafe.

Two checks, in increasing strength:

1. **Route parity.** Re-run V8's router over the task-0 queries with only the
   task-0 experts visible and compare the per-sample Top-2 against the pair
   counts V7 recorded in ``task0/selection_plan.json``.
2. **Answer parity** (``--generate``). Generate the V8 engine's answers under
   those routes and compare them, sample by sample, with the answers V7 actually
   produced in the diagnostic's ``generation_accuracy/task0/actual``.  Identical
   text under an identical route exercises prompt construction, kappa
   calibration, greedy decoding and the metric adapter at once.

Check 1 needs no GPU work beyond loading the frozen keys; check 2 costs one
generation per sample and is therefore opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.data.records import answer_text  # noqa: E402
from compose.eval.formal_ucit_eval import _score_answers  # noqa: E402
from compose.experiments.v8_task_run import (  # noqa: E402
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_FORMAL_ROOT,
    DEFAULT_QUERY_CACHE,
    IMAGES,
    MODEL,
    PROJECTOR,
    VISION,
    _read_json,
    _write_json,
)
from compose.v8.config import V8Config  # noqa: E402
from compose.v8.generate import GenerationEngine  # noqa: E402
from compose.v8.metric_adapter import TaskMetricAdapter  # noqa: E402
from compose.v8.pool import MultiKeyExpertPool  # noqa: E402
from compose.v8.routing import MultiKeyRouter  # noqa: E402


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--diagnostic-root", default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--query-cache", default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--generate", action="store_true",
                        help="also regenerate the routes and diff the answers")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    formal = Path(args.formal_root)
    checkpoint_dir = Path(args.checkpoint_dir or (formal / "task5" / "committed"))
    diagnostic = Path(args.diagnostic_root) / "task0"
    records = _read_json(formal / "task0" / "data" / "val_full.json")
    records_by_id = {
        str(record.get("id", record.get("question_id"))): record for record in records
    }
    sample_ids = sorted(records_by_id)

    payload = torch.load(
        Path(args.query_cache) / "query_cache" / "task0" / "val" / "queries.pt",
        map_location="cpu",
    )
    cache_ids = [str(value) for value in payload["sample_ids"]]
    index = {sample_id: row for row, sample_id in enumerate(cache_ids)}
    queries = torch.stack([payload["queries"][index[sample_id]].float()
                           for sample_id in sample_ids])

    v7_state = torch.load(checkpoint_dir / "v7_keys.pt", map_location="cpu")
    manifest = _read_json(checkpoint_dir / "compose_experts.json")
    pool = MultiKeyExpertPool.load_v7_pool(v7_state, manifest, frozen=True)
    pool.validate()
    expert_ids = pool.active_expert_ids()
    origin = {int(e): int(pool.expert_records[e]["origin_task"]) for e in expert_ids}

    visible = sorted(e for e in expert_ids if origin[e] == 0)
    excluded = sorted(e for e in expert_ids if origin[e] != 0)

    router = MultiKeyRouter(V8Config().routing)
    result = router(queries, pool, excluded_experts=excluded)
    v8_routes = {
        sample_id: [int(value) for value in result.expert_ids[row].tolist()]
        for row, sample_id in enumerate(sample_ids)
    }

    plan = _read_json(diagnostic / "selection_plan.json")
    v7_counts = {tuple(int(part) for part in key.split(",")): int(value)
                 for key, value in plan["actual_route_pair_counts"].items()}
    v8_counts = Counter(tuple(sorted(route)) for route in v8_routes.values())

    route_parity = {
        "visible_expert_ids": visible,
        "excluded_expert_count": len(excluded),
        "v7_pair_counts": {"{},{}".format(*pair): count for pair, count in sorted(v7_counts.items())},
        "v8_pair_counts": {"{},{}".format(*pair): count for pair, count in sorted(v8_counts.items())},
        "matching_samples": sum(
            1 for sample_id, route in v8_routes.items()
            if tuple(sorted(route)) in v7_counts
        ),
        "route_multiset_equal": dict(v7_counts) == dict(v8_counts),
        "distinct_expert_rate": float(
            sum(1 for route in v8_routes.values() if len(set(route)) == len(route))
            / max(len(v8_routes), 1)
        ),
    }

    out: Dict[str, Any] = {
        "task": 0,
        "sample_count": len(sample_ids),
        "query_contract_hash": payload.get("contract_hash"),
        "route_parity": route_parity,
    }

    if args.generate:
        engine = GenerationEngine(
            _load_bundle(checkpoint_dir, args.device),
            image_folder=IMAGES,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        answers = engine.generate_route(v8_routes, records_by_id)
        v7_answers = {
            str(row["question_id"]): str(row["text"])
            for row in _jsonl(diagnostic / "generation_accuracy" / "task0" / "actual" / "answers.jsonl")
        }
        compared = [sample_id for sample_id in sample_ids if sample_id in v7_answers]
        agree = [sample_id for sample_id in compared
                 if answers.get(sample_id, "").strip() == v7_answers[sample_id].strip()]

        adapter = TaskMetricAdapter()
        spec = adapter.require_decomposable(0)
        def score(predicted: Mapping[str, str]) -> Dict[str, Any]:
            run_dir = Path(args.out).parent / "task0_parity_score"
            run_dir.mkdir(parents=True, exist_ok=True)
            annotation = [
                {"question_id": sample_id,
                 "answer": answer_text(records_by_id[sample_id]),
                 "image": records_by_id[sample_id].get("image")}
                for sample_id in sample_ids
            ]
            annotation_path = run_dir / "questions.json"
            _write_json(annotation_path, annotation)
            prediction_path = run_dir / "predictions.jsonl"
            with prediction_path.open("w", encoding="utf-8") as handle:
                for sample_id in sample_ids:
                    handle.write(json.dumps(
                        {"question_id": sample_id, "text": predicted[sample_id]},
                        sort_keys=True, ensure_ascii=False) + "\n")
            return _score_answers(run_dir, 0, 0, prediction_path,
                                  annotation_file=str(annotation_path))

        v8_official = score(answers)
        v7_official = score(v7_answers)
        values_v8 = [adapter.sample_value(0, answers[s], answer_text(records_by_id[s]))
                     for s in sample_ids]
        out["answer_parity"] = {
            "compared_samples": len(compared),
            "identical_answers": len(agree),
            "identical_rate": len(agree) / max(len(compared), 1),
            "v8_official": v8_official,
            "v7_official": v7_official,
            "v8_metric_via_adapter": adapter.aggregate(0, values_v8),
            "metric_equal": (abs(float(v8_official["value"]) - float(v7_official["value"])) < 1e-9),
            "generated": engine.generated_count,
        }
        print("route parity: {}".format("MATCH" if route_parity["route_multiset_equal"] else "DIFFER"))
        print("answer parity: {}/{} identical".format(len(agree), len(compared)))
        print("v8 metric {} vs v7 metric {}".format(
            v8_official["value"], v7_official["value"]))
    else:
        print("route parity: {}".format("MATCH" if route_parity["route_multiset_equal"] else "DIFFER"))
        print("v7 {}\nv8 {}".format(route_parity["v7_pair_counts"], route_parity["v8_pair_counts"]))

    _write_json(Path(args.out), out)


def _load_bundle(checkpoint_dir: Path, device: str):
    from compose.eval.load_compose import load_compose_model
    return load_compose_model(
        model_path=MODEL,
        checkpoint_dir=str(checkpoint_dir),
        vision_tower=VISION,
        projector_path=PROJECTOR,
        expert_id=None,
        device=device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )



if __name__ == "__main__":
    main()
