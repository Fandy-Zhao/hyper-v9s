"""Full-pool capability audit: retrieval failure or capability failure?

The teacher only ever tests the experts the router recalled.  So when a sample
ends in ``Residual`` -- no recalled single solved it and no pair did either --
two very different things could be true:

* **retrieval failure** -- some expert *outside* the recall window solves the
  sample, and V8 simply never looked at it; or
* **capability failure** -- no expert in the visible pool solves it, and the
  sample genuinely needs new capability.

The distinction decides what to fix.  Retrieval failure is a router/key problem
(the part V8's multi-key design is supposed to solve, causal-chain Q2/Q3);
capability failure is what the residual candidate expert is *for*.
``analyse()`` in the runner can only report the recall curve, because deciding
``solved`` needs a generation per (sample, expert) and testing all of them for
every sample is out of budget.  This script pays for a *declared sample* of
Residual samples instead, and tests every visible expert the teacher did not
already test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.data.records import answer_text  # noqa: E402
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
from compose.v8.generate import GenerationEngine  # noqa: E402
from compose.v8.metric_adapter import TaskMetricAdapter  # noqa: E402
from compose.v8.pool import MultiKeyExpertPool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-root", required=True,
                        help="a finished V8-A run root containing task{N}/")
    parser.add_argument("--formal-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--diagnostic-root", default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--query-cache", default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-limit", type=int, default=30,
                        help="how many Residual samples to audit (declared cost)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    formal = Path(args.formal_root)
    checkpoint_dir = Path(args.checkpoint_dir or (formal / "task5" / "committed"))
    run_dir = Path(args.run_root) / "task{}".format(args.task)
    teacher = _read_json(run_dir / "teacher_result.json")
    recall = _read_json(run_dir / "recall.json")
    visible = [int(value) for value in recall["visible_expert_ids"]]
    excluded = [int(value) for value in recall.get("excluded_expert_ids", [])]

    records = _read_json(formal / "task{}".format(args.task) / "data" / "val_full.json")
    records_by_id = {
        str(record.get("id", record.get("question_id"))): record for record in records
    }

    adapter = TaskMetricAdapter()
    spec = adapter.require_decomposable(args.task)

    residual = [record for record in teacher["records"]
                if record["state"] == "Residual"]
    audited = residual[: int(args.sample_limit)]
    if not audited:
        _write_json(Path(args.out), {"task": args.task, "residual_samples": 0,
                                     "note": "no Residual samples to audit"})
        print("no Residual samples")
        return

    # Everything the teacher already tested for these samples is skipped: the
    # audit answers "what did the recall window hide", so re-testing what was
    # already tested would only re-derive the teacher's own verdict.
    plan: Dict[str, List[int]] = {}
    for record in audited:
        sample_id = str(record["sample_id"])
        tested = {int(value) for value in record.get("tested_singles", [])}
        plan[sample_id] = [expert for expert in visible
                           if expert not in tested]

    engine = GenerationEngine(
        _load_bundle(checkpoint_dir, args.device),
        image_folder=IMAGES,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    findings: List[Dict[str, Any]] = []
    for record in audited:
        sample_id = str(record["sample_id"])
        truth = answer_text(records_by_id[sample_id])
        solved_outside: List[int] = []
        for expert in plan[sample_id]:
            answer = engine.generate_route({sample_id: [expert]}, records_by_id)[sample_id]
            value = adapter.sample_value(args.task, answer, truth)
            if value >= spec.solved_value:
                solved_outside.append(expert)
        findings.append({
            "sample_id": sample_id,
            "recall": record.get("recall"),
            "tested_single_count": len(record.get("tested_singles", [])),
            "tested_outside_window": plan[sample_id],
            "solved_outside_window": solved_outside,
            "verdict": "retrieval_failure" if solved_outside else "capability_failure",
        })

    payload = {
        "task": args.task,
        "run_root": str(args.run_root),
        "residual_samples_total": len(residual),
        "residual_samples_audited": len(audited),
        "declared_sample_limit": int(args.sample_limit),
        "visible_expert_ids": visible,
        "excluded_expert_ids": excluded,
        "retrieval_failure": sum(1 for row in findings if row["verdict"] == "retrieval_failure"),
        "capability_failure": sum(1 for row in findings if row["verdict"] == "capability_failure"),
        "generated": engine.generated_count,
        "findings": findings,
    }
    _write_json(Path(args.out), payload)
    print(json.dumps({key: payload[key] for key in
                      ("residual_samples_total", "residual_samples_audited",
                       "retrieval_failure", "capability_failure", "generated")},
                     indent=1, sort_keys=True))


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
