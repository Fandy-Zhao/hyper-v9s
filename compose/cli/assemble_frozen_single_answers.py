#!/usr/bin/env python3
"""Assemble exact answers when every route selects a frozen Stage-01 Single expert."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--expert-answer", action="append", default=[], metavar="EXPERT_ID=JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.routes).read_text(encoding="utf-8"))
    if manifest.get("oracle_used") is not False or manifest.get("answer_features_used") is not False or manifest.get("task_id_lookup_used") is not False:
        raise ValueError("route manifest failed leakage audit")
    routes = {str(item["question_id"]): tuple(item["expert_ids"]) for item in manifest["routes"]}
    sources = {}
    for specification in args.expert_answer:
        expert_text, separator, path = specification.partition("=")
        if not separator:
            raise ValueError("--expert-answer must be EXPERT_ID=JSONL")
        values = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                values[str(item["question_id"])] = item
        sources[int(expert_text)] = values
    assembled = []
    for row in questions:
        question_id = str(row["question_id"])
        selected = routes.get(question_id)
        if selected is None or len(selected) != 1 or selected[0] not in sources or question_id not in sources[selected[0]]:
            return 3
        cached = sources[selected[0]][question_id]
        if cached.get("prompt") != row["text"]:
            raise ValueError("frozen answer prompt mismatch")
        assembled.append({"question_id": row["question_id"], "prompt": row["text"], "text": cached["text"],
                          "answer_id": cached["answer_id"], "model_id": cached["model_id"],
                          "metadata": {"expert_ids": list(selected), "composition_mode": "single",
                                       "reused_frozen_stage01": True, "oracle_used": False,
                                       "answer_features_used": False, "task_id_lookup_used": False}})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in assembled:
                handle.write(json.dumps(row) + "\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    elapsed = time.perf_counter() - started
    metrics = {"samples": len(assembled), "chunk": "frozen_assembly", "model_seconds_per_sample": 0.0,
               "assembly_seconds_per_sample": elapsed / max(1, len(assembled)), "peak_memory_bytes": 0,
               "reused_frozen_stage01_answers": len(assembled),
               "route_counts": {"0": 0, "1": len(assembled), "2": 0},
               "oracle_used": False, "answer_features_used": False, "task_id_lookup_used": False}
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ASSEMBLED_FROZEN_SINGLE", **metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
