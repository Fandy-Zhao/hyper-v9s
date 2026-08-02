#!/usr/bin/env python3
"""Validate and summarize answer-free test routing artifacts for UCIT execution.

The expensive model evaluator consumes this emitted route manifest through the
existing Compose activation runtime; this command never accepts Oracle records.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-decisions", required=True)
    parser.add_argument("--accuracy-matrix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    routes = json.loads(Path(args.route_decisions).read_text(encoding="utf-8"))
    matrix = json.loads(Path(args.accuracy_matrix).read_text(encoding="utf-8"))
    if routes.get("answer_features_used") is not False or routes.get("oracle_used") is not False or routes.get("task_id_lookup_used") is not False:
        raise ValueError("formal UCIT routes failed answer/task/Oracle audit")
    values = matrix["accuracy_matrix"]
    if len(values) != 6 or any(len(row) != 6 for row in values): raise ValueError("full UCIT matrix must be 6x6")
    summary = {**matrix, "average_active_experts": sum(len(row["expert_ids"]) for row in routes["routes"]) / max(1, len(routes["routes"])),
               "route_rates": {str(size): sum(len(row["expert_ids"]) == size for row in routes["routes"]) / max(1, len(routes["routes"])) for size in range(3)},
               "oracle_used": False, "answer_features_used": False, "task_id_lookup_used": False}
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__": main()
