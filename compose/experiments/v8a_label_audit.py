"""Audit the teacher's per-sample key targets against the three-valued rule.

The teacher scores the pool-wide candidate list on every unsolved sample
(``teacher.STEP C`` loops over ``candidates``, not over the sample's own
recall), so a sample's ``recall`` -- its Top-M by key similarity -- can be a
strict subset of the experts that were actually scored.  ``tested_singles`` is
therefore the set that carries evidence, and the rule has to be *total* over it:

* the expert the teacher selected is POSITIVE;
* an expert that solved the sample but was not selected is IGNORE -- PART 9
  forbids "E5 = negative", because that label pushes an alias key away from a
  query its expert demonstrably solves;
* an expert that was scored and failed is NEGATIVE;
* a BaseOnly sample was never scored at all, so every recalled expert is IGNORE.

This script recomputes those labels from the evidence stored in each record and
diffs them against the recorded ``key_targets``.  It exists because an earlier
version built the rule over ``recall`` alone and defaulted the remainder to
NEGATIVE, which mislabelled out-of-recall solvers; the audit measures how often
that actually fired, per run, from real artifacts rather than by argument.

Usage::

    python -m compose.experiments.v8a_label_audit --task 4 \
        --run-root experiments/runs/0911_v8a_all --out label_audit.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.v8.config import (  # noqa: E402
    STATE_BASE_ONLY,
    STATE_REUSE1,
    STATE_REUSE2,
    STATE_RESIDUAL,
    TARGET_IGNORE,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
)


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_targets(record: Dict[str, Any]) -> Dict[str, str]:
    """The three-valued labels the frozen rule must produce for one record."""
    state = str(record.get("state"))
    recall = [int(value) for value in record.get("recall") or []]
    tested = [int(value) for value in record.get("tested_singles") or []]
    scored = sorted(set(recall) | set(tested))

    if state == STATE_BASE_ONLY:
        # Nothing was scored, so nothing is known to be wrong.
        return {str(expert): TARGET_IGNORE for expert in recall}

    solved = {
        int(expert)
        for expert, value in (record.get("single_values") or {}).items()
        if float(value) >= float(record.get("solved_threshold") or 1.0)
    }
    selected = {int(value) for value in record.get("selected_experts") or []}

    if state in (STATE_REUSE1, STATE_REUSE2):
        targets = {}
        for expert in scored:
            if expert in selected:
                targets[str(expert)] = TARGET_POSITIVE
            elif expert in solved:
                targets[str(expert)] = TARGET_IGNORE
            else:
                targets[str(expert)] = TARGET_NEGATIVE
        return targets
    if state == STATE_RESIDUAL:
        # Nothing solved it, so every scored expert is a legitimate negative.
        return {str(expert): TARGET_NEGATIVE for expert in scored}
    raise ValueError(f"unknown state {state!r}")


def audit_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    states: Counter = Counter()
    shrunk: Counter = Counter()          # |tested_singles \ recall|
    label_diffs: List[Dict[str, Any]] = []
    out_of_recall_solvers = 0
    out_of_recall_selected = 0
    scored_total = 0

    for record in records:
        state = str(record.get("state"))
        states[state] += 1
        recall = {int(value) for value in record.get("recall") or []}
        tested = {int(value) for value in record.get("tested_singles") or []}
        scored_total += len(tested)
        shrunk[len(tested - recall)] += 1

        threshold = float(record.get("solved_threshold") or 1.0)
        solved = {
            int(expert)
            for expert, value in (record.get("single_values") or {}).items()
            if float(value) >= threshold
        }
        if solved - recall:
            out_of_recall_solvers += 1
        selected = {int(value) for value in record.get("selected_experts") or []}
        if selected and not selected <= recall:
            out_of_recall_selected += 1

        recorded = {str(k): v for k, v in (record.get("key_targets") or {}).items()}
        expected = expected_targets(record)
        for expert in sorted(set(recorded) | set(expected), key=lambda v: int(v)):
            if recorded.get(expert) != expected.get(expert):
                label_diffs.append({
                    "sample_id": str(record.get("sample_id")),
                    "state": state,
                    "expert_id": int(expert),
                    "recorded": recorded.get(expert),
                    "expected": expected.get(expert),
                    "expert_solved": int(expert) in solved,
                    "in_recall": int(expert) in recall,
                    "selected": int(expert) in selected,
                })

    return {
        "samples": len(records),
        "states": dict(sorted(states.items())),
        "scored_experts_total": scored_total,
        "tested_minus_recall_size_distribution": {
            str(key): value for key, value in sorted(shrunk.items())
        },
        "samples_with_out_of_recall_solver": out_of_recall_solvers,
        "samples_with_out_of_recall_selection": out_of_recall_selected,
        "label_mismatches": len(label_diffs),
        "examples": label_diffs[:20],
        "conformant": not label_diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-root", required=True,
                        help="run root; the record file is <root>/task<N>/teacher_result.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result_path = Path(args.run_root) / "task{}".format(args.task) / "teacher_result.json"
    payload = _read(result_path)
    report = {
        "task": int(args.task),
        "run_root": str(args.run_root),
        "teacher_result": str(result_path),
        **audit_records(payload["records"]),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["conformant"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
