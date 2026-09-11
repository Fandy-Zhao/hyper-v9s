"""Split a task's V8-A reuse into self-reuse and historical reuse.

The ``all`` scope lets the teacher route to every expert in the committed pool,
including the task's *own* experts -- which do not exist yet when the task is
being learned.  The ``history-only`` scope excludes them.  Reporting only the
``all`` number would credit V8 with reuse it could never have performed
(PART 15), and reporting only the history number hides how much of the task the
pool covers at all, so this script joins the two runs sample by sample.

Definitions, all read from ``selected_experts`` rather than from the three-valued
``key_targets`` so the result does not depend on which labelling rule wrote the
record:

* ``solved``      -- the teacher selected a non-empty route for the sample;
* ``self_reuse``  -- solved in ``all``, not solved in ``history``: the route
  needed an expert that the history-only scope excludes, i.e. the task's own or
  a later task's expert;
* ``historical``  -- solved in both scopes: the route exists in the pool as it
  stood when the task was learned;
* ``unresolved``  -- not solved in either scope: the capability is not in the
  historical pool at all, as far as the recalled candidates go.

Usage::

    python -m compose.experiments.v8a_scope_gap --task 4 \
        --all-root experiments/runs/0911_v8a_all \
        --history-root experiments/runs/0911_v8a_history \
        --out scope_gap_task4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(root: Path, task: int) -> Dict[str, Dict[str, Any]]:
    payload = _read(Path(root) / "task{}".format(task) / "teacher_result.json")
    return {str(record["sample_id"]): record for record in payload["records"]}


def _selected(record: Dict[str, Any]) -> List[int]:
    return [int(value) for value in record.get("selected_experts") or []]


def _only_historical(record: Dict[str, Any], excluded: set) -> bool:
    selected = _selected(record)
    return bool(selected) and not (set(selected) & excluded)


def compare(all_root: Path, history_root: Path, task: int) -> Dict[str, Any]:
    all_records = _records(all_root, task)
    history_records = _records(history_root, task)
    recall = _read(Path(history_root) / "task{}".format(task) / "recall.json")
    excluded = {int(value) for value in recall.get("excluded_expert_ids") or []}

    shared = sorted(set(all_records) & set(history_records))
    missing = sorted(set(all_records) ^ set(history_records))

    historical = 0
    self_reuse = 0
    unresolved = 0
    history_only_solved: List[str] = []
    self_examples: List[Dict[str, Any]] = []

    for sample_id in shared:
        all_selected = _selected(all_records[sample_id])
        history_selected = _selected(history_records[sample_id])
        if history_selected and not all_selected:  # would break monotonicity
            history_only_solved.append(sample_id)
        if all_selected and history_selected:
            historical += 1
        elif all_selected and not history_selected:
            self_reuse += 1
            if len(self_examples) < 10:
                self_examples.append({
                    "sample_id": sample_id,
                    "all_selected": all_selected,
                    "all_state": str(all_records[sample_id]["state"]),
                    "history_state": str(history_records[sample_id]["state"]),
                    "needed_excluded_expert": sorted(set(all_selected) & excluded),
                })
        else:
            unresolved += 1

    total = len(shared)
    return {
        "task": int(task),
        "all_root": str(all_root),
        "history_root": str(history_root),
        "excluded_expert_ids": sorted(excluded),
        "samples_compared": total,
        "missing_from_one_side": missing,
        "historical_reuse": historical,
        "historical_reuse_rate": (historical / total) if total else 0.0,
        "self_reuse": self_reuse,
        "self_reuse_rate": (self_reuse / total) if total else 0.0,
        "unresolved": unresolved,
        "unresolved_rate": (unresolved / total) if total else 0.0,
        "solved_in_history_only_by_labels": history_only_solved,
        "self_reuse_examples": self_examples,
        "note": (
            "historical_reuse counts samples whose all-scope route survives with the "
            "task's own experts removed; self_reuse counts the rest of the all-scope "
            "wins.  Both are lower bounds on what each pool can do, because only the "
            "recalled candidates are ever scored"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--all-root", required=True)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = compare(Path(args.all_root), Path(args.history_root), int(args.task))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
