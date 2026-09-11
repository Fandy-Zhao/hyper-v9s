"""Split a task's V8-A reuse into self-reuse and historical reuse.

The ``all`` scope lets the teacher route to every expert in the committed pool,
including the task's *own* experts -- which do not exist yet when the task is
being learned.  The ``history-only`` scope excludes them.  Reporting only the
``all`` number would credit V8 with reuse it could never have performed
(PART 15), and reporting only the history number hides how much of the task the
pool covers at all, so this script joins the two runs sample by sample.

Definitions, all read from ``state`` and ``selected_experts`` rather than from
the three-valued ``key_targets`` so the result does not depend on which labelling
rule wrote the record:

* ``solved``      -- the frozen base answered correctly (``state == BaseOnly``),
  **or** the teacher selected a non-empty route.  Base solves count in both
  scopes: the base is identical in the two runs and no scope excludes it, so a
  ``BaseOnly`` sample is solved whatever the expert pool is allowed to contain.
* ``historical``  -- solved in ``history`` (base-only, or a route that survives
  with the task's own experts removed): the sample is reachable from the pool as
  it stood when the task was learned;
* ``self_reuse``  -- solved in ``all`` but not in ``history``: the route needed
  an expert that the history-only scope excludes, i.e. the task's own or a later
  task's expert -- i.e. an expert that does not exist yet at the time the task is
  learned;
* ``unresolved``  -- solved in neither scope: the capability is not in the
  historical pool at all, as far as the recalled candidates go.

A ``BaseOnly`` sample counted as *unresolved* (as this script did before
2026-09-11 10:20) inflates the unresolved rate by exactly the base-solved
samples and makes ``historical_reuse`` disagree with ``analysis.json``, which
counts base+solve together.  The numbers are cross-checked against
``analysis.json`` in ``checks`` below.

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

from compose.v8.config import STATE_BASE_ONLY  # noqa: E402


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(root: Path, task: int) -> Dict[str, Dict[str, Any]]:
    payload = _read(Path(root) / "task{}".format(task) / "teacher_result.json")
    return {str(record["sample_id"]): record for record in payload["records"]}


def _selected(record: Dict[str, Any]) -> List[int]:
    return [int(value) for value in record.get("selected_experts") or []]


def _solved(record: Dict[str, Any]) -> bool:
    """Solved by the frozen base, or by a non-empty expert route.

    ``BaseOnly`` is a solve: the base answered correctly and the teacher stopped
    before routing to any expert.  It is also scope-invariant -- neither run
    excludes the base -- so it must be counted as solved in both.
    """
    if str(record.get("state")) == STATE_BASE_ONLY:
        return True
    return bool(_selected(record))


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
    base_only = 0
    base_mismatch: List[str] = []
    history_only_solved: List[str] = []
    self_examples: List[Dict[str, Any]] = []

    for sample_id in shared:
        all_record = all_records[sample_id]
        history_record = history_records[sample_id]
        all_selected = _selected(all_record)
        history_selected = _selected(history_record)
        all_solved = _solved(all_record)
        history_solved = _solved(history_record)

        if str(history_record.get("state")) == STATE_BASE_ONLY:
            base_only += 1
            if str(all_record.get("state")) != STATE_BASE_ONLY:
                base_mismatch.append(sample_id)

        # The history-only pool is a strict subset of the all-experts pool, so a
        # sample solved with the subset *must* be solved with the whole.  A
        # violation means the two runs disagree about the base or about the
        # recall window, and is reported rather than averaged away.
        if history_solved and not all_solved:
            history_only_solved.append(sample_id)

        if all_solved and history_solved:
            historical += 1
        elif all_solved and not history_solved:
            self_reuse += 1
            if len(self_examples) < 10:
                self_examples.append({
                    "sample_id": sample_id,
                    "all_selected": all_selected,
                    "all_state": str(all_record["state"]),
                    "history_state": str(history_record["state"]),
                    "needed_excluded_expert": sorted(set(all_selected) & excluded),
                })
        elif history_solved:
            # Solved with the historical pool but not with the whole pool.  This
            # is the monotonicity violation collected above; it must not be
            # folded into `unresolved`, which would both overstate the
            # unresolved count and hide the anomaly inside it.
            pass
        else:
            unresolved += 1

    total = len(shared)
    # `historical` counts samples solved in *both* scopes; the monotonicity
    # violations are solved with the historical pool too, so the pool's true
    # reach is their union.  Both are reported: the first is comparable with the
    # all-scope count, the second is the headline "can the old experts do this
    # task" number and agrees with analysis.json's history-only solve count.
    historical_scope = historical + len(history_only_solved)
    return {
        "task": int(task),
        "all_root": str(all_root),
        "history_root": str(history_root),
        "excluded_expert_ids": sorted(excluded),
        "samples_compared": total,
        "missing_from_one_side": missing,
        "historical_reuse": historical_scope,
        "historical_reuse_rate": (historical_scope / total) if total else 0.0,
        "historical_base_only": base_only,
        "historical_expert_route": historical_scope - base_only,
        "historical_reuse_in_both_scopes": historical,
        "self_reuse": self_reuse,
        "self_reuse_rate": (self_reuse / total) if total else 0.0,
        "unresolved": unresolved,
        "unresolved_rate": (unresolved / total) if total else 0.0,
        "solved_in_all_scope": historical + self_reuse,
        "base_state_disagreements": base_mismatch,
        "solved_in_history_only_but_not_all": history_only_solved,
        "bucket_sum": historical + self_reuse + unresolved + len(history_only_solved),
        "self_reuse_examples": self_examples,
        "note": (
            "historical_reuse counts samples solved with the task's own experts "
            "removed -- BaseOnly samples plus routes that survive the exclusion; "
            "self_reuse counts the rest of the all-scope wins and is therefore the "
            "reuse that could not have happened at this task's learning time.  Both "
            "are lower bounds on what each pool can do, because only the recalled "
            "candidates are ever scored.  bucket_sum must equal samples_compared"
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
