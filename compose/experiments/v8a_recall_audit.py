"""Recall audit over the V8-A teacher records (report section 16).

The runner's own ``teacher_expert_recall_at_k`` uses ``key_targets == positive``
as its reference set, i.e. the expert the teacher *selected*.  That is the right
denominator for "did the router put the route it chose near the top?", but it
undercounts capability: when two experts solve a sample, the second one is
labelled IGNORE by the three-valued rule, and when a pair is chosen both members
are positive but an equally good single is not.

This audit recomputes the curves over three reference sets from the same
artifacts, so "the pool cannot do this" can be told apart from "the router did
not find it" (PART 16):

* ``selected``  -- the expert(s) the teacher actually selected (the runner's
  metric, recomputed here as a cross-check);
* ``solver``    -- every expert the record shows solving the sample: the scored
  experts whose metric met ``solved_threshold``, unioned with the selected
  route.

Capability failure and retrieval failure are then separated by
``capability_present_but_outside_recall_window``: solvers that exist in the full
visible order but fall outside the Top-M window the router is allowed to see.
(No separate "oracle recall" curve is reported: the recall window *is*
``full_order[:M]``, so such a curve would be identical to the recall curve for
every k <= M, which is the only range this experiment measures.)

The solver set is read from ``single_values`` and ``solved_threshold`` --
primary evidence -- rather than from the three-valued ``key_targets``.  That
matters: for a sample whose selected expert fell outside its own Top-M recall,
the pre-fix rule emitted no POSITIVE label at all, so a label-based solver set
would silently drop the very sample it is meant to measure.  Every number
remains a lower bound on the true solver set, because only recalled experts are
ever scored; the audit says so in its own output rather than leaving the reader
to assume otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.v8.config import STATE_BASE_ONLY  # noqa: E402


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_rank(expert_ids: Sequence[int], order: Sequence[int]) -> int | None:
    """1-based rank of the highest-ranked member, or None if absent."""
    positions = [order.index(int(expert_id)) for expert_id in expert_ids
                 if int(expert_id) in order]
    return (min(positions) + 1) if positions else None


def _curve(hits: List[int | None], ks: Sequence[int]) -> Dict[str, float]:
    """Fraction of samples with a rank strictly inside each Top-k window."""
    total = len(hits)
    if not total:
        return {str(k): 0.0 for k in ks}
    return {
        str(k): sum(1 for rank in hits if rank is not None and rank <= k) / total
        for k in ks
    }


def audit(run_root: Path, task: int, ks: Sequence[int] = (1, 2, 4, 8)) -> Dict[str, Any]:
    run = Path(run_root) / "task{}".format(task)
    teacher = _read(run / "teacher_result.json")
    recall_payload = _read(run / "recall.json")
    recall_map: Dict[str, List[int]] = {
        str(key): [int(value) for value in order]
        for key, order in recall_payload["recall"].items()
    }
    full_order: Dict[str, List[int]] = {
        str(key): [int(value) for value in order]
        for key, order in recall_payload["full_order"].items()
    }
    excluded = {int(value) for value in recall_payload.get("excluded_expert_ids") or []}
    top_m = int(recall_payload.get("recall_top_m") or 0)

    selected_hits: List[int | None] = []
    solver_hits: List[int | None] = []
    capability_present_not_recalled = 0
    selected_outside = 0
    residual = 0
    base_solved = 0
    examples: List[Dict[str, Any]] = []

    for record in teacher["records"]:
        sample_id = str(record["sample_id"])
        state = str(record["state"])
        if state == STATE_BASE_ONLY:
            base_solved += 1
            continue
        threshold = float(record.get("solved_threshold") or 1.0)
        selected = [int(value) for value in record.get("selected_experts") or []]
        solvers = sorted({
            int(expert)
            for expert, value in (record.get("single_values") or {}).items()
            if float(value) >= threshold
        } | set(selected))
        if not solvers:
            residual += 1
            continue

        order = recall_map.get(sample_id, [])
        visible = full_order.get(sample_id, [])
        selected_hits.append(_best_rank(selected, order))
        solver_hits.append(_best_rank(solvers, order))

        # STEP C scores the *pool-wide* candidate union, which is wider than any
        # single sample's Top-M window.  A route that names an expert outside
        # this sample's own window could therefore be selected by the teacher
        # but never by the deployed per-sample router, so the achieved metric is
        # an upper bound by exactly these samples.  Counting them makes the
        # bound a number rather than a caveat.
        if selected and not set(selected) <= set(order):
            selected_outside += 1

        if _best_rank(solvers, order) is None:
            if _best_rank(solvers, visible) is not None:
                capability_present_not_recalled += 1
                if len(examples) < 10:
                    examples.append({
                        "sample_id": sample_id,
                        "state": state,
                        "solvers": sorted(solvers),
                        "best_oracle_rank": _best_rank(solvers, visible),
                        "recall_window": top_m,
                    })

    solved = len(solver_hits)
    return {
        "task": int(task),
        "run_root": str(run_root),
        "scope": "history-only" if recall_payload.get("history_only") else "all-experts",
        "recall_top_m": top_m,
        "excluded_expert_ids": sorted(excluded),
        "samples": len(teacher["records"]),
        "base_solved": base_solved,
        "solved_by_an_expert": solved,
        "residual_no_solver_found": residual,
        "selected_recall_at_k": _curve(selected_hits, ks),
        "solver_recall_at_k": _curve(solver_hits, ks),
        "capability_present_but_outside_recall_window": capability_present_not_recalled,
        "selected_route_outside_own_recall_window": selected_outside,
        "solved_with_a_route_inside_own_window": solved - selected_outside,
        "examples_of_retrieval_miss": examples,
        "note": (
            "solver sets are lower bounds: only recalled experts are scored, so a "
            "sample counted as residual or as a retrieval miss may still be "
            "solvable by an expert that was never recalled and never tried"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = audit(Path(args.run_root), int(args.task))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
