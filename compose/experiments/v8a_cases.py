"""Per-sample case studies from finished V8-A runs (report section 18).

Four categories, all computed from real artifacts rather than hand-picked:

* ``v7_miss_v8_fix``      -- the V7 actual route answered wrong, the V8 policy
                             answered right on the same sample.
* ``reusable_not_historical`` -- the all-experts run reused an expert for the
                             sample while the history-only run fell back to
                             Residual: the capability was there, but not in the
                             pool that exists when the task is learned.
* ``alternative_solved``   -- more than one recalled expert solved it, which is
                             the case the specification requires be recorded as
                             IGNORE rather than as a negative.
* ``residual_with_context`` -- no route solved it, and the teacher still kept a
                             historical context: the Residual state's input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.data.records import answer_text  # noqa: E402
from compose.v8.metric_adapter import TaskMetricAdapter  # noqa: E402


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _correctness(task: int, answers: List[Dict[str, Any]], records: Dict[str, Dict]) -> Dict[str, bool]:
    adapter = TaskMetricAdapter()
    spec = adapter.require_decomposable(task)
    out = {}
    for row in answers:
        sample_id = str(row.get("question_id", row.get("id")))
        if sample_id not in records:
            continue
        out[sample_id] = bool(
            adapter.sample_value(task, str(row["text"]), answer_text(records[sample_id]))
            >= spec.solved_value
        )
    return out


def load_run(root: Path, task: int) -> Dict[str, Any]:
    out = root / "task{}".format(task)
    return {
        "root": str(root),
        "out": out,
        "teacher": _read(out / "teacher_result.json"),
        "answers": _jsonl(out / "v8_policy_answers.jsonl"),
        "config": _read(out / "run_config.json"),
        "complete": _read(out / "COMPLETE.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--all-root", required=True, help="run root, all-experts scope")
    parser.add_argument("--history-root", required=True, help="run root, --history-only scope")
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = {
        str(r.get("id", r.get("question_id"))): r
        for r in _read(Path(args.formal_root) / "task{}".format(args.task) / "data" / "val_full.json")
    }
    all_run = load_run(Path(args.all_root), args.task)
    hist_run = load_run(Path(args.history_root), args.task)
    v8_all = _correctness(args.task, all_run["answers"], records)
    v8_hist = _correctness(args.task, hist_run["answers"], records)
    v7 = _correctness(
        args.task,
        _jsonl(Path(args.diagnostic_root) / "generation_accuracy" / "task{}".format(args.task)
               / "actual" / "answers.jsonl"),
        records,
    )

    hist_by_id = {str(r["sample_id"]): r for r in hist_run["teacher"]["records"]}
    all_by_id = {str(r["sample_id"]): r for r in all_run["teacher"]["records"]}

    cases: Dict[str, List[Dict[str, Any]]] = {
        "v7_miss_v8_fix": [],
        "reusable_not_historical": [],
        "alternative_solved": [],
        "residual_with_context": [],
    }
    for sample_id in sorted(v7):
        v7_ok = v7.get(sample_id)
        hist = hist_by_id.get(sample_id, {})
        seen = all_by_id.get(sample_id, {})
        positives = [int(k) for k, v in (hist.get("key_targets") or {}).items() if v == "positive"]
        summary = {
            "sample_id": sample_id,
            "question": str(records[sample_id].get("conversations", [{}])[0].get("value", ""))[:200],
            "ground_truth": answer_text(records[sample_id]),
            "v7_correct": v7_ok,
            "v8_history_correct": v8_hist.get(sample_id),
            "v8_all_correct": v8_all.get(sample_id),
            "history_state": hist.get("state"),
            "history_selected": hist.get("selected_experts"),
            "history_reason": hist.get("decision_reason"),
            "all_state": seen.get("state"),
            "all_selected": seen.get("selected_experts"),
            "solved_experts": positives,
            "residual_context": hist.get("residual_context"),
            "recall": hist.get("recall"),
        }
        if v7_ok is False and v8_hist.get(sample_id) is True:
            cases["v7_miss_v8_fix"].append(summary)
        if (hist.get("state") == "Residual"
                and seen.get("state") in ("Reuse1", "Reuse2")):
            cases["reusable_not_historical"].append(summary)
        if len(positives) > 1:
            cases["alternative_solved"].append(summary)
        if hist.get("state") == "Residual" and hist.get("residual_context"):
            cases["residual_with_context"].append(summary)

    payload = {
        "task": args.task,
        "counts": {name: len(rows) for name, rows in cases.items()},
        "v7_correct_total": sum(1 for v in v7.values() if v),
        "v8_history_correct_total": sum(1 for v in v8_hist.values() if v),
        "cases": {name: rows[:12] for name, rows in cases.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["counts"], indent=1, sort_keys=True))
    print("v7 correct {} / v8(history) correct {} of {}".format(
        payload["v7_correct_total"], payload["v8_history_correct_total"], len(v7)))


if __name__ == "__main__":
    main()
