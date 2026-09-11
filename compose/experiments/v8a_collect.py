"""Collect finished V8-A runs into one table.

Each run writes ``task{N}/COMPLETE.json`` last, so a directory that has that
file is a run that finished; anything else is reported as incomplete rather
than silently averaged in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect(runs: List[tuple]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    incomplete: List[str] = []
    for label, root in runs:
        root = Path(root)
        for task_dir in sorted(root.glob("task*")):
            complete = task_dir / "COMPLETE.json"
            entry: Dict[str, Any] = {
                "run": label,
                "root": str(root),
                "task": int(task_dir.name[4:]),
            }
            if not complete.is_file():
                entry["status"] = "INCOMPLETE"
                incomplete.append(str(task_dir))
                rows.append(entry)
                continue
            analysis = _read(task_dir / "analysis.json")
            config = _read(task_dir / "run_config.json")
            done = _read(complete)
            entry.update({
                "status": "COMPLETE",
                "task_name": analysis.get("task_name"),
                "scope": "history-only" if config.get("history_only") else "all-experts",
                "samples": analysis.get("samples"),
                "v8_policy_metric": analysis.get("v8_policy_metric"),
                "v7_actual_route_metric": analysis.get("v7_actual_route_metric"),
                "v7_nll_oracle_pair_metric": analysis.get("v7_nll_oracle_pair_metric"),
                "states": analysis.get("states"),
                "policy_cardinality_histogram": analysis.get("policy_cardinality_histogram"),
                "recall_top_m": config.get("recall_top_m"),
                "teacher_recall_at_k": analysis.get("teacher_expert_recall_at_k"),
                "full_pool_recall_at_k": analysis.get("full_pool_oracle_recall_at_k"),
                "gap_closed": analysis.get("gap_closed"),
                "diagnosis": (analysis.get("diagnosis") or {}).get("case"),
                "diagnosis_interpretation": (analysis.get("diagnosis") or {}).get("interpretation"),
                "excluded_expert_ids": config.get("excluded_expert_ids"),
                "seed_verification": done.get("seed_verification", {}).get("status"),
                "duration_seconds": done.get("duration_seconds"),
                "generation": analysis.get("generation"),
            })
            rows.append(entry)
    return {"runs": rows, "incomplete": incomplete}


def markdown(rows: List[Dict[str, Any]]) -> str:
    head = ("| task | scope | samples | V8 policy | V7 actual | V7 NLL-oracle | "
            "BaseOnly | Reuse1 | Reuse2 | Residual | recall@8 | case | seed |")
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [head, sep]
    for row in rows:
        if row["status"] != "COMPLETE":
            lines.append("| {} | {} | - | INCOMPLETE | | | | | | | | | |".format(
                row["task"], row["run"]))
            continue
        states = row.get("states") or {}
        recall = (row.get("teacher_recall_at_k") or {}).get("8")
        lines.append("| {} {} | {} | {} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["task"], row.get("task_name"), row["scope"], row["samples"],
            row["v8_policy_metric"], row["v7_actual_route_metric"],
            row["v7_nll_oracle_pair_metric"],
            states.get("BaseOnly", 0), states.get("Reuse1", 0),
            states.get("Reuse2", 0), states.get("Residual", 0),
            "{:.3f}".format(recall) if recall is not None else "-",
            row.get("diagnosis"), row.get("seed_verification"),
        ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        metavar="LABEL=PATH", help="repeatable")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = []
    for item in args.run:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit("--run wants LABEL=PATH, got {!r}".format(item))
        runs.append((label, path))
    payload = collect(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    table = markdown(payload["runs"])
    (out / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    if payload["incomplete"]:
        print("incomplete: {}".format(", ".join(payload["incomplete"])))


if __name__ == "__main__":
    main()
