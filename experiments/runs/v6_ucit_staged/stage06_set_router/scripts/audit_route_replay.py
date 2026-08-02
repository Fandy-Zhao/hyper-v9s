#!/usr/bin/env python3
"""Audit accepted route manifests and deterministic replay of changed cells."""

import argparse
import hashlib
import json
from pathlib import Path


def jsonl_by_id(path):
    with Path(path).open(encoding="utf-8") as handle:
        return {str(row["question_id"]): row for row in map(json.loads, handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-routes", type=Path, required=True)
    parser.add_argument("--new-routes", type=Path, required=True)
    parser.add_argument("--old-changed-cell-answers", type=Path, required=True)
    parser.add_argument("--new-changed-cell-answers", type=Path, required=True)
    parser.add_argument("--changed-cell", default="task6_Flickr30k.json")
    parser.add_argument("--accepted-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    total = changed = 0
    changed_cells, changed_ids, transitions = [], [], {}
    manifest_hashes = {}
    accepted_old_routes = accepted_new_routes = None
    for path in sorted(args.new_routes.glob("*.json")):
        old = json.loads((args.old_routes / path.name).read_text(encoding="utf-8"))
        new = json.loads(path.read_text(encoding="utf-8"))
        assert len(old["routes"]) == len(new["routes"]) == 3000
        differences = []
        for before, after in zip(old["routes"], new["routes"]):
            assert str(before["question_id"]) == str(after["question_id"])
            if (before["expert_ids"], before["cardinality"]) != (after["expert_ids"], after["cardinality"]):
                differences.append(str(after["question_id"]))
                transition = "{}->{}".format(before["cardinality"], after["cardinality"])
                transitions[transition] = transitions.get(transition, 0) + 1
        if differences:
            changed_cells.append({"cell": path.name, "changed": len(differences)})
            changed_ids.extend(differences)
        if path.name == args.changed_cell:
            accepted_old_routes, accepted_new_routes = old["routes"], new["routes"]
        changed += len(differences)
        total += len(new["routes"])
        assert new["router_checkpoint_sha256"] == args.accepted_checkpoint_sha256
        assert all(new[key] is False for key in ("oracle_used", "answer_features_used", "task_id_lookup_used"))
        manifest_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert total == 63000 and accepted_old_routes is not None
    old_answers = jsonl_by_id(args.old_changed_cell_answers)
    new_answers = jsonl_by_id(args.new_changed_cell_answers)
    unchanged_mismatches = []
    for before, after in zip(accepted_old_routes, accepted_new_routes):
        if (before["expert_ids"], before["cardinality"]) == (after["expert_ids"], after["cardinality"]):
            question_id = str(after["question_id"])
            if old_answers[question_id]["text"] != new_answers[question_id]["text"]:
                unchanged_mismatches.append(question_id)
    if unchanged_mismatches:
        raise RuntimeError("deterministic replay changed unchanged answer texts")
    audit = {
        "status": "PASSED", "files": 21, "samples": total,
        "changed_decisions_vs_pre_replay_checkpoint": changed,
        "changed_cells": changed_cells, "changed_question_ids": changed_ids,
        "cardinality_transitions": transitions,
        "unchanged_answer_text_mismatches_after_deterministic_replay": 0,
        "accepted_router_checkpoint_sha256": args.accepted_checkpoint_sha256,
        "oracle_used": False, "answer_features_used": False, "task_id_lookup_used": False,
        "route_manifest_sha256": manifest_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "files", "samples", "changed_decisions_vs_pre_replay_checkpoint", "cardinality_transitions")}, sort_keys=True))


if __name__ == "__main__":
    main()
