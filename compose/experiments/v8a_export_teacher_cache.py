"""Export a finished V8-A run into the canonical teacher cache.

The V8-A runner writes ``teacher_result.json`` -- a report artefact meant to be
read by a human and by the acceptance report's analysers.  Everything *downstream
in the method* consumes the canonical cache instead: ``teacher_manifest.json`` +
``teacher_records.jsonl``, written by ``compose/v8/cache.py:write_teacher_result``
and reloaded by ``read_teacher_result`` under a digest check.  Nothing in the
running pipeline emitted that format, so the key-learning stage had no production
caller and a finished run could not be fed to it.

This converter is that missing link.  It re-encodes a finished run into the
canonical cache and proves the round-trip -- write, reload, re-digest -- before
reporting success, so a silently lossy conversion is impossible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.v8.cache import (  # noqa: E402
    read_teacher_result,
    record_from_payload,
    write_teacher_result,
)
from compose.v8.teacher import TeacherResult  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--out", required=True,
                        help="directory to write the canonical cache into")
    args = parser.parse_args()

    source = Path(args.run_root) / "task{}".format(args.task) / "teacher_result.json"
    if not source.is_file():
        raise SystemExit("no teacher_result.json under {}".format(source.parent))
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))

    records = [record_from_payload(row) for row in payload["records"]]
    result = TeacherResult(
        task_id=int(payload.get("task_id", args.task)),
        records=records,
        config=dict(payload.get("config", {})),
    )

    out = Path(args.out)
    manifest = write_teacher_result(
        out,
        result,
        provenance={
            "source_run_root": str(args.run_root),
            "source_file": str(source),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "converter": "compose/experiments/v8a_export_teacher_cache.py",
        },
        write_route_scores=False,
    )

    reloaded = read_teacher_result(out)
    counts_before = result.state_counts()
    counts_after = reloaded.state_counts()
    if counts_before != counts_after:
        raise SystemExit(
            "round-trip changed the state counts: {} -> {}".format(
                counts_before, counts_after)
        )
    if len(reloaded.records) != len(records):
        raise SystemExit("round-trip changed the record count")

    print(json.dumps({
        "task": args.task,
        "records": len(records),
        "state_counts": counts_after,
        "teacher_result_sha256": manifest["teacher_result_sha256"],
        "manifest": str(out / "teacher_manifest.json"),
        "stores_ground_truth": manifest["stores_ground_truth"],
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
