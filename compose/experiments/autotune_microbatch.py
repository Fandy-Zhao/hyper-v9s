"""S6: choose the micro-batch / accumulation split, holding the batch fixed.

Micro-batch width is a *scheduling* choice.  The optimisation recipe sees one
gradient over ``micro_batch x accumulation x world_size`` samples either way,
so the split may move as long as that product stays at the baseline value --
this script refuses to propose anything that breaks it.

It runs a candidate list of splits, each for a fixed number of optimizer steps,
and ranks them by the measured median step time from the profiler trace
(``compose.train.profiler``).  Peak allocated memory is reported alongside, and
an out-of-memory candidate is recorded as such rather than aborting the sweep.

The runner is supplied by the caller as a command template, because the
training command differs per task; ``{bs}`` and ``{ga}`` are substituted::

    python -m compose.experiments.autotune_microbatch \
        --template "bash /tmp/run_s5_ab.sh autotune --per_device_train_batch_size {bs} \
                    --gradient_accumulation_steps {ga}" \
        --world-size 2 --steps 6 \
        --candidates 1:32 2:16 4:8 \
        --report docs/reports/data/microbatch_autotune.json

Each candidate must write its profiler trace to
``<output_dir>/profile_steps.jsonl`` (or ``.rankN.jsonl``); the script reads
whichever exists and aggregates over ranks.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="command with {bs}/{ga}")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="where traces land; {bs}/{ga} may be used when each candidate needs "
        "its own directory",
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--effective-batch", type=int, default=64, help="frozen baseline value"
    )
    parser.add_argument("--steps", type=int, default=6, help="optimizer steps per candidate")
    parser.add_argument("--skip-steps", type=int, default=2)
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="micro_batch:gradient_accumulation_steps pairs",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args(argv)


def _candidates(values: Sequence[str]) -> List[Tuple[int, int]]:
    parsed = []
    for value in values:
        match = re.fullmatch(r"(\d+):(\d+)", value.strip())
        if not match:
            raise SystemExit("candidate must be MICRO:ACCUM, got {!r}".format(value))
        parsed.append((int(match.group(1)), int(match.group(2))))
    return parsed


def _read_steps(output_dir: str, skip: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name in sorted(os.listdir(output_dir)):
        if not (name.startswith("profile_steps") and name.endswith(".jsonl")):
            continue
        with open(os.path.join(output_dir, name), "r", encoding="utf-8") as handle:
            per_rank = [
                json.loads(line)
                for line in handle
                if line.strip() and json.loads(line).get("kind", "step") == "step"
            ]
        # Aggregate ranks step by step: the slowest rank bounds the step.
        for index in range(skip, len(per_rank)):
            rows.append(per_rank[index])
    return rows


def _summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"steps": 0}
    wall = [float(row["window_wall_time"]) for row in rows if row.get("window_wall_time")]
    peak = [int(row["peak_allocated_bytes"]) for row in rows if row.get("peak_allocated_bytes")]
    return {
        "steps": len(rows),
        "step_time_median": statistics.median(wall) if wall else None,
        "step_time_mean": statistics.mean(wall) if wall else None,
        "peak_allocated_bytes": max(peak) if peak else None,
        "samples_per_second": (
            statistics.mean(
                [
                    float(row.get("samples", 0)) / float(row["window_wall_time"])
                    for row in rows
                    if row.get("window_wall_time")
                ]
            )
            if wall
            else None
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    results = []
    for micro, accum in _candidates(args.candidates):
        effective = micro * accum * args.world_size
        record: Dict[str, object] = {
            "micro_batch": micro,
            "accumulation": accum,
            "effective_batch": effective,
        }
        if effective != args.effective_batch:
            record["status"] = "REFUSED"
            record["reason"] = (
                "effective batch {} != frozen baseline {}".format(
                    effective, args.effective_batch
                )
            )
            results.append(record)
            print("REFUSED {}x{}: {}".format(micro, accum, record["reason"]), flush=True)
            continue
        command = args.template.format(bs=micro, ga=accum)
        record["command"] = command
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                timeout=args.timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            record["returncode"] = completed.returncode
            record["tail"] = completed.stdout.decode("utf-8", "replace")[-2000:]
        except subprocess.TimeoutExpired:
            record["status"] = "TIMEOUT"
            results.append(record)
            print("TIMEOUT {}x{}".format(micro, accum), flush=True)
            continue
        record["wall_seconds"] = round(time.perf_counter() - started, 3)
        if completed.returncode != 0:
            record["status"] = (
                "OOM" if b"out of memory" in completed.stdout.lower() else "FAILED"
            )
            results.append(record)
            print(
                "{} {}x{}".format(record["status"], micro, accum),
                flush=True,
            )
            continue
        output_dir = args.output_dir.format(bs=micro, ga=accum)
        record["output_dir"] = output_dir
        summary = _summarize(_read_steps(output_dir, args.skip_steps))
        record.update(summary)
        record["status"] = "OK" if summary.get("steps") else "NO_TRACE"
        results.append(record)
        print(
            "OK {}x{}: {:.2f} s/step, peak {:.2f} GiB".format(
                micro,
                accum,
                record.get("step_time_median") or 0.0,
                (record.get("peak_allocated_bytes") or 0) / (1024 ** 3),
            ),
            flush=True,
        )

    usable = [row for row in results if row.get("status") == "OK"]
    payload = {
        "effective_batch": args.effective_batch,
        "world_size": args.world_size,
        "steps_per_candidate": args.steps,
        "candidates": results,
        "best": (
            min(usable, key=lambda row: row["step_time_median"]) if usable else None
        ),
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    print(json.dumps(payload.get("best") or {"best": None}, indent=2, default=str))
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
