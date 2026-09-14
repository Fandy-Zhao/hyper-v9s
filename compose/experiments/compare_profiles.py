"""Compare two or more ``profile_steps*.jsonl`` traces, arm by arm.

The V8 acceleration work is judged on measured step phases, so every claim in
the reports is produced by this script rather than by hand.  It reads the
JSONL written by :mod:`compose.train.profiler`, drops warm-up steps, and prints
a markdown table of per-step means with the ratio against the first arm.

Only ``kind == "step"`` rows are used; ``startup`` rows are summarised
separately because they are per-run, not per-step.

Usage::

    python -m compose.experiments.compare_profiles \
        --arm S0=/path/to/s0/profile_steps.jsonl \
        --arm S5=/path/to/s5/profile_steps.jsonl \
        --skip-steps 2 --report docs/reports/data/s5_ab.json
"""

import argparse
import json
import statistics
import sys
from typing import Dict, List, Optional, Sequence

#: Phases compared in the report, in the order they are printed.
METRICS: Sequence[str] = (
    "window_wall_time",
    "step_body_time",
    "current_expert_forward_time",
    "backward_time",
    "routing_time",
    "audit_time",
    "allreduce_time",
    "optimizer_time",
    "metrics_time",
    "data_wait_time",
    "vision_forward_time",
    "llm_forward_time",
    "compose_linear_cpu_time",
    "lora_expert_cpu_time",
    "unaccounted_step_time",
    "samples",
    "micro_steps",
    "tokens",
    "mean_seq_len",
    "text_valid_tokens",
    "text_padded_tokens",
    "text_padding_ratio",
    "peak_allocated_bytes",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="one trace; repeat for each arm, first arm is the baseline",
    )
    parser.add_argument("--skip-steps", type=int, default=2)
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


#: Phases that are disjoint from ``step_body_time`` (see
#: :class:`compose.train.profiler.TrainingProfiler`).  Mirrored here so a trace
#: can be normalised without importing the profiler.
OUTSIDE_STEP_BODY: Sequence[str] = ("optimizer_time",)


def normalize(row: Dict[str, object]) -> Dict[str, object]:
    """Derive the fields that changed meaning between profiler generations.

    Three things were fixed after the 200-step baseline had already started, and
    a trace is a measurement rather than a conclusion -- every input to all
    three is still in the row -- so they are recomputed here rather than
    trusted.  This makes traces written before and after the fix comparable
    field for field, which is the only way the arms in these reports can be read
    side by side.

    1. ``accounted_step_time`` used to add ``routing_time``, ``allreduce_time``
       and ``metrics_time``, which are *inside* ``step_body_time``.

    2. ``tokens`` used to mean two different things.  Rows written before the
       collator reported ``token_stats`` fell back to the model hook's
       post-expansion sequence length; rows written after took the collator's
       pre-expansion count.  Those differ by a factor of ~8.5 (a single
       ``<image>`` placeholder becomes 576 patch tokens inside the model), so
       the column is now *always* the post-expansion count the LLM actually
       processes: ``mean_seq_len x micro_steps``.

    3. The collator's counts are kept, but under names that say what they are --
       ``text_*`` -- because they measure text padding only.
    """
    covered = sum(
        float(row.get(name) or 0.0) for name in ("step_body_time",) + OUTSIDE_STEP_BODY
    )
    window = row.get("window_wall_time")
    row["accounted_step_time"] = round(covered, 6)
    row["unaccounted_step_time"] = round(
        0.0 if window is None else float(window) - covered, 6
    )
    samples = int(row.get("samples") or row.get("micro_steps") or 0)
    mean_seq_len = float(row.get("mean_seq_len") or 0.0)
    if samples and mean_seq_len:
        # ``mean_seq_len x samples`` is correct under every generation: the
        # oldest wrote a per-sample length and ``samples == micro_steps`` at
        # micro-batch one, the middle wrote the batch width, and the newest
        # writes a per-sample length with an explicit ``samples``.
        row["tokens"] = int(round(mean_seq_len * samples))
    if "valid_tokens" in row:
        row["text_valid_tokens"] = int(row.pop("valid_tokens"))
    if "padded_tokens" in row:
        row["text_padded_tokens"] = int(row.pop("padded_tokens"))
    if row.get("text_padded_tokens"):
        row["text_padding_ratio"] = round(
            1.0 - row["text_valid_tokens"] / row["text_padded_tokens"], 6
        )
    elif "padding_ratio" in row:
        # Traces that wrote the ratio but not its two inputs.
        row["text_padding_ratio"] = row["padding_ratio"]
    return row


def load_steps(path: str, derive: bool = True) -> List[Dict[str, object]]:
    steps: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # ``kind`` was added after the first traces were written; the
            # authoritative marker of a step row is that it carries a step
            # index, so both generations of trace are readable.
            if row.get("kind", "step") != "step" or "step" not in row:
                continue
            steps.append(normalize(row) if derive else row)
    return steps


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"steps": len(rows)}
    for name in METRICS:
        values = [
            float(row[name]) for row in rows if isinstance(row.get(name), (int, float))
        ]
        if values:
            summary[name + "_mean"] = statistics.mean(values)
            summary[name + "_median"] = statistics.median(values)
    windows = [
        float(row["window_wall_time"])
        for row in rows
        if isinstance(row.get("window_wall_time"), (int, float))
    ]
    if windows:
        summary["window_wall_time_mean"] = statistics.mean(windows)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    arms = []
    for item in args.arm:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit("--arm expects LABEL=PATH, got {!r}".format(item))
        arms.append((label, path))

    summaries = {}
    for label, path in arms:
        rows = load_steps(path)
        kept = rows[args.skip_steps :]
        summaries[label] = summarize(kept)
        summaries[label]["path"] = path
        summaries[label]["skipped_steps"] = min(args.skip_steps, len(rows))

    base_label = arms[0][0]
    base = summaries[base_label]

    others = [label for label, _ in arms[1:]]
    header = (
        "| phase | "
        + " | ".join(label for label, _ in arms)
        + " | "
        + " | ".join("{} / {}".format(label, base_label) for label in others)
        + " |"
    )
    divider = "|---" * (len(arms) + len(others) + 1) + "|"
    lines = [header, divider]
    for name in METRICS:
        key = name + "_mean"
        cells = []
        for label, _ in arms:
            value = summaries[label].get(key)
            cells.append("--" if value is None else _fmt(name, value))
        ratios = []
        for label in others:
            current = summaries[label].get(key)
            if current is None or not base.get(key):
                ratios.append("--")
            elif name == "text_padding_ratio":
                ratios.append("{:+.4f}".format(current - base[key]))
            else:
                ratios.append("{:.3f}x".format(current / base[key]))
        lines.append("| {} | {} | {} |".format(name, " | ".join(cells), " | ".join(ratios)))

    print("\n".join(lines))
    for label, _ in arms:
        print(
            "{}: {} steps measured, {:.2f} s/step".format(
                label,
                summaries[label]["steps"],
                summaries[label].get("window_wall_time_mean", 0.0),
            )
        )
    payload = {"base": base_label, "arms": summaries, "skip_steps": args.skip_steps}
    if args.report:
        import os

        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


def _fmt(name: str, value: float) -> str:
    if name.endswith("bytes"):
        return "{:.2f} GiB".format(value / (1024 ** 3))
    if name in (
        "tokens",
        "mean_seq_len",
        "samples",
        "micro_steps",
        "text_valid_tokens",
        "text_padded_tokens",
    ):
        return "{:.0f}".format(value)
    return "{:.3f}".format(value)


if __name__ == "__main__":
    sys.exit(main())
