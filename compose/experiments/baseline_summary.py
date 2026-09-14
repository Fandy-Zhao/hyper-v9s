"""Summarise a ``profile_steps*.jsonl`` trace into the P0 question list.

This is the source of every number in ``docs/reports/V8_SPEED_BASELINE.md``.  The
brief asks for a specific set of quantities; this tool answers them from the
trace rather than by hand, so a reader can re-run one command and get the table
back (the reports' section 2 and section 4 tables are both its output).

Two things are worth knowing about the mapping from the brief's question list to
what is measurable:

* Some requested quantities are *not per-step* and so cannot appear as a phase
  column: ``query_load`` is a one-shot at dataset build, and ``token_load`` is
  not a phase at all -- tokenisation happens inside the DataLoader worker and is
  hidden by the loader's prefetch, which is exactly what ``data_wait_time``
  measures.  The one-shot phases live in the ``.startup.json`` sidecar that
  :meth:`compose.train.profiler.TrainingProfiler.close` writes next to the trace;
  this tool reads it when present and says so when it is absent (the sidecar only
  exists once the run has closed cleanly).
* ``allreduce_time`` is a window *sum*, and its distribution is heavily skewed
  (a handful of windows carry a multi-second stall).  The mean alone misreports
  it, so the median, the max and the across-rank correlation are printed too.

Usage::

    python -m compose.experiments.baseline_summary \
        --trace rank0=<run>/training/profile_steps.rank0.jsonl \
        --trace rank1=<run>/training/profile_steps.rank1.jsonl \
        --skip-steps 3 \
        --util-tsv /tmp/baseline_util_full.tsv --util-gpus 3,7 \
        --json docs/reports/data/baseline_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

#: Phase columns, in the order the reports print them.
PHASES: Sequence[str] = (
    "window_wall_time",
    "step_body_time",
    "data_wait_time",
    "routing_time",
    "vision_forward_time",
    "current_expert_forward_time",
    "historical_expert_forward_time",
    "llm_forward_time",
    "backward_time",
    "allreduce_time",
    "audit_time",
    "compose_linear_cpu_time",
    "lora_expert_cpu_time",
    "metrics_time",
    "optimizer_time",
    "accounted_step_time",
    "unaccounted_step_time",
)

#: Columns whose mean is a poor summary because the distribution is skewed; the
#: median and max go in the table as well.
SKEWED = ("allreduce_time",)


def load_trace(path: Path, skip_steps: int) -> List[dict]:
    """Read a trace through :mod:`compare_profiles`'s normaliser.

    Deliberately not a second reader.  Two of the columns this tool reports --
    ``accounted_step_time`` and ``unaccounted_step_time`` -- are *derived*, and
    the values the profiler wrote into the rows come from an earlier definition
    of them that the implementation report records as wrong.  ``normalize()``
    recomputes both from the raw phase columns, which were always correct, so
    going through it is what keeps this table and the comparison tables
    describing the same run with the same arithmetic.
    """
    from compose.experiments.compare_profiles import load_steps

    return [row for row in load_steps(str(path)) if int(row["step"]) > skip_steps]


def trace_stats(rows: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for phase in PHASES:
        vals = [float(r[phase]) for r in rows if r.get(phase) is not None]
        if not vals:
            continue
        out[phase] = statistics.fmean(vals)
        if phase in SKEWED:
            out[phase + "_median"] = statistics.median(vals)
            out[phase + "_max"] = max(vals)
            out[phase + "_min"] = min(vals)

    # Cost per optimizer step, and the throughput the brief asks for.
    wall = out.get("window_wall_time")
    samples = [int(r["samples"]) for r in rows]
    tokens = [int(r["tokens"]) for r in rows]
    world = int(rows[0].get("world_size", 1))
    out["samples_per_step"] = statistics.fmean(samples)
    out["tokens_per_step"] = statistics.fmean(tokens)
    if wall:
        out["steps_per_sec"] = 1.0 / wall
        out["samples_per_sec_global"] = out["samples_per_step"] * world / wall
        out["samples_per_sec_per_gpu"] = out["samples_per_step"] / wall
        out["tokens_per_sec_global"] = out["tokens_per_step"] * world / wall
        out["tokens_per_sec_per_gpu"] = out["tokens_per_step"] / wall

    # Schedule shape: what one optimizer step actually issues.
    for col in ("model_forwards", "lora_expert_evaluations", "micro_steps",
                "compose_layers_forward", "gradient_accumulation_steps"):
        vals = [float(r[col]) for r in rows if r.get(col) is not None]
        if vals:
            out[col] = vals[0] if len(set(vals)) == 1 else statistics.fmean(vals)

    for col in ("mean_seq_len", "text_padding_ratio"):
        vals = [float(r[col]) for r in rows if r.get(col) is not None]
        if vals:
            out[col] = statistics.fmean(vals)

    # VRAM: the peak is a maximum (a mean of per-step peaks is not a peak), but
    # the mean is what a sizing decision wants.
    alloc = [int(r["peak_allocated_bytes"]) for r in rows if r.get("peak_allocated_bytes")]
    resv = [int(r["peak_reserved_bytes"]) for r in rows if r.get("peak_reserved_bytes")]
    if alloc:
        out["peak_allocated_gib_max"] = max(alloc) / 2**30
        out["peak_allocated_gib_mean"] = statistics.fmean(alloc) / 2**30
    if resv:
        out["peak_reserved_gib_max"] = max(resv) / 2**30
        out["peak_reserved_gib_mean"] = statistics.fmean(resv) / 2**30
    return out


def cross_rank(pairs: Dict[str, List[dict]]) -> Dict[str, float]:
    """Correlation of a phase across ranks, and of allreduce against the wall.

    A negative across-rank correlation is the diagnostic for a *blocking*
    collective: the rank that waits long is precisely the rank the other one did
    not wait on, so the two are anti-correlated rather than moving together.
    """
    names = sorted(pairs)
    if len(names) != 2:
        return {}
    a, b = (pairs[n] for n in names)
    n = min(len(a), len(b))
    out: Dict[str, float] = {}

    def corr(xs: Sequence[float], ys: Sequence[float]) -> float | None:
        if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
            return None
        return statistics.correlation(xs, ys)

    for phase in ("allreduce_time", "window_wall_time", "step_body_time"):
        xs = [float(a[i][phase]) for i in range(n) if a[i].get(phase) is not None]
        ys = [float(b[i][phase]) for i in range(n) if b[i].get(phase) is not None]
        c = corr(xs, ys)
        if c is not None:
            out[f"corr_{phase}_{names[0]}_vs_{names[1]}"] = c

    for name, rows in pairs.items():
        xs = [float(r["allreduce_time"]) for r in rows[:n]]
        ys = [float(r["window_wall_time"]) for r in rows[:n]]
        c = corr(xs, ys)
        if c is not None:
            out[f"corr_allreduce_vs_wall_{name}"] = c
    return out


def util_stats(tsv: Path, gpus: Sequence[int]) -> Dict[str, float]:
    """Summarise a ``nvidia-smi --query-gpu=index,utilization.gpu,memory.used`` TSV."""
    series: Dict[int, List[float]] = {g: [] for g in gpus}
    mem: Dict[int, List[float]] = {g: [] for g in gpus}
    for line in tsv.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue  # the footer line the probe appends
        try:
            idx, util, used = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if idx in series:
            series[idx].append(util)
            mem[idx].append(used)
    out: Dict[str, float] = {}
    allv: List[float] = []
    for g in gpus:
        if not series[g]:
            continue
        out[f"gpu{g}_util_mean_pct"] = statistics.fmean(series[g])
        out[f"gpu{g}_util_median_pct"] = statistics.median(series[g])
        out[f"gpu{g}_mem_used_mean_mib"] = statistics.fmean(mem[g])
        out[f"gpu{g}_n_samples"] = float(len(series[g]))
        allv.extend(series[g])
    if allv:
        out["util_mean_pct"] = statistics.fmean(allv)
        out["util_median_pct"] = statistics.median(allv)
    return out


def _fmt(value: float, unit: str = "") -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}{unit}"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.5f}{unit}"
        return f"{value:.4f}{unit}"
    return f"{value}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", required=True,
                    metavar="LABEL=PATH", help="repeat per rank")
    ap.add_argument("--skip-steps", type=int, default=3,
                    help="optimizer steps to drop as warm-up (default 3)")
    ap.add_argument("--util-tsv", type=Path, default=None)
    ap.add_argument("--util-gpus", default="3,7")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    pairs: Dict[str, List[dict]] = {}
    for spec in args.trace:
        label, _, path = spec.partition("=")
        pairs[label] = load_trace(Path(path), args.skip_steps)

    result: Dict[str, object] = {"skip_steps": args.skip_steps, "ranks": {}}
    for label, rows in pairs.items():
        stats = trace_stats(rows)
        startup = Path(str(dict(spec.split("=", 1) for spec in args.trace)[label]) + ".startup.json")
        stats["_steps_used"] = float(len(rows))
        stats["_steps_span"] = float(rows[-1]["step"] - rows[0]["step"] + 1)
        if startup.exists():
            side = json.loads(startup.read_text())
            stats["startup"] = side.get("startup", {})
            stats["startup_total_sec"] = side.get("startup_total_sec")
        else:
            stats["startup"] = None
        result["ranks"][label] = stats

    result["cross_rank"] = cross_rank(pairs)
    if args.util_tsv is not None:
        result["util"] = util_stats(args.util_tsv, [int(g) for g in args.util_gpus.split(",")])

    labels = sorted(pairs)
    print(f"steps used: {args.skip_steps + 1} .. "
          f"{pairs[labels[0]][-1]['step']} ({len(pairs[labels[0]])} steps)\n")
    print("| phase | " + " | ".join(labels) + " |")
    print("| --- | " + " | ".join("---" for _ in labels) + " |")
    keys = list(PHASES) + ["allreduce_time_median", "allreduce_time_max", "allreduce_time_min"]
    for key in keys:
        cells = []
        for label in labels:
            v = result["ranks"][label].get(key)  # type: ignore[index]
            cells.append("—" if v is None else _fmt(float(v), " s"))
        print(f"| `{key}` | " + " | ".join(cells) + " |")

    print("\nthroughput and schedule")
    print("| quantity | " + " | ".join(labels) + " |")
    print("| --- | " + " | ".join("---" for _ in labels) + " |")
    derived = [
        ("samples_per_step", ""), ("tokens_per_step", ""), ("mean_seq_len", ""),
        ("text_padding_ratio", ""), ("steps_per_sec", " /s"),
        ("samples_per_sec_per_gpu", " /s"), ("samples_per_sec_global", " /s"),
        ("tokens_per_sec_global", " /s"), ("model_forwards", " /step"),
        ("lora_expert_evaluations", " /step"), ("compose_layers_forward", " /step"),
        ("peak_allocated_gib_max", " GiB"), ("peak_allocated_gib_mean", " GiB"),
        ("peak_reserved_gib_max", " GiB"), ("peak_reserved_gib_mean", " GiB"),
    ]
    for key, unit in derived:
        cells = []
        for label in labels:
            v = result["ranks"][label].get(key)  # type: ignore[index]
            cells.append("—" if v is None else _fmt(float(v), unit))
        print(f"| `{key}` | " + " | ".join(cells) + " |")

    if result["cross_rank"]:
        print("\ncross-rank / within-rank correlations")
        for k, v in result["cross_rank"].items():  # type: ignore[union-attr]
            print(f"  {k} = {v:+.3f}")

    if "util" in result:
        print("\nGPU utilisation (nvidia-smi samples)")
        for k, v in result["util"].items():  # type: ignore[union-attr]
            unit = " %" if "pct" in k else (" MiB" if "mib" in k else "")
            print(f"  {k} = {v:.2f}{unit}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
