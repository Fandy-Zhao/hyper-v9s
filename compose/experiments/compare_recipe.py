"""Recipe-equivalence A/B: compare the numbers two training arms actually produced.

The V8 acceleration brief allows an execution change only if the produced
numbers do not move.  ``training_step_sec`` and friends are the *profiler's*
view; this module is the other half -- it reads the per-micro-step metric stream
that the trainer already writes (``metrics/train_steps.rankN.jsonl``) and asks,
field by field, whether the two arms are the same computation.

What is compared, and why each one is here:

``sample_ids``
    The data stream the sampler produced.  A different seed, a different
    bucketing or a resumed-vs-fresh run shows up here first; everything
    downstream is meaningless if this disagrees.
``route_types`` / ``selected_expert_ids``
    The routing decision per sample.  This is the part of V8 that a careless
    acceleration silently changes (a frozen teacher, a truncated candidate
    list, a skipped current-expert competition).  ``selected_expert_ids`` is
    the per-sample field and the only one that can be read positionally; see
    ``_row_expert_ids`` for why ``selected_current_ids`` cannot.
``total_loss`` / ``answer_loss`` / ``key_loss``
    The optimisation objective.  Compared both per micro-step and aggregated
    per optimizer step, because the aggregate is what the optimiser sees.
``selected_current_key_grad_norm`` / ``selected_current_lora_grad_norm``
    The gradient actually being applied to the trainable parameters.
``old_old_noop`` / ``gradient_window_synced`` / ``local_batch_size``
    Bookkeeping that a micro-batch change would be expected to disturb.

Two modes
---------
``--mode stream``
    Per-rank, per-micro-step alignment.  The two arms must have the same world
    size and the same accumulation, so micro-step ``k`` of rank ``r`` is the
    same unit of work on both sides.  This is the mode for an execution change
    that is supposed to be invisible -- a cache, a kernel, a removed sync.
``--mode window``
    Per-optimizer-step, rank-merged.  Grouping by the ``step`` field and
    flattening the ranks reconstructs the *global* accumulation window, which is
    the unit the recipe is actually defined over: "these 64 samples, one
    gradient, one update".

    **The effective batch is the invariant; how it is split is free.**  The
    sampler's unit is the *megabatch*,
    ``(per_device x n_gpu) x (world x GA) = per_device x world x GA``, because
    ``n_gpu`` is 1 under a distributed run -- and that equals the effective
    batch.  One optimizer step consumes exactly one megabatch.  So ``per_device``,
    world size and ``GA`` may all differ between the arms provided their product
    is held fixed; the arms then walk the same samples in the same steps.  Change
    the *effective batch* and the chopping moves, the arms read genuinely
    different samples, and comparing them measures nothing.  See
    ``docs/reports/V8_RECIPE_EQUIVALENCE_REPORT.md`` section 1.1, which pins
    this with a real two-arm trace rather than by reading the sampler.

    What a micro-batch change *does* move is the order inside the window and
    the row width, so bit-identity is not available and is not asked for.
    Routing is compared exactly, per sample, via the
    ``sample_id -> (route_type, selected_expert_ids)`` map built per window.

Verdicts
--------
``BIT_IDENTICAL``
    Every compared value is equal to the last bit.  Only an execution change
    that preserves the arithmetic order can produce this.
``EQUIVALENT``
    Losses and norms agree within ``--rel-tol``/``--abs-tol`` and *every*
    discrete decision (data stream, route types, expert ids) agrees exactly.
    A micro-batch split changes the summation order and so lands here.
``MISMATCH``
    Anything else.  The first divergent micro-step is named.

Usage::

    python -m compose.experiments.compare_recipe \
        --arm S0=/run/s0/metrics --arm S5=/run/s5/metrics \
        --accumulation 32 --world-size 2 --steps 100 \
        --report docs/reports/data/recipe_ab.json

    python -m compose.experiments.compare_recipe \
        --mode window \
        --arm S0=/run/s0/metrics --arm C=/run/c/metrics \
        --steps 200 --report docs/reports/data/s6_window.json
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: Per-micro-step fields compared exactly (no tolerance).
DISCRETE_FIELDS: Sequence[str] = (
    "sample_ids",
    "route_types",
    "selected_expert_ids",
    "selected_current_ids",
    "old_old_noop",
    "local_batch_size",
    "gradient_window_synced",
)

#: Relative tolerance for a comparison that changes the *row width* -- a
#: micro-batch split -- as opposed to the summation order alone.
#:
#: The default ``--rel-tol`` (1e-3) is calibrated for a pure summation-order
#: change, which in practice lands at ~1e-7 (S5 measures exactly 0.0).  Widening
#: a micro-batch also changes the shape of every matmul in the forward -- and,
#: under a selection plan, which samples share an expert's grouped matmul -- so
#: bf16 rounds differently.  Measured on the fixed S6 ladder: ~5e-3 window-level
#: worst case over six steps, up to ~3.4e-2 for a single row, with the whole
#: difference already present at step 0 where both arms hold identical
#: parameters.  That measurement, not a wish to see a pass, is what sets this
#: number; pass it explicitly with ``--rel-tol`` so that loosening the gate is
#: always a visible act.  See
#: ``docs/reports/V8_RECIPE_EQUIVALENCE_REPORT.md`` sections 3.3-3.4.
WIDTH_REL_TOL = 1e-2

#: Per-micro-step fields compared numerically.
NUMERIC_FIELDS: Sequence[str] = (
    "total_loss",
    "answer_loss",
    "key_loss",
    "selected_current_key_grad_norm",
    "selected_current_lora_grad_norm",
)

#: Aggregated over one optimizer step as the mean over the window's rows, which
#: is the accumulated objective ``Trainer.training_step`` backpropagates (see
#: ``_window_objective``).
STEP_FIELDS: Sequence[str] = ("total_loss", "answer_loss", "key_loss")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="metrics directory or train_steps*.jsonl file; repeat per arm",
    )
    parser.add_argument(
        "--mode",
        choices=("stream", "window"),
        default="stream",
        help="stream: per-rank micro-step alignment (needs equal world size and "
        "accumulation). window: per-optimizer-step, rank-merged, multiset "
        "comparison (world size and micro-batch may differ).",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=0,
        help="micro-steps per optimizer step; required by --mode stream",
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="optimizer steps to compare (0 = all)"
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help=(
            "relative tolerance on the numeric fields; 1e-3 is calibrated for a "
            "summation-order change. Pass WIDTH_REL_TOL (1e-2) when comparing "
            "arms of *different row widths* -- see that constant for the "
            "measured basis."
        ),
    )
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--report", default=None, help="write the full JSON here")
    parser.add_argument(
        "--markdown", default=None, help="write the summary table here"
    )
    return parser.parse_args(argv)


def _rank_of(path: str) -> int:
    match = re.search(r"\.rank(\d+)\.jsonl$", path)
    return int(match.group(1)) if match else 0


def load_arm(path: str) -> Dict[int, List[Dict[str, object]]]:
    """Return ``{rank: rows}`` for a metrics directory or a single metrics file."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "train_steps*.jsonl")))
    else:
        files = sorted(glob.glob(path))
    if not files:
        raise SystemExit("no metrics files under {!r}".format(path))
    arm: Dict[int, List[Dict[str, object]]] = {}
    for name in files:
        with open(name, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rank = _rank_of(name)
        if rank in arm:
            raise SystemExit("two files claim rank {} in {!r}".format(rank, path))
        arm[rank] = rows
    return arm


def _discrete_equal(left, right) -> bool:
    return left == right


def _flat(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def compare_numeric(
    left: Sequence[Dict[str, object]],
    right: Sequence[Dict[str, object]],
    field: str,
) -> Dict[str, object]:
    pairs: List[Tuple[int, float, float]] = []
    missing = 0
    for index, (a, b) in enumerate(zip(left, right)):
        x, y = _flat(a.get(field)), _flat(b.get(field))
        if x is None or y is None:
            missing += 1
            continue
        pairs.append((index, x, y))
    if not pairs:
        return {"field": field, "compared": 0, "missing": missing}
    diffs = [abs(x - y) for _, x, y in pairs]
    rels = [abs(x - y) / max(abs(x), abs(y), 1e-12) for _, x, y in pairs]
    worst = max(pairs, key=lambda row: abs(row[1] - row[2]))
    return {
        "field": field,
        "compared": len(pairs),
        "missing": missing,
        "bit_identical": max(diffs) == 0.0,
        "num_differing": sum(1 for d in diffs if d != 0.0),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": statistics.mean(diffs),
        "max_rel_diff": max(rels),
        "left_mean": statistics.mean(x for _, x, _ in pairs),
        "right_mean": statistics.mean(y for _, _, y in pairs),
        "worst_index": worst[0],
        "worst_left": worst[1],
        "worst_right": worst[2],
    }


def compare_discrete(
    left: Sequence[Dict[str, object]],
    right: Sequence[Dict[str, object]],
    field: str,
) -> Dict[str, object]:
    compared = 0
    differing: List[int] = []
    missing = 0
    for index, (a, b) in enumerate(zip(left, right)):
        if field not in a or field not in b:
            missing += 1
            continue
        compared += 1
        if not _discrete_equal(a[field], b[field]):
            differing.append(index)
    return {
        "field": field,
        "compared": compared,
        "missing": missing,
        "exact_match": not differing,
        "num_differing": len(differing),
        "first_divergence": differing[0] if differing else None,
        "divergence_indices": differing[:20],
    }


def _window_objective(
    rows: Sequence[Dict[str, object]], field: str
) -> Optional[float]:
    """The objective one accumulation window contributes to the optimizer.

    ``Trainer.training_step`` divides every micro-step's loss by
    ``gradient_accumulation_steps`` before backward -- it returns
    ``loss.detach() / self.args.gradient_accumulation_steps`` -- so what a rank
    optimises over one window is ``(1/GA) x sum(row values)``.  A healthy step
    writes exactly one row per micro-step, i.e. ``GA`` rows, so that expression
    is the plain *mean over the rank's rows*; DDP averages the ranks, which is
    the mean over the rank-merged window this function is handed.

    The mean is the arm-comparable quantity *without* knowing either arm's GA,
    which is exactly what a micro-batch A/B needs: the two arms are by
    construction the ones with different GAs (1x64 vs 4x16), so any statistic
    that has to be told the accumulation cannot compare them.

    It is also the only statistic that can see the failure that matters.  If the
    trainer stores a per-micro-batch *sum* for one term of the loss and a
    per-micro-batch *mean* for another, widening the micro-batch multiplies the
    summed term by the width -- the accumulation divides by a GA that shrank by
    that same width, so the amplification cancels nowhere -- while the averaged
    term stays put.  The mean over rows picks that up as a difference that grows
    with the width.  Summing the rows instead makes the summed term look exactly
    invariant (it is the window total either way) and makes the averaged one look
    like it drifts, i.e. it reports the opposite of the truth in both
    directions.  See ``docs/reports/V8_RECIPE_EQUIVALENCE_REPORT.md`` section 3.
    """
    values = [_flat(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _mean_row_width(rows: Sequence[Dict[str, object]]) -> Optional[float]:
    """How many samples the average row of this window covers.

    A per-sample normalisation of the loss cannot be written down without first
    knowing whether a row stores a per-micro-batch sum or a per-micro-batch mean
    -- dividing by the width is right for one and double-counts for the other --
    so the comparison does not guess.  It reports the width instead, and the
    ratio of the two arms' objectives against the ratio of their widths settles
    the convention from the data: a field stored as a sum scales its objective
    by exactly the width ratio, a field stored as a mean leaves it at 1.
    """
    widths = [len(row.get("sample_ids") or []) for row in rows]
    return statistics.mean(widths) if widths else None


def _scales_with_row_width(
    objective_ratio: Optional[float], width_ratio: Optional[float]
) -> Optional[bool]:
    """Whether the objective grows with the row width -- i.e. the field is a sum.

    ``None`` when the arms have the same row width, because then the two
    hypotheses are indistinguishable and the flag would not mean anything.
    """
    if objective_ratio is None or width_ratio is None:
        return None
    if abs(width_ratio - 1.0) < 1e-9:
        return None
    return abs(objective_ratio - width_ratio) <= 0.05 * width_ratio


def _mean(rows: Sequence[Dict[str, object]], field: str) -> Optional[float]:
    values = [_flat(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def optimizer_steps(
    rows: Sequence[Dict[str, object]], accumulation: int
) -> List[Dict[str, object]]:
    """Chunk the micro-step stream into optimizer steps and average the losses."""
    windows = []
    for start in range(0, len(rows) - accumulation + 1, accumulation):
        chunk = rows[start : start + accumulation]
        record = {"micro_start": start, "micro_end": start + len(chunk) - 1}
        for field in STEP_FIELDS:
            record[field] = _mean(chunk, field)
        windows.append(record)
    return windows


def compare_optimizer_steps(
    left: Sequence[Dict[str, object]],
    right: Sequence[Dict[str, object]],
    accumulation: int,
) -> Dict[str, object]:
    a = optimizer_steps(left, accumulation)
    b = optimizer_steps(right, accumulation)
    shared = min(len(a), len(b))
    out: Dict[str, object] = {"compared_steps": shared}
    for field in STEP_FIELDS:
        diffs = []
        rels = []
        worst = (0, 0.0, 0.0)
        for index in range(shared):
            x, y = a[index][field], b[index][field]
            if x is None or y is None:
                continue
            diffs.append(abs(x - y))
            rels.append(abs(x - y) / max(abs(x), abs(y), 1e-12))
            if abs(x - y) > abs(worst[1] - worst[2]):
                worst = (index, x, y)
        out[field] = {
            "bits_identical": bool(diffs) and max(diffs) == 0.0,
            "num_differing": sum(1 for d in diffs if d != 0.0),
            "max_abs_diff": max(diffs) if diffs else None,
            "max_rel_diff": max(rels) if rels else None,
            "left_mean": statistics.mean(
                v for v in (row[field] for row in a[:shared]) if v is not None
            )
            if shared
            else None,
            "right_mean": statistics.mean(
                v for v in (row[field] for row in b[:shared]) if v is not None
            )
            if shared
            else None,
            "worst_step": worst[0],
        }
    return out


def optimizer_windows(
    arm: Dict[int, List[Dict[str, object]]], steps: int = 0
) -> Dict[int, List[Dict[str, object]]]:
    """Merge every rank into the global accumulation window it contributed to.

    ``step`` is the optimizer-step index the trainer stamps on each row, so the
    window is recoverable without knowing the accumulation or the world size --
    which is exactly what makes this comparison valid when those two differ
    between the arms.
    """
    windows: Dict[int, List[Dict[str, object]]] = {}
    for rank in sorted(arm):
        for row in arm[rank]:
            index = row.get("step")
            if index is None:
                continue
            windows.setdefault(int(index), []).append(row)
    if steps:
        windows = {index: rows for index, rows in windows.items() if index < steps}
    return windows


def _row_expert_ids(row: Dict[str, object]) -> Optional[List[object]]:
    """The per-sample expert selection of one micro-step row, or ``None``.

    ``selected_expert_ids`` is the per-sample field: one list of expert ids per
    sample in the row, so it can be indexed by the sample's position.

    ``selected_current_ids`` is **not** per-sample.  The trainer writes it as
    ``sorted(... for value in current_selected)`` -- the sorted *union* of the
    current-task experts selected anywhere in the micro-batch (see
    ``compose/v7/hf_trainer.py:_write_step_metrics``).  Its length tracks the
    batch width, not the sample count: a one-sample row selects two experts and
    stores two ids.  Reading it positionally therefore mispairs samples as soon
    as the micro-batch is wider than one -- which is exactly the case window
    mode exists to compare, and it produced a confident "MISMATCH" on arms that
    agreed.  It is still honoured as a fallback, but only when it is per-sample
    shaped.
    """
    ids = row.get("sample_ids")
    if not isinstance(ids, list):
        return None
    experts = row.get("selected_expert_ids")
    if isinstance(experts, list) and len(experts) == len(ids):
        return list(experts)
    legacy = row.get("selected_current_ids")
    if isinstance(legacy, list) and len(legacy) == len(ids):
        return [value if isinstance(value, (list, tuple)) else [value] for value in legacy]
    return None


def _routing_map(rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, Tuple[object, object]], int]:
    """``sample_id -> (route_type, sorted selected expert ids)``, plus a count.

    Flattening the window this way compares the routing *decision per sample*
    rather than a histogram, so it stays exact even though the samples are
    distributed differently across ranks on the two sides.

    The second return value counts samples whose expert ids could not be
    established.  Those would otherwise compare equal to each other for the
    trivial reason that both sides are missing, so the count is surfaced rather
    than swallowed.
    """
    routing: Dict[str, Tuple[object, object]] = {}
    unverifiable = 0
    for row in rows:
        ids = row.get("sample_ids")
        routes = row.get("route_types")
        if not isinstance(ids, list) or not isinstance(routes, list):
            continue
        ids = [str(value) for value in ids]
        routes = list(routes)
        experts = _row_expert_ids(row)
        for position, sample_id in enumerate(ids):
            route = routes[position] if position < len(routes) else None
            if experts is None:
                routing[sample_id] = (route, None)
                unverifiable += 1
                continue
            expert_ids = experts[position]
            if isinstance(expert_ids, (list, tuple)):
                expert_ids = tuple(sorted(int(value) for value in expert_ids))
            routing[sample_id] = (route, expert_ids)
    return routing, unverifiable


def compare_windows(
    left_arm: Dict[int, List[Dict[str, object]]],
    right_arm: Dict[int, List[Dict[str, object]]],
    steps: int,
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, object]:
    """Compare the global accumulation window, rank-merged, step by step."""
    left = optimizer_windows(left_arm, steps)
    right = optimizer_windows(right_arm, steps)
    shared = sorted(set(left) & set(right))
    out: Dict[str, object] = {
        "mode": "window",
        "steps_left": len(left),
        "steps_right": len(right),
        "steps_compared": len(shared),
        "steps_only_left": sorted(set(left) - set(right))[:10],
        "steps_only_right": sorted(set(right) - set(left))[:10],
    }
    if not shared:
        out.update({"verdict": "MISMATCH", "reasons": ["no shared optimizer step"]})
        return out

    sample_mismatch: List[int] = []
    routing_mismatch: List[int] = []
    routing_diffs: List[Tuple[int, str]] = []
    routing_unverifiable = 0
    window_sizes: List[Tuple[int, int, int]] = []
    losses: Dict[str, List[Tuple[int, float, float]]] = {
        field: [] for field in STEP_FIELDS
    }
    row_widths: List[Tuple[int, float, float]] = []
    for index in shared:
        a, b = left[index], right[index]
        left_samples = sorted(
            str(sid) for row in a for sid in (row.get("sample_ids") or [])
        )
        right_samples = sorted(
            str(sid) for row in b for sid in (row.get("sample_ids") or [])
        )
        window_sizes.append((index, len(left_samples), len(right_samples)))
        if left_samples != right_samples:
            sample_mismatch.append(index)
            continue  # routing is meaningless when the sample sets already differ
        left_routing, left_unverifiable = _routing_map(a)
        right_routing, right_unverifiable = _routing_map(b)
        routing_unverifiable += left_unverifiable + right_unverifiable
        for sample_id in left_samples:
            if left_routing.get(sample_id) != right_routing.get(sample_id):
                routing_mismatch.append(index)
                if len(routing_diffs) < 20:
                    routing_diffs.append(
                        (
                            index,
                            "{}: {} vs {}".format(
                                sample_id,
                                left_routing.get(sample_id),
                                right_routing.get(sample_id),
                            ),
                        )
                    )
                break
        left_width, right_width = _mean_row_width(a), _mean_row_width(b)
        row_widths.append((index, left_width or 0.0, right_width or 0.0))
        for field in STEP_FIELDS:
            x, y = _window_objective(a, field), _window_objective(b, field)
            if x is not None and y is not None:
                losses[field].append((index, x, y))

    out["sample_sets_identical"] = not sample_mismatch
    out["steps_with_different_samples"] = sample_mismatch[:10]
    out["num_steps_with_different_samples"] = len(sample_mismatch)
    out["routing_identical"] = not routing_mismatch
    out["num_steps_with_different_routing"] = len(routing_mismatch)
    out["routing_divergences"] = routing_diffs
    out["routing_unverifiable_samples"] = routing_unverifiable
    out["window_size_min"] = min(size for _, size, _ in window_sizes)
    out["window_size_max"] = max(size for _, size, _ in window_sizes)

    left_width = statistics.mean(x for _, x, _ in row_widths)
    right_width = statistics.mean(y for _, _, y in row_widths)
    out["left_mean_row_width"] = left_width
    out["right_mean_row_width"] = right_width
    width_ratio = (right_width / left_width) if left_width else None

    numeric_ok = True
    bit_identical = not sample_mismatch and not routing_mismatch
    out["losses"] = {}
    for field in STEP_FIELDS:
        pairs = losses[field]
        if not pairs:
            continue
        diffs = [abs(x - y) for _, x, y in pairs]
        rels = [abs(x - y) / max(abs(x), abs(y), 1e-12) for _, x, y in pairs]
        worst = max(pairs, key=lambda row: abs(row[1] - row[2]))
        left_mean = statistics.mean(x for _, x, _ in pairs)
        right_mean = statistics.mean(y for _, _, y in pairs)
        objective_ratio = (right_mean / left_mean) if left_mean else None
        block = {
            "compared": len(pairs),
            "unit": (
                "mean over the window's rows = accumulated objective "
                "(HF divides each micro-step by gradient_accumulation_steps)"
            ),
            "bit_identical": max(diffs) == 0.0,
            "num_differing": sum(1 for d in diffs if d != 0.0),
            "max_abs_diff": max(diffs),
            "max_rel_diff": max(rels),
            "left_mean": left_mean,
            "right_mean": right_mean,
            "objective_ratio": objective_ratio,
            "row_width_ratio": width_ratio,
            "scales_with_row_width": _scales_with_row_width(
                objective_ratio, width_ratio
            ),
            "worst_step": worst[0],
            "worst_left": worst[1],
            "worst_right": worst[2],
        }
        out["losses"][field] = block
        if not block["bit_identical"]:
            bit_identical = False
        if block["max_abs_diff"] > abs_tol and block["max_rel_diff"] > rel_tol:
            numeric_ok = False

    reasons = []
    if sample_mismatch:
        reasons.append(
            "{} optimizer windows contain different samples".format(len(sample_mismatch))
        )
    if routing_mismatch:
        reasons.append(
            "the routing decision differs on {} of {} windows".format(
                len(routing_mismatch), len(shared)
            )
        )
    if not numeric_ok:
        reason = (
            "the accumulated objective (mean over the window's rows) moved "
            "beyond abs {} / rel {}".format(abs_tol, rel_tol)
        )
        scaled = [
            field
            for field in STEP_FIELDS
            if (out["losses"].get(field) or {}).get("scales_with_row_width")
        ]
        if scaled:
            reason += (
                "; {} scale(s) with the row width, so the trainer stores {} as a "
                "per-micro-batch sum and the accumulation no longer divides it "
                "back down".format(" and ".join(scaled), " and ".join(scaled))
            )
        reasons.append(reason)
    if routing_unverifiable:
        reasons.append(
            "{} sample(s) carried no per-sample expert ids, so their routing "
            "could not be compared".format(routing_unverifiable)
        )
    out["verdict"] = (
        "BIT_IDENTICAL"
        if (bit_identical and not routing_unverifiable)
        else (
            "EQUIVALENT"
            if (not sample_mismatch and not routing_mismatch and numeric_ok)
            else "MISMATCH"
        )
    )
    out["reasons"] = reasons
    return out


def compare_arms(
    left_arm: Dict[int, List[Dict[str, object]]],
    right_arm: Dict[int, List[Dict[str, object]]],
    accumulation: int,
    steps: int,
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, object]:
    if sorted(left_arm) != sorted(right_arm):
        raise SystemExit(
            "rank sets differ: {} vs {}".format(sorted(left_arm), sorted(right_arm))
        )
    limit = steps * accumulation if steps else None
    per_rank: Dict[str, object] = {}
    discrete_ok = True
    numeric_ok = True
    bit_identical = True

    for rank in sorted(left_arm):
        left, right = left_arm[rank], right_arm[rank]
        if limit is not None:
            left, right = left[:limit], right[:limit]
        entry: Dict[str, object] = {"rows_left": len(left), "rows_right": len(right)}
        if len(left) != len(right):
            entry["length_mismatch"] = True
            discrete_ok = numeric_ok = bit_identical = False
        n = min(len(left), len(right))
        entry["rows_compared"] = n

        entry["discrete"] = {}
        for field in DISCRETE_FIELDS:
            result = compare_discrete(left[:n], right[:n], field)
            entry["discrete"][field] = result
            if result["compared"] and not result["exact_match"]:
                discrete_ok = False
                bit_identical = False

        entry["numeric"] = {}
        for field in NUMERIC_FIELDS:
            result = compare_numeric(left[:n], right[:n], field)
            entry["numeric"][field] = result
            if not result.get("compared"):
                continue
            if not result["bit_identical"]:
                bit_identical = False
            if result["max_abs_diff"] > abs_tol and result["max_rel_diff"] > rel_tol:
                numeric_ok = False

        entry["optimizer_steps"] = compare_optimizer_steps(
            left[:n], right[:n], accumulation
        )
        for field in STEP_FIELDS:
            block = entry["optimizer_steps"][field]
            if not block["bits_identical"]:
                bit_identical = False
            if (
                block["max_abs_diff"] is not None
                and block["max_abs_diff"] > abs_tol
                and (block["max_rel_diff"] or 0.0) > rel_tol
            ):
                numeric_ok = False

        per_rank["rank{}".format(rank)] = entry

    verdict = (
        "BIT_IDENTICAL"
        if bit_identical
        else ("EQUIVALENT" if (discrete_ok and numeric_ok) else "MISMATCH")
    )
    reasons = []
    if not discrete_ok:
        reasons.append("at least one discrete decision (data stream / route / expert id) differs")
    if not numeric_ok:
        reasons.append(
            "a loss or gradient norm moved beyond abs {} / rel {}".format(abs_tol, rel_tol)
        )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "accumulation": accumulation,
        "steps_requested": steps,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "discrete_ok": discrete_ok,
        "numeric_ok": numeric_ok,
        "bit_identical": bit_identical,
        "ranks": per_rank,
    }


def render_markdown(label_a: str, label_b: str, result: Dict[str, object]) -> str:
    lines = [
        "### {} vs {}: `{}`".format(label_a, label_b, result["verdict"]),
        "",
        "| rank | field | compared | differing | max abs diff | max rel diff |",
        "|---|---|---|---|---|---|",
    ]
    for rank, entry in sorted(result["ranks"].items()):
        for field in NUMERIC_FIELDS:
            block = entry["numeric"][field]
            if not block.get("compared"):
                continue
            lines.append(
                "| {} | {} | {} | {} | {:.3e} | {:.3e} |".format(
                    rank,
                    field,
                    block["compared"],
                    block["num_differing"],
                    block["max_abs_diff"],
                    block["max_rel_diff"],
                )
            )
        for field in DISCRETE_FIELDS:
            block = entry["discrete"][field]
            if not block["compared"]:
                continue
            lines.append(
                "| {} | {} | {} | {} | n/a | n/a |".format(
                    rank, field, block["compared"], block["num_differing"]
                )
            )
    return "\n".join(lines) + "\n"


def render_window_markdown(
    label_a: str, label_b: str, result: Dict[str, object]
) -> str:
    lines = [
        "### {} vs {} (global optimizer window): `{}`".format(
            label_a, label_b, result["verdict"]
        ),
        "",
        "{} optimizer steps compared; each window holds {}..{} samples. "
        "Sample sets identical: **{}**. Per-sample routing identical: **{}**.".format(
            result["steps_compared"],
            result["window_size_min"],
            result["window_size_max"],
            result["sample_sets_identical"],
            result["routing_identical"],
        ),
        "",
        "| window loss | compared | differing | max abs diff | max rel diff | mean (left) | mean (right) |",
        "|---|---|---|---|---|---|---|",
    ]
    for field, block in sorted(result.get("losses", {}).items()):
        lines.append(
            "| {} | {} | {} | {:.3e} | {:.3e} | {:.6f} | {:.6f} |".format(
                field,
                block["compared"],
                block["num_differing"],
                block["max_abs_diff"],
                block["max_rel_diff"],
                block["left_mean"],
                block["right_mean"],
            )
        )
    if result["reasons"]:
        lines.append("")
        for reason in result["reasons"]:
            lines.append("* {}".format(reason))
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    arms: List[Tuple[str, str]] = []
    for item in args.arm:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit("--arm expects LABEL=PATH, got {!r}".format(item))
        arms.append((label, path))
    if len(arms) < 2:
        raise SystemExit("need at least two arms")

    if args.mode == "stream" and not args.accumulation:
        raise SystemExit("--mode stream needs --accumulation")

    loaded = [(label, load_arm(path)) for label, path in arms]
    reports: Dict[str, object] = {}
    markdown_parts = []
    for index in range(1, len(loaded)):
        label_a, arm_a = loaded[0]
        label_b, arm_b = loaded[index]
        if args.mode == "window":
            result = compare_windows(
                arm_a, arm_b, args.steps, args.abs_tol, args.rel_tol
            )
        else:
            result = compare_arms(
                arm_a, arm_b, args.accumulation, args.steps, args.abs_tol, args.rel_tol
            )
        reports["{}_vs_{}".format(label_a, label_b)] = result
        print("{} vs {}: {}".format(label_a, label_b, result["verdict"]), flush=True)
        if args.mode == "window":
            markdown_parts.append(render_window_markdown(label_a, label_b, result))
            print(
                "  {} optimizer windows compared ({}..{} samples each)".format(
                    result["steps_compared"],
                    result["window_size_min"],
                    result["window_size_max"],
                ),
                flush=True,
            )
            print(
                "  sample sets identical: {} | routing identical: {}".format(
                    result["sample_sets_identical"], result["routing_identical"]
                ),
                flush=True,
            )
            for field, block in sorted(result.get("losses", {}).items()):
                print(
                    "  {}: max abs diff {:.3e}, max rel {:.3e}, {} of {} windows differ".format(
                        field,
                        block["max_abs_diff"],
                        block["max_rel_diff"],
                        block["num_differing"],
                        block["compared"],
                    ),
                    flush=True,
                )
            for reason in result["reasons"]:
                print("  REASON: {}".format(reason), flush=True)
            continue
        markdown_parts.append(render_markdown(label_a, label_b, result))
        for entry in sorted(result["ranks"]):
            block = result["ranks"][entry]["optimizer_steps"]
            print(
                "  {} optimizer steps compared: total_loss max abs diff {:.3e}".format(
                    block["compared_steps"], block["total_loss"]["max_abs_diff"] or 0.0
                ),
                flush=True,
            )
            sample_ids = result["ranks"][entry]["discrete"]["sample_ids"]
            print(
                "  {} sample_id stream: {} of {} rows differ{}".format(
                    entry,
                    sample_ids["num_differing"],
                    sample_ids["compared"],
                    ""
                    if sample_ids["num_differing"] == 0
                    else " (first at micro-step {})".format(
                        sample_ids["first_divergence"]
                    ),
                ),
                flush=True,
            )

    payload = {
        "arms": {label: path for (label, path), _ in zip(arms, loaded)},
        "comparisons": reports,
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write("\n".join(markdown_parts))

    worst = "BIT_IDENTICAL"
    for result in reports.values():
        if result["verdict"] == "MISMATCH":
            worst = "MISMATCH"
        elif result["verdict"] == "EQUIVALENT" and worst == "BIT_IDENTICAL":
            worst = "EQUIVALENT"
    print("VERDICT: {}".format(worst))
    return 0 if worst != "MISMATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
