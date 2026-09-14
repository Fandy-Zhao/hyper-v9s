"""Strict equivalence comparator for RMS calibration and pruning (spec §20).

The acceleration work is only acceptable if the *result* is untouched, so this
tool is deliberately unforgiving: it reports exact-match booleans and refuses
to call anything "close".  Two runs are compared through a *reference bundle*
-- a flat JSON view of every quantity the pipeline computes -- which both the
baseline and the optimized run emit.

Bundle layout (every section optional; only sections present in both bundles
are compared)::

    {
      "label": "...",
      "rms":        {"<layer>": {"<expert>": float}},   # raw delta RMS
      "calibration":{"<layer>": {"<expert>": float}},   # kappa, post-merge
      "routing":    {"<config>": {"<sample_id>": [ids]}},
      "usage":      {"<expert>": int},
      "removal":    {"<config>/<candidate>": float},    # official metric
      "redundancy": {"<a>|<b>": float},
      "decision":   {"<candidate>": "KEEP"|"PRUNE"},
      "retained_expert_ids": [ids],
      "pruned_expert_ids":   [ids],
      "commit_manifest":     {...},
    }

Tolerances are explicit and are *recording* aids only.  The verdict is driven
by the exact-match flags: a continuous quantity may be reported as "within
noise" while still failing the run if the brief requires bit-identity.  Only
``--float-tolerance`` loosens the *comparison report*; the decision flags
(route agreement, usage agreement, removal decision agreement, final retain
set) are always exact.

Usage::

    python -m compose.experiments.compare_rms_pruning \\
        --baseline baseline_rms_t5 --optimized optimized_rms_t5 \\
        --output compare_rms.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Artifact filenames the pipeline writes, in the order they are probed when a
#: directory (rather than a bundle) is passed as a side.
RMS_STATISTICS = "rms_statistics.json"
RMS_CALIBRATION = "rms_calibration.json"
REFERENCE_BUNDLE = "reference_bundle.json"

SECTION_KEYS = (
    "rms",
    "calibration",
    "routing",
    "usage",
    "removal",
    "redundancy",
    "decision",
    "retained_expert_ids",
    "pruned_expert_ids",
    "commit_manifest",
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_side(spec: str) -> Dict[str, Any]:
    """Load a comparison side from a reference bundle or a run directory."""
    path = Path(spec)
    if path.is_file():
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError("{} is not a bundle object".format(path))
        return payload
    if not path.is_dir():
        raise FileNotFoundError(spec)
    bundle_path = path / REFERENCE_BUNDLE
    if bundle_path.is_file():
        return _load_json(bundle_path)
    bundle = {}  # type: Dict[str, Any]
    stats_path = path / RMS_STATISTICS
    if stats_path.is_file():
        state = _load_json(stats_path)
        bundle["rms"] = _rms_from_statistics(state)
        bundle["rms_provenance"] = state.get("provenance", {})
    calibration_path = path / RMS_CALIBRATION
    if calibration_path.is_file():
        payload = _load_json(calibration_path)
        bundle["calibration"] = payload.get("calibration", payload)
    for key in SECTION_KEYS:
        candidate = path / "{}.json".format(key)
        if candidate.is_file() and key not in bundle:
            bundle[key] = _load_json(candidate)
    if not bundle:
        raise ValueError(
            "{} holds none of {}, {}, or a reference bundle".format(
                path, RMS_STATISTICS, RMS_CALIBRATION
            )
        )
    return bundle


def _rms_from_statistics(state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """``{layer: {expert: delta_rms}}`` from an ``rms_statistics.json`` state.

    ``delta_rms`` is ``sqrt(sum_squares / count)``, recomputed here from the
    persisted moments so the comparison sees the raw statistic rather than a
    rounded report field.
    """
    rows = {}  # type: Dict[str, Dict[str, float]]
    for packed in state.get("entries", {}).values():
        key = packed["key"]
        moments = packed["delta"]
        count = int(moments["count"])
        if count <= 0:
            continue
        rms = math.sqrt(max(float(moments["sum_squares"]) / count, 0.0))
        rows.setdefault(key["layer_name"], {})[str(key["expert_id"])] = rms
    return rows


def _flatten_mapping(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts into ``dotted.path`` keys for a uniform diff."""
    flat = {}  # type: Dict[str, Any]
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            flat.update(_flatten_mapping(value[key], child))
    elif isinstance(value, list):
        # Lists are compared whole: order matters (retained order, routes).
        flat[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        flat[prefix] = value
    return flat


def compare_numeric(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
    label: str,
) -> Dict[str, Any]:
    """Per-key diff with absolute and relative error for a value mapping."""
    left = _flatten_mapping(baseline)
    right = _flatten_mapping(optimized)
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    rows = []  # type: List[Dict[str, Any]]
    worst_abs = 0.0
    worst_rel = 0.0
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                rows.append({"key": key, "baseline": a, "optimized": b, "equal": False})
            continue
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            if a != b:
                rows.append({"key": key, "baseline": a, "optimized": b, "equal": False})
            continue
        absolute = abs(fa - fb)
        relative = absolute / abs(fa) if fa else (0.0 if absolute == 0.0 else float("inf"))
        worst_abs = max(worst_abs, absolute)
        if math.isfinite(relative):
            worst_rel = max(worst_rel, relative)
        if fa != fb:
            rows.append(
                {
                    "key": key,
                    "baseline": fa,
                    "optimized": fb,
                    "abs_diff": absolute,
                    "rel_diff": relative,
                }
            )
    return {
        "label": label,
        "baseline_keys": len(left),
        "optimized_keys": len(right),
        "only_in_baseline": only_left,
        "only_in_optimized": only_right,
        "equal": not rows and not only_left and not only_right,
        "differing": rows,
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
    }


def compare_exact(
    baseline: Any, optimized: Any, label: str
) -> Dict[str, Any]:
    """Whole-structure equality for sets, orders, decisions and manifests."""
    same = baseline == optimized
    detail = {}  # type: Dict[str, Any]
    if not same:
        if isinstance(baseline, list) and isinstance(optimized, list):
            detail["only_in_baseline"] = sorted(set(map(str, baseline)) - set(map(str, optimized)))
            detail["only_in_optimized"] = sorted(set(map(str, optimized)) - set(map(str, baseline)))
            detail["order_equal"] = baseline == optimized
        elif isinstance(baseline, dict) and isinstance(optimized, dict):
            keys = sorted(set(baseline) | set(optimized), key=str)
            detail["differing"] = [
                {"key": str(key), "baseline": baseline.get(key), "optimized": optimized.get(key)}
                for key in keys
                if baseline.get(key) != optimized.get(key)
            ]
        else:
            detail["baseline"] = baseline
            detail["optimized"] = optimized
    return {"label": label, "equal": same, "detail": detail}


def compare_routing(
    baseline: Dict[str, Any], optimized: Dict[str, Any]
) -> Dict[str, Any]:
    """Per-sample route agreement across every configuration."""
    configs = sorted(set(baseline) | set(optimized), key=str)
    rows = []  # type: List[Dict[str, Any]]
    agree = 0
    total = 0
    for config in configs:
        left = baseline.get(config, {})
        right = optimized.get(config, {})
        for sample_id in sorted(set(left) | set(right), key=str):
            total += 1
            same = left.get(sample_id) == right.get(sample_id)
            agree += int(same)
            if not same:
                rows.append(
                    {
                        "configuration": config,
                        "sample_id": sample_id,
                        "baseline_route": left.get(sample_id),
                        "optimized_route": right.get(sample_id),
                        "agreement": False,
                    }
                )
    return {
        "samples": total,
        "agreement_count": agree,
        "agreement_rate": (agree / total) if total else 1.0,
        "disagreements": rows,
        "equal": agree == total,
    }


def compare_counts(
    baseline: Dict[str, Any], optimized: Dict[str, Any]
) -> Dict[str, Any]:
    """Usage/edit counts keyed by expert, reported as a per-expert table."""
    keys = sorted(set(baseline) | set(optimized), key=lambda value: str(value))
    rows = []  # type: List[Dict[str, Any]]
    for key in keys:
        a = baseline.get(key)
        b = optimized.get(key)
        if a != b:
            rows.append({"expert": key, "baseline_count": a, "optimized_count": b})
    return {
        "experts": len(keys),
        "differing": rows,
        "equal": not rows and set(baseline) == set(optimized),
    }


def build_report(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
    float_tolerance: float,
) -> Dict[str, Any]:
    report = {
        "baseline_label": baseline.get("label"),
        "optimized_label": optimized.get("label"),
        "sections": {},
        "flags": {},
    }  # type: Dict[str, Any]
    sections = report["sections"]
    flags = report["flags"]

    for key in ("rms", "calibration", "redundancy"):
        if key in baseline and key in optimized:
            result = compare_numeric(baseline[key], optimized[key], key)
            result["within_tolerance"] = (
                result["max_rel_diff"] <= float_tolerance
                and not result["only_in_baseline"]
                and not result["only_in_optimized"]
            )
            sections[key] = result

    if "routing" in baseline and "routing" in optimized:
        sections["routing"] = compare_routing(baseline["routing"], optimized["routing"])

    if "usage" in baseline and "usage" in optimized:
        sections["usage"] = compare_counts(baseline["usage"], optimized["usage"])

    if "removal" in baseline and "removal" in optimized:
        result = compare_numeric(baseline["removal"], optimized["removal"], "removal")
        result["within_tolerance"] = (
            result["max_rel_diff"] <= float_tolerance
            and not result["only_in_baseline"]
            and not result["only_in_optimized"]
        )
        sections["removal"] = result
        decisions = {}  # type: Dict[str, Dict[str, Any]]
        for key in sorted(set(baseline["removal"]) | set(optimized["removal"])):
            a = baseline["removal"].get(key)
            b = optimized["removal"].get(key)
            if a is None or b is None:
                decisions[key] = {
                    "baseline_metric": a,
                    "optimized_metric": b,
                    "delta": None,
                    "agreement": False,
                }
            else:
                decisions[key] = {
                    "baseline_metric": a,
                    "optimized_metric": b,
                    "delta": b - a,
                    "agreement": a == b,
                }
        sections["removal_decisions"] = decisions

    if "decision" in baseline and "decision" in optimized:
        sections["decision"] = compare_exact(
            baseline["decision"], optimized["decision"], "decision"
        )

    for key in ("retained_expert_ids", "pruned_expert_ids", "commit_manifest"):
        if key in baseline and key in optimized:
            sections[key] = compare_exact(baseline[key], optimized[key], key)

    # ---- the brief's named flags (§20, §23) --------------------------------
    flags["FINAL_RETAIN_SET_MATCH"] = sections.get(
        "retained_expert_ids", {"equal": None}
    )["equal"]
    flags["PRUNED_SET_MATCH"] = sections.get("pruned_expert_ids", {"equal": None})["equal"]
    flags["ROUTE_AGREEMENT"] = sections.get("routing", {"equal": None})["equal"]
    flags["USAGE_AGREEMENT"] = sections.get("usage", {"equal": None})["equal"]
    flags["REMOVAL_METRIC_EXACT"] = sections.get("removal", {"equal": None})["equal"]
    flags["DECISION_AGREEMENT"] = sections.get("decision", {"equal": None})["equal"]
    flags["RMS_EXACT"] = sections.get("rms", {"equal": None})["equal"]
    flags["CALIBRATION_EXACT"] = sections.get("calibration", {"equal": None})["equal"]
    flags["REDUNDANCY_EXACT"] = sections.get("redundancy", {"equal": None})["equal"]

    # The verdict: every flag that was actually *checked* must hold.  A flag is
    # checked when its section exists on both sides; an absent section is
    # reported as null rather than silently passing.
    required = [name for name, value in flags.items() if value is not None]
    unchecked = [name for name, value in flags.items() if value is None]
    report["unchecked_flags"] = unchecked
    report["checked_flags"] = required
    report["equivalence"] = all(flags[name] for name in required) if required else None
    report["verdict"] = (
        "PASS_EXACT_PRUNING_EQUIVALENCE"
        if report["equivalence"] is True
        else ("FAIL" if report["equivalence"] is False else "INCOMPLETE")
    )
    return report


def _print_human(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print("baseline : {}".format(report["baseline_label"]))
    print("optimized: {}".format(report["optimized_label"]))
    print("=" * 78)
    for name, section in report["sections"].items():
        if name in ("routing", "usage", "removal_decisions", "decision"):
            continue
        if "max_abs_diff" in section:
            print(
                "{:<14} keys={:<6} equal={!s:<5} max_abs={:.3e} max_rel={:.3e}".format(
                    name,
                    section.get("baseline_keys"),
                    section["equal"],
                    section["max_abs_diff"],
                    section["max_rel_diff"],
                )
            )
        else:
            print("{:<14} equal={}".format(name, section.get("equal")))
    routing = report["sections"].get("routing")
    if routing:
        print(
            "routing        agreement={}/{} ({:.4%})".format(
                routing["agreement_count"], routing["samples"], routing["agreement_rate"]
            )
        )
    usage = report["sections"].get("usage")
    if usage:
        print("usage          experts={} differing={}".format(usage["experts"], len(usage["differing"])))
    print("-" * 78)
    for name in sorted(report["flags"]):
        value = report["flags"][name]
        print("  {:<24} {}".format(name, "n/a" if value is None else str(value).upper()))
    print("-" * 78)
    print("VERDICT = {}".format(report["verdict"]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="bundle JSON or run directory")
    parser.add_argument("--optimized", required=True, help="bundle JSON or run directory")
    parser.add_argument("--output", default=None, help="write the full JSON report")
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=0.0,
        help=(
            "relative tolerance used only to label continuous sections as "
            "'within tolerance'; the verdict always uses exact equality, so "
            "the default of 0 keeps the report honest"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="JSON only, no table")
    args = parser.parse_args(argv)

    report = build_report(
        load_side(args.baseline), load_side(args.optimized), args.float_tolerance
    )
    if not args.quiet:
        _print_human(report)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0 if report["verdict"] == "PASS_EXACT_PRUNING_EQUIVALENCE" else (
        2 if report["verdict"] == "FAIL" else 1
    )


if __name__ == "__main__":
    sys.exit(main())
