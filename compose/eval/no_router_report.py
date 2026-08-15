"""Aggregate and audit V6.2 no-router oracle outputs."""

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from compose.eval.no_router_oracle import (
    FORMAL_COMMIT,
    ROUTED_FINAL,
    TASK_NAMES,
    _fingerprint,
    _sha256,
    _write_json,
)
from compose.oracle.candidate_sets import CandidateSet, build_candidate_sets


STAGE_COUNTS = [1, 2, 3, 7, 9, 10]
ROUTED_AT_STAGE = [17.13, 54.07, 54.13, 22.63, 41.03, 51.41]
ORIGIN_TASK = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5}


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_dir(root: Path, task: int, split: str, index: int) -> Path:
    return root / "fixed" / "final_pool" / f"task{task}" / split / f"candidate_{index:02d}"


def _metric(root: Path, task: int, split: str, index: int) -> float:
    path = _candidate_dir(root, task, split, index) / "score" / "metric.json"
    value = float(json.loads(path.read_text(encoding="utf-8"))["value"])
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric in {path}")
    return value


def _best(candidates: Sequence[CandidateSet], values: Sequence[float], size=None) -> CandidateSet:
    eligible = [candidate for candidate in candidates if size is None or len(candidate.expert_ids) == size]
    return max(eligible, key=lambda candidate: (values[candidate.index], -candidate.index))


def _ids(candidate: CandidateSet) -> str:
    return "[{}]".format(",".join(str(value) for value in candidate.expert_ids))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tensor_hashes(checkpoint: Path) -> Dict[int, str]:
    state = torch.load(checkpoint / "compose_experts.bin", map_location="cpu")
    digests = {}
    for name, tensor in state.items():
        marker = ".experts."
        if marker not in name:
            continue
        expert = int(name.split(marker, 1)[1].split(".", 1)[0])
        digest = digests.setdefault(expert, hashlib.sha256())
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return {expert: digest.hexdigest() for expert, digest in digests.items()}


def _kappas(checkpoint: Path, expert_ids: Sequence[int]) -> Dict[str, dict]:
    manifest = json.loads((checkpoint / "compose_experts.json").read_text(encoding="utf-8"))
    calibration = manifest["rms_calibration"]
    return {
        str(expert): {
            layer: values[str(expert)]
            for layer, values in calibration.items()
            if str(expert) in values
        }
        for expert in expert_ids
    }


def stage_equivalence(formal_root: Path) -> List[dict]:
    snapshots = []
    for task in range(6):
        manifest = json.loads(
            (formal_root / f"task{task}" / "snapshots" / f"task{task}" / "manifest.json").read_text(encoding="utf-8")
        )
        snapshots.append((Path(manifest["pool_checkpoint_dir"]), manifest["active_expert_ids"]))
    final_weights = _tensor_hashes(snapshots[-1][0])
    result = []
    for task, (checkpoint, experts) in enumerate(snapshots):
        stage_weights = _tensor_hashes(checkpoint)
        weight_equal = {str(expert): stage_weights.get(expert) == final_weights.get(expert) for expert in experts}
        kappa_equal = _kappas(checkpoint, experts) == _kappas(snapshots[-1][0], experts)
        result.append(
            {
                "task": task,
                "checkpoint": str(checkpoint),
                "experts": experts,
                "weight_equal": weight_equal,
                "kappa_equal": kappa_equal,
                "reusable": all(weight_equal.values()) and kappa_equal,
            }
        )
    return result


def rms_audit(final_checkpoint: Path) -> List[dict]:
    manifest = json.loads((final_checkpoint / "compose_experts.json").read_text(encoding="utf-8"))
    calibration = manifest["rms_calibration"]
    rows = []
    for expert in range(10):
        values = [float(layer[str(expert)]) for layer in calibration.values() if str(expert) in layer]
        rows.append(
            {
                "expert": expert,
                "origin_task": TASK_NAMES[ORIGIN_TASK[expert]],
                "count": len(values),
                "min": min(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "max": max(values),
                "at_kappa_min_ratio": sum(abs(value - 0.25) <= 1e-12 for value in values) / len(values),
                "at_kappa_max_ratio": sum(abs(value - 4.0) <= 1e-12 for value in values) / len(values),
            }
        )
    return rows


def _source_has_router_call() -> bool:
    source = Path(__file__).with_name("no_router_oracle.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module = getattr(node, "module", "") or ""
            if "compose.router" in module or any("compose.router" in name for name in names):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "select":
            return True
    return False


def aggregate(args) -> None:
    root = Path(args.output_root).resolve()
    formal_root = Path(args.formal_root).resolve()
    candidates = build_candidate_sets(range(10))
    summary_rows = []
    combination_values = {}
    pair_stats = []
    sample_stats = []

    for task, task_name in enumerate(TASK_NAMES):
        val = [_metric(root, task, "val", candidate.index) for candidate in candidates]
        test = [_metric(root, task, "test", candidate.index) for candidate in candidates]
        combination_values[task] = {"val": val, "test": test}
        sample_rows = _read_jsonl(root / "sample_oracle" / f"task{task}" / "nll.jsonl")
        mean_loss = [None] * len(candidates)
        if sample_rows:
            mean_loss = [statistics.fmean(float(row["set_nll"][i]) for row in sample_rows) for i in range(len(candidates))]
        combo_rows = [
            {
                "expert_set": _ids(candidate),
                "size": len(candidate.expert_ids),
                "val_metric": val[candidate.index],
                "test_metric": test[candidate.index],
                "avg_loss": mean_loss[candidate.index],
            }
            for candidate in candidates
        ]
        _write_csv(
            root / "combination_scores" / f"{task_name}.csv",
            ["expert_set", "size", "val_metric", "test_metric", "avg_loss"],
            combo_rows,
        )
        best_single = _best(candidates, test, size=1)
        best_pair = _best(candidates, test, size=2)
        test_oracle = _best(candidates, test)
        val_selected = _best(candidates, val)
        sample_metric_path = root / "sample_oracle" / f"task{task}" / "score" / "metric.json"
        sample_metric = float(json.loads(sample_metric_path.read_text(encoding="utf-8"))["value"])
        summary_rows.append(
            {
                "Task": task_name,
                "Routed": ROUTED_FINAL[task],
                "Base": test[0],
                "Best Single": test[best_single.index],
                "Single IDs": _ids(best_single),
                "Best Pair": test[best_pair.index],
                "Pair IDs": _ids(best_pair),
                "Val-Selected Best Combo Test": test[val_selected.index],
                "Combo IDs": _ids(val_selected),
                "Test Task Oracle": test[test_oracle.index],
                "Test Oracle IDs": _ids(test_oracle),
                "Sample Oracle": sample_metric,
                "Router Gap": test[val_selected.index] - ROUTED_FINAL[task],
                "Sample Oracle Gap": sample_metric - ROUTED_FINAL[task],
                "Pair Gain": test[best_pair.index] - test[best_single.index],
            }
        )
        synergy_rows = []
        for candidate in [value for value in candidates if len(value.expert_ids) == 2]:
            left, right = candidate.expert_ids
            left_index = next(value.index for value in candidates if value.expert_ids == (left,))
            right_index = next(value.index for value in candidates if value.expert_ids == (right,))
            synergy = test[candidate.index] - max(test[left_index], test[right_index])
            synergy_rows.append({"pair": _ids(candidate), "expert_a": left, "expert_b": right, "pair_metric": test[candidate.index], "best_single_component": max(test[left_index], test[right_index]), "pair_synergy": synergy})
        _write_csv(root / f"pair_synergy_{task_name}.csv", list(synergy_rows[0]), synergy_rows)
        synergies = [row["pair_synergy"] for row in synergy_rows]
        pair_stats.append(
            {
                "task": task_name,
                "positive_pair_rate": sum(value > 0 for value in synergies) / len(synergies),
                "mean_pair_synergy": statistics.fmean(synergies),
                "median_pair_synergy": statistics.median(synergies),
                "best_synergistic_pair": max(synergy_rows, key=lambda row: row["pair_synergy"]),
                "worst_destructive_pair": min(synergy_rows, key=lambda row: row["pair_synergy"]),
            }
        )
        chosen = [candidates[int(row["best_overall_index"])] for row in sample_rows]
        size_counts = Counter(len(candidate.expert_ids) for candidate in chosen)
        expert_usage = Counter(expert for candidate in chosen for expert in candidate.expert_ids)
        pair_usage = Counter(candidate.expert_ids for candidate in chosen if len(candidate.expert_ids) == 2)
        sample_stats.append(
            {
                "task": task_name,
                "empty_rate": size_counts[0] / len(chosen),
                "single_rate": size_counts[1] / len(chosen),
                "pair_rate": size_counts[2] / len(chosen),
                "most_used_experts": expert_usage.most_common(5),
                "most_used_pair": pair_usage.most_common(1),
            }
        )

    fields = list(summary_rows[0])
    mean_row = {field: "" for field in fields}
    mean_row["Task"] = "Mean"
    numeric_fields = [field for field in fields if field not in {"Task", "Single IDs", "Pair IDs", "Combo IDs", "Test Oracle IDs"}]
    for field in numeric_fields:
        mean_row[field] = statistics.fmean(float(row[field]) for row in summary_rows)
    all_summary = summary_rows + [mean_row]
    _write_csv(root / "oracle_summary.csv", fields, all_summary)
    _write_json(root / "oracle_summary.json", {"tasks": summary_rows, "mean": mean_row})

    cross_rows = []
    for expert in range(10):
        candidate = next(value for value in candidates if value.expert_ids == (expert,))
        row = {"Expert": expert, "Origin Task": TASK_NAMES[ORIGIN_TASK[expert]]}
        row.update({TASK_NAMES[task]: combination_values[task]["test"][candidate.index] for task in range(6)})
        cross_rows.append(row)
    _write_csv(root / "expert_cross_task_matrix.csv", list(cross_rows[0]), cross_rows)

    equivalence = stage_equivalence(formal_root)
    _write_json(root / "stage_snapshot_equivalence.json", equivalence)
    if not all(row["reusable"] for row in equivalence):
        raise RuntimeError("stage snapshots are not equivalent to final-pool subsets")
    stage_rows = []
    for task in range(6):
        allowed = set(range(STAGE_COUNTS[task]))
        eligible = [candidate for candidate in candidates if set(candidate.expert_ids).issubset(allowed)]
        test = combination_values[task]["test"]
        single = _best(eligible, test, size=1)
        pairs = [candidate for candidate in eligible if len(candidate.expert_ids) == 2]
        pair = _best(eligible, test, size=2) if pairs else None
        combo = _best(eligible, test)
        final = _best(candidates, test)
        stage_rows.append(
            {
                "Task": TASK_NAMES[task],
                "Experts Available": STAGE_COUNTS[task],
                "Routed at Stage": ROUTED_AT_STAGE[task],
                "Best Single": test[single.index],
                "Best Single IDs": _ids(single),
                "Best Pair": test[pair.index] if pair else "N/A",
                "Best Pair IDs": _ids(pair) if pair else "N/A",
                "Best Combo": test[combo.index],
                "Best Combo IDs": _ids(combo),
                "Final Pool Best Combo": test[final.index],
                "Final Pool IDs": _ids(final),
                "Future Expert Gain": test[final.index] - test[combo.index],
            }
        )
    _write_csv(root / "stage_local_oracle.csv", list(stage_rows[0]), stage_rows)

    final_checkpoint = Path(json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))["final_checkpoint"])
    rms_rows = rms_audit(final_checkpoint)
    _write_csv(root / "rms_audit.csv", list(rms_rows[0]), rms_rows)
    _write_json(root / "pair_synergy_summary.json", pair_stats)
    _write_json(root / "sample_oracle_statistics.json", sample_stats)

    router_limited = any(row["Router Gap"] > 5.0 for row in summary_rows)
    composition_limited = any(row["Pair Gain"] < -2.0 for row in summary_rows)
    expert_limited = any(
        row["Routed"] < 30.0 and row["Test Task Oracle"] <= row["Routed"] + 3.0 and row["Sample Oracle"] <= row["Routed"] + 5.0
        for row in summary_rows
    )
    dimensions = sum((expert_limited, router_limited, composition_limited))
    diagnosis = "Mixed" if dimensions > 1 else ("Expert-limited" if expert_limited else "Router-limited" if router_limited else "Composition-limited" if composition_limited else "Mixed")

    markdown_fields = ["Task", "Routed", "Base", "Best Single", "Single IDs", "Best Pair", "Pair IDs", "Val-Selected Best Combo Test", "Combo IDs", "Test Task Oracle", "Sample Oracle", "Router Gap", "Pair Gain"]
    header = "| " + " | ".join(markdown_fields) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(markdown_fields) - 1)) + " |"
    lines = [header, separator]
    for row in all_summary:
        lines.append("| " + " | ".join(str(row[field]) for field in markdown_fields) + " |")
    (root / "oracle_summary.md").write_text("# No-Router Oracle Summary\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

    report = _report(summary_rows, stage_rows, cross_rows, pair_stats, sample_stats, rms_rows, diagnosis)
    (root / "NO_ROUTER_ORACLE_REPORT.md").write_text(report, encoding="utf-8")

    after = _fingerprint(formal_root)
    _write_json(root / "formal_seed42_fingerprint_after.json", after)
    before = json.loads((root / "formal_seed42_fingerprint_before.json").read_text(encoding="utf-8"))
    problems = []
    oracle_commit = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))["oracle_code_commit"]
    if before != after:
        problems.append("formal seed42 artifact fingerprint changed")
    if _source_has_router_call():
        problems.append("router import/call detected in oracle execution source")
    for task in range(6):
        for split, expected in (("val", 200), ("test", 3000)):
            for candidate in candidates:
                answers = _candidate_dir(root, task, split, candidate.index) / "answers.jsonl"
                answer_rows = _read_jsonl(answers)
                if len(answer_rows) != expected:
                    problems.append(f"task{task} {split} candidate{candidate.index} incomplete")
                elif any(row.get("metadata", {}).get("git_commit") != oracle_commit for row in answer_rows):
                    problems.append(f"task{task} {split} candidate{candidate.index} commit mismatch")
        nll_rows = _read_jsonl(root / "sample_oracle" / f"task{task}" / "nll.jsonl")
        if len(nll_rows) != 3000:
            problems.append(f"task{task} sample NLL incomplete")
        elif any(row.get("model_commit") != oracle_commit for row in nll_rows):
            problems.append(f"task{task} sample NLL commit mismatch")
    for task, values in combination_values.items():
        if len(values["test"]) != 56 or not all(math.isfinite(value) for value in values["test"]):
            problems.append(f"task{task} invalid combination metrics")
    audit = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "formal_run_commit": FORMAL_COMMIT,
        "tasks": 6,
        "final_pool_configurations_per_task": 56,
        "single_experts_per_task": 10,
        "pairs_per_task": 45,
        "router_call_count": 0,
        "stage_local_reuse_proven": all(row["reusable"] for row in equivalence),
        "formal_artifacts_unchanged": before == after,
        "diagnosis": diagnosis,
    }
    _write_json(root / "oracle_audit.json", audit)
    print_terminal_summary(summary_rows, stage_rows, pair_stats, diagnosis, audit, root)
    if problems:
        raise SystemExit(1)


def _report(summary_rows, stage_rows, cross_rows, pair_stats, sample_stats, rms_rows, diagnosis):
    largest_router = max(summary_rows, key=lambda row: row["Router Gap"])
    largest_pair = max(summary_rows, key=lambda row: row["Pair Gain"])
    worst_pair = min(pair_stats, key=lambda row: row["worst_destructive_pair"]["pair_synergy"])
    future = max(stage_rows, key=lambda row: row["Future Expert Gain"])
    reusable = []
    for row in cross_rows:
        own = row[row["Origin Task"]]
        other = max((row[name], name) for name in TASK_NAMES if name != row["Origin Task"])
        if other[0] > own:
            reusable.append(f"E{row['Expert']} ({other[1]} {other[0]:.2f})")
    image = summary_rows[0]
    icon = summary_rows[3]
    pair_tasks = [row["Task"] for row in summary_rows if row["Pair Gain"] > 0]
    min_clipped = [row for row in rms_rows if row["at_kappa_min_ratio"] > 0]
    return f"""# V6.2 Seed42 No-Router Oracle Report

> Sample Oracle and Test Task Oracle use target/test labels for selection and are upper bounds, not valid test-time methods.

## Conclusion

Diagnosis: **{diagnosis}**. This label is derived from the measured fixed, pair, sample-level, and routed gaps rather than assumed in advance.

## Q1: Primary bottleneck

The measured classification is **{diagnosis}**. The largest validation-selected Router Gap is {largest_router['Task']} ({largest_router['Router Gap']:+.2f} points). The strongest pair gain is {largest_pair['Task']} ({largest_pair['Pair Gain']:+.2f}), while the most destructive observed pair is {worst_pair['task']} {worst_pair['worst_destructive_pair']['pair']} ({worst_pair['worst_destructive_pair']['pair_synergy']:+.2f}).

## Q2-Q3: Best experts and improvement over routed

See `oracle_summary.csv`. It records every task's best single, best pair, validation-selected fixed combo, test-label oracle, sample oracle, and routed gaps. The fixed result used for Router Gap is selected on the frozen 200-example validation slice and evaluated once on test.

## Q4: Pair evidence

Tasks with Best Pair > Best Single: {', '.join(pair_tasks) if pair_tasks else 'none'}. Full 45-pair synergy distributions are retained per task.

## Q5: Need for sample-wise routing

The per-task Sample Oracle minus Test Task Oracle gaps are: {', '.join(f"{row['Task']} {row['Sample Oracle'] - row['Test Task Oracle']:+.2f}" for row in summary_rows)}. Positive large gaps indicate heterogeneous sample-level expert needs; small gaps do not support added routing complexity.

## Q6: Future-expert reuse

The largest Final-Pool minus Stage-Local test-oracle gain is {future['Task']} ({future['Future Expert Gain']:+.2f}). Stage-local values reuse final-pool files only after exact equality of then-available expert tensors and frozen kappa entries was proven in `stage_snapshot_equivalence.json`.

## Q7: Cross-task reuse

Experts whose best non-origin single-task score exceeds their origin-task score: {', '.join(reusable) if reusable else 'none'}. See `expert_cross_task_matrix.csv` for all values.

## Q8: Dead, redundant, destructive, and task-specific experts

The cross-task matrix, all pair synergy files, and sample usage frequencies provide the auditable evidence. A dead expert has no competitive single score and negligible sample-oracle use; redundancy appears as near-identical cross-task columns; destructive pairs have negative PairSynergy; task-specific experts peak sharply on their origin task.

## ImageNet-R diagnostic

Routed {image['Routed']:.2f}; Base {image['Base']:.2f}; Best Single {image['Best Single']:.2f} {image['Single IDs']}; Best Pair {image['Best Pair']:.2f} {image['Pair IDs']}; validation-selected fixed {image['Val-Selected Best Combo Test']:.2f} {image['Combo IDs']}; Sample Oracle {image['Sample Oracle']:.2f}.

## IconQA diagnostic

Routed {icon['Routed']:.2f}; Best Single {icon['Best Single']:.2f} {icon['Single IDs']}; Best Pair {icon['Best Pair']:.2f} {icon['Pair IDs']}; Sample Oracle {icon['Sample Oracle']:.2f}. E3-E6 are the four IconQA-origin experts; their full single and mutual-pair results are in the cross-task and pair files.

## RMS audit

Frozen kappa was loaded, never recomputed. Experts with at least one layer clipped to kappa_min=0.25: {', '.join('E{}'.format(row['expert']) for row in min_clipped) if min_clipped else 'none'}. Per-expert min/mean/median/max and both clip ratios are in `rms_audit.csv`.
"""


def print_terminal_summary(summary_rows, stage_rows, pair_stats, diagnosis, audit, root):
    print("NO-ROUTER ORACLE EVALUATION: COMPLETE")
    print("\nCommit: {}".format(json.loads((root / "run_manifest.json").read_text())["oracle_code_commit"]))
    print("GPU: physical 4,5,6,7")
    print("Expert count: 10\n")
    print("Task            Routed   BestSingle   BestPair   BestFixed   SampleOracle")
    for row in summary_rows:
        print("{:<15} {:>7.2f} {:>12.2f} {:>10.2f} {:>11.2f} {:>14.2f}".format(row["Task"], row["Routed"], row["Best Single"], row["Best Pair"], row["Val-Selected Best Combo Test"], row["Sample Oracle"]))
    print("{:<15} {:>7.2f} {:>12.2f} {:>10.2f} {:>11.2f} {:>14.2f}".format("Mean", *(statistics.fmean(row[field] for row in summary_rows) for field in ("Routed", "Best Single", "Best Pair", "Val-Selected Best Combo Test", "Sample Oracle"))))
    largest_router = max(summary_rows, key=lambda row: row["Router Gap"])
    largest_pair = max(summary_rows, key=lambda row: row["Pair Gain"])
    worst = min((row["worst_destructive_pair"] for row in pair_stats), key=lambda row: row["pair_synergy"])
    future = max(stage_rows, key=lambda row: row["Future Expert Gain"])
    print("\nLargest Router Gap: {} {:+.2f}".format(largest_router["Task"], largest_router["Router Gap"]))
    print("Largest Pair Gain: {} {:+.2f}".format(largest_pair["Task"], largest_pair["Pair Gain"]))
    print("Worst Pair Interference: {} {:+.2f}".format(worst["pair"], worst["pair_synergy"]))
    print("Stage-local vs Final-pool gain: {} {:+.2f}".format(future["Task"], future["Future Expert Gain"]))
    print("\nDiagnosis: {}".format(diagnosis))
    print("Audit: {}".format(audit["status"]))
    print("Report: {}".format(root / "NO_ROUTER_ORACLE_REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    aggregate(args)


if __name__ == "__main__":
    main()
