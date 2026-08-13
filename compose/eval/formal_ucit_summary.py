"""Final aggregation for the formal UCIT run (spec §26-§32).

Run after the whole Task0..Task5 sequence completed:

1. ``continual_matrix.csv/.json`` -- the 6x6 lower-triangular matrix A[t][j].
2. Continual metrics -- computed by the ORIGINAL Hyper-LLaVA
   ``scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py`` on the mirrored
   result layout (authoritative), and recomputed by this wrapper from the
   matrix for the §28 self-consistency check (|A - B| < 1e-8 required).
3. ``final_ucit_table.csv`` -- Hyper-LLaVA-paper row + Compose-seed42 row.
   The Compose per-task columns are A[5][j] (final continual performance),
   never the diagonal (spec §29).
4. Method diagnostics: expert growth, cross-task reuse, clustering, routing,
   keys, RMS statistics (spec §15-§18, §32).
5. Routing-collapse audit (pair usage > 90% flags a warning, never stops).

Everything here reads persisted run artifacts; nothing re-trains or
re-evaluates.
"""

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from compose.eval.formal_ucit_eval import HYPER_TASKS

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
ORIGINAL_METRICS_SCRIPT = (
    "/home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py"
)

#: Hyper-LLaVA paper Table 2 published reference (spec §8). Reference only --
#: never used by any computation.
PAPER_ROW = {
    "Method": "Hyper-LLaVA-paper",
    "ImgNetR": 87.20, "ArxivQA": 93.70, "VizWiz": 57.24, "IconQA": 75.20,
    "CLEVR": 65.60, "Flickr": 57.92,
    "MFN": 72.81, "MAA": 81.87, "MFT": 78.03, "BWT": -5.22,
}

DATASET_SHORT = ["ImgNetR", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr"]


def _load_matrix(root: Path) -> Dict[str, Any]:
    path = root / "evaluation" / "continual_matrix.json"
    if not path.is_file():
        raise FileNotFoundError("no continual_matrix.json at {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_rows(matrix: Dict[str, Any], num_tasks: int = 6) -> List[List[Optional[float]]]:
    """Rows A[t][j] for j <= t; NA (None) for not-yet-learned tasks."""
    rows = []
    for t in range(num_tasks):
        row = [None] * num_tasks
        for j in range(t + 1):
            entry = matrix["rows"].get(str(t), {}).get(str(j))
            if entry is not None:
                row[j] = float(entry["value"])
        rows.append(row)
    return rows


def _rounded_mean(values: List[float]) -> float:
    return round(statistics.fmean(values), 6) if values else None


def wrapper_metrics(rows: List[List[Optional[float]]]) -> Dict[str, float]:
    """Wrapper recomputation of MFN/MAA/MFT/BWT from the matrix -- mirrors
    the original summarize_continual_metrics.py formulas exactly (including
    ``round(statistics.fmean(...), 6)``) so the §28 self-consistency check
    is meaningful."""
    num_tasks = len(rows)
    stage_averages = [_rounded_mean([value for value in row if value is not None]) for row in rows]
    final_row = rows[-1]
    diagonal = [rows[i][i] for i in range(num_tasks)]
    if num_tasks > 1:
        bwt_values = [final_row[i] - diagonal[i] for i in range(num_tasks - 1)]
        bwt = _rounded_mean(bwt_values)
    else:
        bwt = 0.0
    return {
        "MAA": _rounded_mean(stage_averages),
        "MFN": _rounded_mean(final_row),
        "MFT": _rounded_mean(diagonal),
        "BWT": bwt,
    }


def original_metrics(root: Path) -> Dict[str, Any]:
    """Run the ORIGINAL Hyper-LLaVA continual metrics implementation on the
    mirrored result layout (authoritative metrics source)."""
    mirror_root = root / "evaluation" / "hyper_result_root"
    if not mirror_root.is_dir():
        raise FileNotFoundError(
            "mirrored hyper result root missing: {} (run formal_ucit_eval "
            "for every stage first)".format(mirror_root)
        )
    result = subprocess.run(
        [PYTHON, ORIGINAL_METRICS_SCRIPT, "--result-root", str(mirror_root),
         "--num-tasks", "6", "--output-file",
         str(root / "evaluation" / "continual_metrics_original.json")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("original metrics script failed:\n{}".format(result.stderr[-4000:]))
    payload_path = root / "evaluation" / "continual_metrics_original.json"
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _write_continual_matrix_csv(root: Path, rows: List[List[Optional[float]]]) -> None:
    path = root / "evaluation" / "continual_matrix.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage"] + DATASET_SHORT)
        for t, row in enumerate(rows):
            writer.writerow(["After T{}".format(t)] + [
                value if value is not None else "NA" for value in row
            ])


def _write_final_table(root: Path, compose_metrics: Dict[str, float]) -> None:
    row = {
        "Method": "Compose-seed42",
        "ImgNetR": compose_metrics["final_per_task"][0],
        "ArxivQA": compose_metrics["final_per_task"][1],
        "VizWiz": compose_metrics["final_per_task"][2],
        "IconQA": compose_metrics["final_per_task"][3],
        "CLEVR": compose_metrics["final_per_task"][4],
        "Flickr": compose_metrics["final_per_task"][5],
        "MFN": compose_metrics["MFN"],
        "MAA": compose_metrics["MAA"],
        "MFT": compose_metrics["MFT"],
        "BWT": compose_metrics["BWT"],
    }
    path = root / "evaluation" / "final_ucit_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAPER_ROW.keys()))
        writer.writeheader()
        writer.writerow(PAPER_ROW)
        writer.writerow(row)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_expert_diagnostics(root: Path, num_tasks: int = 6) -> Dict[str, Any]:
    """Per-task diagnostics and the cross-task expert reuse matrix from the
    persisted run artifacts (spec §15-§17)."""
    growth_rows = []
    reuse_rows: Dict[int, Dict[str, Any]] = {}
    routing_rows = []
    cluster_rows = []
    key_rows = []
    rms_rows = []
    running_experts: Dict[int, int] = {}

    for t in range(num_tasks):
        task_root = root / "task{}".format(t)
        teacher = _load_json(task_root / "teacher" / "summary.json") or {}
        residual = _load_json(task_root / "residual" / "summary.json") or {}
        assignment = _load_json(task_root / "cluster" / "assignment_stats.json") or {}
        commits = _load_json(task_root / "committed" / "commit_summary.json") or {}
        run_summary = _load_json(task_root / "eval_output" / "run_summary.json") or {}
        rms_summary = _load_json(task_root / "rms" / "rms_summary.json") or {}
        keys = _load_json(task_root / "keys" / "key_learning_results.json") or {}
        recall = _load_json(task_root / "teacher" / "recall_audit.json") or {}
        contribution = _load_json(task_root / "contribution" / "summary.json") or {}

        teacher_train = _load_json(task_root / "teacher" / "teacher_records_train.json") or []
        committed_ids = [int(item["expert_id"]) for item in commits.get("committed", [])]
        new_ids = set(committed_ids)
        for expert_id in committed_ids:
            entry = reuse_rows.setdefault(expert_id, {
                "expert_id": expert_id,
                "creation_task": t,
                "selected_count": [0] * num_tasks,
                "teacher_positive_count": [0] * num_tasks,
                "reuse_count": [0] * num_tasks,
            })
            entry["creation_task"] = t
        # Per-sample teacher sets for this task (train split): an expert is
        # "selected" when it was retrieved, "teacher-positive" when it is in
        # the teacher set, "reused" when teacher-positive AND pre-existing.
        for record in teacher_train:
            retrieved = [int(v) for v in record.get("retrieved_top_m", [])]
            teacher_set = [int(v) for v in record.get("teacher_set", [])]
            for expert_id in retrieved:
                if expert_id in reuse_rows:
                    reuse_rows[expert_id]["selected_count"][t] += 1
            for expert_id in teacher_set:
                if expert_id in reuse_rows:
                    reuse_rows[expert_id]["teacher_positive_count"][t] += 1
                    if expert_id not in new_ids:
                        reuse_rows[expert_id]["reuse_count"][t] += 1

        running_experts = dict(running_experts)
        for expert_id in committed_ids:
            running_experts[expert_id] = t
        growth_rows.append({
            "task": t,
            "old_experts": len(running_experts) - len(committed_ids),
            "residual_samples": residual.get("residual_train_count"),
            "selected_K": assignment.get("selected_k"),
            "silhouette": assignment.get("selected_silhouette"),
            "cluster_sizes": assignment.get("cluster_sizes"),
            "noise_count": len(assignment.get("noise_sample_ids", []) or []),
            "new_experts": len(committed_ids),
            "total_experts": len(running_experts),
            "reuse_rate": residual.get("reuse_rate"),
            "residual_rate": residual.get("residual_rate"),
        })
        routing_rows.append({
            "task": t,
            "teacher_empty": teacher.get("train_teacher_empty_count"),
            "teacher_single": teacher.get("train_teacher_single_count"),
            "teacher_pair": teacher.get("train_teacher_pair_count"),
            "eval_histogram": run_summary.get("router_selection_histogram"),
            "EmptyAcc": contribution.get("EmptyAcc"),
            "SingleRecall": contribution.get("SingleRecall"),
            "PairRecall": contribution.get("PairRecall"),
            "SetExactAcc": contribution.get("SetExactAcc"),
            "average_active_experts": contribution.get("average_active_experts"),
            "historical_route_decision_drift": contribution.get("historical_route_decision_drift"),
        })
        cluster_rows.append({
            "task": t,
            "selected_k": assignment.get("selected_k"),
            "silhouette": assignment.get("selected_silhouette"),
            "silhouette_by_k": assignment.get("silhouette_by_k"),
            "cluster_sizes": assignment.get("cluster_sizes"),
        })
        key_rows.append({
            "task": t,
            "results": {
                str(expert_id): {
                    "final_loss": info.get("final_loss"),
                    "positive_similarity_after": info.get("positive_similarity_after"),
                    "epochs_run": info.get("epochs_run"),
                }
                for expert_id, info in sorted(keys.items())
            },
        })
        rms_rows.append({
            "task": t,
            "calibration_sha256": rms_summary.get("calibration_sha256"),
            "layers_with_kappa": rms_summary.get("layers_with_kappa"),
            "split": rms_summary.get("calibration_split"),
        })

    reuse_table = []
    for expert_id in sorted(reuse_rows):
        entry = reuse_rows[expert_id]
        row = {
            "expert_id": expert_id,
            "creation_task": entry["creation_task"],
            "future_task_reuse_count": sum(entry["reuse_count"][entry["creation_task"] + 1:]),
            "cross_task_reuse_support": any(
                count > 0 for count in entry["reuse_count"][entry["creation_task"] + 1:]
            ),
        }
        for t in range(num_tasks):
            row["selected_T{}".format(t)] = entry["selected_count"][t]
            row["positive_T{}".format(t)] = entry["teacher_positive_count"][t]
            row["reuse_T{}".format(t)] = entry["reuse_count"][t]
        reuse_table.append(row)

    return {
        "expert_growth": growth_rows,
        "expert_reuse": reuse_table,
        "routing": routing_rows,
        "clustering": cluster_rows,
        "keys": key_rows,
        "rms": rms_rows,
    }


def _write_csv_rows(root: Path, name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = root / "evaluation" / name
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _routing_collapse_audit(root: Path, routing_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    warnings = []
    for row in routing_rows:
        histogram = row.get("eval_histogram") or {}
        total = sum(histogram.values())
        if not total:
            continue
        pair_rate = histogram.get(2, 0) / total
        single_rate = histogram.get(1, 0) / total
        empty_rate = histogram.get(0, 0) / total
        entry = {
            "task": row["task"],
            "empty_rate": empty_rate,
            "single_rate": single_rate,
            "pair_rate": pair_rate,
            "flags": [],
        }
        if pair_rate > 0.9:
            entry["flags"].append("ROUTING_COLLAPSE_WARNING")
        if pair_rate >= 1.0:
            entry["flags"].append("METHOD_OBSERVATION")
        warnings.append(entry)
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    matrix = _load_matrix(root)
    rows = _matrix_rows(matrix)
    _write_continual_matrix_csv(root, rows)

    original = original_metrics(root)
    wrapper = wrapper_metrics(rows)
    checks = {}
    for key in ("MFN", "MAA", "MFT", "BWT"):
        diff = abs(original["metrics"][key] - wrapper[key])
        checks[key] = {"original": original["metrics"][key], "wrapper": wrapper[key],
                       "difference": diff, "pass": diff < 1e-8}
    if not all(check["pass"] for check in checks.values()):
        raise RuntimeError(
            "metric self-consistency FAILED: {}".format(json.dumps(checks, indent=2))
        )

    final_per_task = [original["final_performance_by_task"][task["dataset"]]
                      for task in HYPER_TASKS]
    compose_metrics = {
        "MFN": original["metrics"]["MFN"],
        "MAA": original["metrics"]["MAA"],
        "MFT": original["metrics"]["MFT"],
        "BWT": original["metrics"]["BWT"],
        "final_per_task": final_per_task,
    }
    _write_final_table(root, compose_metrics)

    diagnostics = collect_expert_diagnostics(root)
    _write_csv_rows(root, "expert_growth.csv", diagnostics["expert_growth"])
    _write_csv_rows(root, "cross_task_contribution.csv", diagnostics["expert_reuse"])
    collapse = _routing_collapse_audit(root, diagnostics["routing"])
    payload = {
        "schema_version": 1,
        "continual_matrix": rows,
        "continual_metrics_original": original["metrics"],
        "continual_metrics_wrapper": wrapper,
        "self_consistency": checks,
        "final_per_task": {
            DATASET_SHORT[j]: final_per_task[j] for j in range(6)
        },
        "diagnostics": diagnostics,
        "routing_collapse_audit": collapse,
    }
    with open(root / "evaluation" / "compose_method_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(root / "evaluation" / "routing_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump({"routing": diagnostics["routing"], "collapse_audit": collapse}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    wrapper_path = root / "evaluation" / "continual_metrics_wrapper.json"
    wrapper_path.write_text(json.dumps({"metrics": wrapper, "self_consistency": checks}, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "metrics_original": original["metrics"],
        "self_consistency": checks,
        "final_per_task": payload["final_per_task"],
        "final_ucit_table": str(root / "evaluation" / "final_ucit_table.csv"),
        "diagnostics": str(root / "evaluation" / "compose_method_diagnostics.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
