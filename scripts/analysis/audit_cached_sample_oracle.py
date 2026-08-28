#!/usr/bin/env python3
"""Audit and aggregate the cached V6.2 sample-oracle results offline.

This script deliberately performs no model loading and no GPU work.  It reads
the per-sample 56-way NLL cache, and uses already materialized fixed-pool
prediction metrics only when all 56 candidates are present for a task.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
ROOT = Path("/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/compose_v62_seed42_no_router_oracle")
REPORT_DIR = REPO / "experiments" / "reports"
OUT_DIR = ROOT / "cache_audit"
TASKS = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
ROUTED = [19.83, 88.53, 54.19, 26.37, 41.03, 51.41]
EXPECTED_ROWS = 3000
EXPECTED_CANDIDATES = 56
ORACLE_COMMIT = "1b2de8cc6cc07714ea24b0fe795d4a0dfb2224d3"
FORMAL_COMMIT = "fb5e9083d52cc996cdc2ab5a58634a8c489c5b6b"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:  # pragma: no cover - audit diagnostic
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(ids) -> str:
    ids = list(ids)
    return "[]" if not ids else "[{}]".format(",".join(f"E{x}" for x in ids))


def row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def fixed_metrics(task: int, split: str = "test"):
    values = []
    for index in range(EXPECTED_CANDIDATES):
        path = ROOT / "fixed" / "final_pool" / f"task{task}" / split / f"candidate_{index:02d}" / "score" / "metric.json"
        if not path.is_file():
            return None
        try:
            value = float(load_json(path)["value"])
        except (KeyError, ValueError, TypeError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def top_combos(values, candidates, count=5, reverse=False):
    return [
        {"combination": label(candidates[i]), "value": values[i], "index": i}
        for i in sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)[:count]
    ]


OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
all_rows = []
task_rows = []
combination_rows = []
audit_problems = []

candidate_sets = None
for task, task_name in enumerate(TASKS):
    nll_path = ROOT / "sample_oracle" / f"task{task}" / "nll.jsonl"
    rows = load_jsonl(nll_path)
    if len(rows) != EXPECTED_ROWS:
        audit_problems.append(f"{task_name}: nll rows={len(rows)} (expected {EXPECTED_ROWS})")
    if rows:
        current_candidates = [tuple(int(x) for x in ids) for ids in rows[0]["candidate_expert_ids"]]
        if candidate_sets is None:
            candidate_sets = current_candidates
        if current_candidates != candidate_sets:
            audit_problems.append(f"{task_name}: candidate set order differs across tasks")
    else:
        current_candidates = []
    if len(current_candidates) != EXPECTED_CANDIDATES:
        audit_problems.append(f"{task_name}: candidate count={len(current_candidates)}")

    width_counts = Counter(len(row.get("set_nll", [])) for row in rows)
    if width_counts != Counter({EXPECTED_CANDIDATES: EXPECTED_ROWS}):
        audit_problems.append(f"{task_name}: set_nll width counts={dict(width_counts)}")
    if any(row.get("router_called") is not False for row in rows):
        audit_problems.append(f"{task_name}: router_called was not false for every row")
    if any(row.get("model_commit") != ORACLE_COMMIT for row in rows):
        audit_problems.append(f"{task_name}: model commit mismatch in NLL cache")

    losses = []
    for index in range(len(current_candidates)):
        values = [float(row["set_nll"][index]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            audit_problems.append(f"{task_name}: non-finite loss in candidate {index}")
        losses.append(values)
    mean_losses = [statistics.fmean(values) for values in losses] if losses else []
    sample_oracle_losses = [
        float(row["set_nll"][int(row["best_overall_index"])]) for row in rows
    ]
    sample_oracle_loss = statistics.fmean(sample_oracle_losses) if sample_oracle_losses else None
    selected = [current_candidates[int(row["best_overall_index"])] for row in rows]
    size_counts = Counter(len(ids) for ids in selected)
    expert_usage = Counter(expert for ids in selected for expert in ids)
    pair_usage = Counter(ids for ids in selected if len(ids) == 2)
    test_metric_values = fixed_metrics(task, "test")
    sample_metric_path = ROOT / "sample_oracle" / f"task{task}" / "score" / "metric.json"
    sample_metric = float(load_json(sample_metric_path)["value"]) if sample_metric_path.is_file() else None

    loss_best_index = min(range(len(mean_losses)), key=lambda i: mean_losses[i])
    single_indices = [i for i, ids in enumerate(current_candidates) if len(ids) == 1]
    pair_indices = [i for i, ids in enumerate(current_candidates) if len(ids) == 2]
    best_single_loss_index = min(single_indices, key=lambda i: mean_losses[i])
    best_pair_loss_index = min(pair_indices, key=lambda i: mean_losses[i])

    for index, ids in enumerate(current_candidates):
        metric = test_metric_values[index] if test_metric_values is not None else None
        combination_rows.append(
            {
                "task": task_name,
                "combination": label(ids),
                "size": len(ids),
                "candidate_index": index,
                "mean_teacher_forcing_loss": mean_losses[index],
                "sample_count": len(rows),
                "test_metric": metric,
                "test_metric_source": "fixed/final_pool/.../score/metric.json" if metric is not None else "UNAVAILABLE",
            }
        )

    task_rows.append(
        {
            "task": task_name,
            "routed_metric": ROUTED[task],
            "nll_rows": len(rows),
            "candidate_count": len(current_candidates),
            "loss_best_combo": label(current_candidates[loss_best_index]),
            "loss_best_mean": mean_losses[loss_best_index],
            "loss_best_single": label(current_candidates[best_single_loss_index]),
            "loss_best_single_mean": mean_losses[best_single_loss_index],
            "loss_best_pair": label(current_candidates[best_pair_loss_index]),
            "loss_best_pair_mean": mean_losses[best_pair_loss_index],
            "sample_oracle_mean_loss": sample_oracle_loss,
            "sample_adaptivity_gain_loss": mean_losses[loss_best_index] - sample_oracle_loss,
            "sample_oracle_metric": sample_metric,
            "task_metric_status": "COMPLETE_56_COMBOS" if test_metric_values is not None else "UNAVAILABLE",
            "task_metric_best_single": (
                test_metric_values[max(single_indices, key=lambda i: test_metric_values[i])]
                if test_metric_values is not None else None
            ),
            "task_metric_best_single_combo": (
                label(current_candidates[max(single_indices, key=lambda i: test_metric_values[i])])
                if test_metric_values is not None else None
            ),
            "task_metric_oracle": max(test_metric_values) if test_metric_values is not None else None,
            "task_metric_oracle_combo": (
                label(current_candidates[max(range(len(test_metric_values)), key=lambda i: test_metric_values[i])])
                if test_metric_values is not None else None
            ),
            "empty_rate": size_counts[0] / len(rows),
            "single_rate": size_counts[1] / len(rows),
            "pair_rate": size_counts[2] / len(rows),
            "top_experts": [[expert, count] for expert, count in expert_usage.most_common(5)],
            "top_pairs": [[list(ids), count] for ids, count in pair_usage.most_common(5)],
            "top_loss_combos": top_combos(mean_losses, current_candidates, reverse=False),
            "source_nll": str(nll_path),
            "source_nll_sha256": sha256(nll_path),
            "source_generation": str(ROOT / "sample_oracle" / f"task{task}" / "generation_chunks"),
        }
    )
    all_rows.extend(rows)

manifest = load_json(ROOT / "run_manifest.json")
if manifest.get("oracle_code_commit") != ORACLE_COMMIT:
    audit_problems.append("run_manifest oracle_code_commit mismatch")
if manifest.get("formal_run_commit") != FORMAL_COMMIT:
    audit_problems.append("run_manifest formal_run_commit mismatch")
if manifest.get("physical_gpus_allowed") != [4, 5, 6, 7]:
    audit_problems.append("run_manifest physical_gpus_allowed mismatch")

old_cache = Path("/data/ckpt/zhaozhuofan/compose/oracle/task1_task4_experts01_rank8_seed42/oracle_cache.jsonl")
old_cache_info = {
    "path": str(old_cache),
    "present": old_cache.is_file(),
    "rows": None,
    "candidate_width": None,
    "tasks": [],
    "model_commits": [],
    "used_for_current_aggregation": False,
}
if old_cache.is_file():
    old_rows = load_jsonl(old_cache)
    old_cache_info["rows"] = len(old_rows)
    old_cache_info["candidate_width"] = len(old_rows[0].get("candidate_expert_ids", [])) if old_rows else 0
    old_cache_info["tasks"] = sorted({row.get("task_id") for row in old_rows})
    old_cache_info["model_commits"] = sorted({row.get("model_commit") for row in old_rows})

metric_complete_tasks = [row["task"] for row in task_rows if row["task_metric_status"] == "COMPLETE_56_COMBOS"]
summary = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "CACHE STATUS: COMPLETE (LOSS-LEVEL)",
    "metric_status": "METRIC CACHE STATUS: PARTIAL",
    "no_model_inference_performed": True,
    "gpu_used": False,
    "source_experiment": str(ROOT),
    "oracle_code_commit": ORACLE_COMMIT,
    "formal_run_commit": FORMAL_COMMIT,
    "expert_count": 10,
    "candidate_count": EXPECTED_CANDIDATES,
    "sample_count_per_task": EXPECTED_ROWS,
    "complete_loss_tasks": TASKS,
    "complete_56_combo_prediction_metric_tasks": metric_complete_tasks,
    "sample_oracle_metrics": {row["task"]: row["sample_oracle_metric"] for row in task_rows},
    "tasks": task_rows,
    "old_auxiliary_cache": old_cache_info,
    "audit_problems": sorted(set(audit_problems)),
}

audit = {
    "status": "PASS" if not audit_problems else "FAIL",
    "cache_status": summary["status"],
    "metric_status": summary["metric_status"],
    "checks": {
        "six_tasks_have_3000_nll_rows": all(row["nll_rows"] == EXPECTED_ROWS for row in task_rows),
        "six_tasks_have_56_nll_values_per_sample": all(row["candidate_count"] == EXPECTED_CANDIDATES for row in task_rows),
        "all_nll_values_finite": not any("non-finite" in item for item in audit_problems),
        "router_disabled_in_cache": not any("router_called" in item for item in audit_problems),
        "commit_matches_formal_run_manifest": not any("commit" in item for item in audit_problems),
        "no_model_inference_performed": True,
        "gpu_used": False,
    },
    "problems": sorted(set(audit_problems)),
    "prediction_metric_limit": "Only Task0 and Task1 have complete 56-candidate test prediction metrics. Task2 test was interrupted; Task3-5 test candidate prediction caches are absent.",
}

atomic_json(OUT_DIR / "cache_audit.json", audit)
atomic_json(OUT_DIR / "cache_summary.json", summary)
atomic_json(OUT_DIR / "combination_matrix.json", combination_rows)
with (OUT_DIR / "combination_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = list(combination_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(combination_rows)

lines = [
    "# Cached Sample Oracle Audit and Offline Task-Level Decomposition",
    "",
    "## Final status",
    "",
    f"- **{summary['status']}**",
    f"- **{summary['metric_status']}**",
    "- `STOPPED — 未启动任何重新推理或补充实验。`",
    "- GPU 使用：`False`（本轮仅 CPU 文件审计与聚合）。",
    "",
    "The six formal Sample Oracle NLL caches are complete at the loss level: every task has 3,000 samples and every sample has all 56 candidate-set losses. This is sufficient for exact mean teacher-forcing-loss matrices and loss-based Task Oracle decomposition. It is not sufficient to reconstruct benchmark accuracy for tasks whose per-combination prediction cache is absent.",
    "",
    "## Cache audit",
    "",
    f"- Source experiment: `{ROOT}`",
    f"- Oracle code commit: `{ORACLE_COMMIT}`",
    f"- Formal run commit: `{FORMAL_COMMIT}`",
    "- Expert pool: `E0..E9` (10 experts)",
    "- Candidate sets: `[] + 10 singles + 45 pairs = 56`",
    "- Primary raw cache: `sample_oracle/task{0..5}/nll.jsonl`",
    "- Each NLL file: `3000 rows × 56 set_nll values`, valid JSON, router_called=false",
    "- Sample generation cache: 3,000 selected-combination predictions per task, not 56 predictions per sample",
    "- Complete 56-combination test prediction/metric cache: Task0, Task1",
    "- Partial/interrupted test prediction cache: Task2",
    "- Missing test prediction cache: Task3, Task4, Task5",
    "- Direct Uniform/UR combination cache identified: none in the current formal oracle outputs; no values were inferred.",
    "",
    "## Task-level decomposition",
    "",
    "`Mean Loss` columns are exact offline aggregates from the 56-way NLL cache. `Metric` columns are benchmark metrics only where all 56 fixed candidate prediction files already exist. A lower loss is better.",
    "",
    "| Task | Best Single (loss) | Task Oracle (loss) | Sample Oracle (loss) | Adaptivity gain (loss) | Best Single (metric) | Task Oracle (metric) | Sample Oracle metric |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for row in task_rows:
    metric_single = "N/A" if row["task_metric_best_single"] is None else f"{row['task_metric_best_single']:.2f} {row['task_metric_best_single_combo']}"
    metric_oracle = "N/A" if row["task_metric_oracle"] is None else f"{row['task_metric_oracle']:.2f} {row['task_metric_oracle_combo']}"
    sample_metric = "N/A" if row["sample_oracle_metric"] is None else f"{row['sample_oracle_metric']:.2f}"
    lines.append(
        f"| {row['task']} | {row['loss_best_single']} ({row['loss_best_single_mean']:.6f}) | "
        f"{row['loss_best_combo']} ({row['loss_best_mean']:.6f}) | {row['sample_oracle_mean_loss']:.6f} | "
        f"{row['sample_adaptivity_gain_loss']:.6f} | {metric_single} | {metric_oracle} | {sample_metric} |"
    )

lines.extend(
    [
        "",
        "## Sample Oracle composition distribution",
        "",
        "| Task | Empty | Single | Pair | Top experts | Top pairs |",
        "|---|---:|---:|---:|---|---|",
    ]
)
for row in task_rows:
    experts_text = ", ".join(f"E{e}:{n}" for e, n in row["top_experts"])
    pairs_text = ", ".join(f"{label(ids)}:{n}" for ids, n in row["top_pairs"][:2]) or "N/A"
    lines.append(
        f"| {row['task']} | {100*row['empty_rate']:.2f}% | {100*row['single_rate']:.2f}% | "
        f"{100*row['pair_rate']:.2f}% | {experts_text} | {pairs_text} |"
    )

lines.extend(
    [
        "",
        "## Interpretation and limits",
        "",
        "1. The complete six-task 56-way loss cache makes the Task Oracle in the loss domain exactly reproducible without model inference.",
        "2. Sample Oracle benchmark scores remain available from the existing selected-combination generation outputs, but they must not be compared to a missing Task Oracle accuracy as if it were known.",
        "3. Task0 and Task1 have complete fixed-pool candidate predictions, so their benchmark Best Single/Task Oracle metrics can be recovered exactly from existing files. Task2 test was interrupted at the previously preserved partial checkpoint, and Task3–5 have no complete fixed-pool test prediction matrices.",
        "4. The old auxiliary cache `compose/oracle/task1_task4_experts01_rank8_seed42/oracle_cache.jsonl` contains only 6,000 rows over ImageNet-R/IconQA with four candidate sets and an older commit; it is not merged into this V6.2 ten-expert audit.",
        "",
        "## Generated artifacts",
        "",
        f"- Full loss/metric matrix: `{OUT_DIR / 'combination_matrix.csv'}`",
        f"- Machine-readable matrix: `{OUT_DIR / 'combination_matrix.json'}`",
        f"- Cache summary: `{OUT_DIR / 'cache_summary.json'}`",
        f"- Cache audit: `{OUT_DIR / 'cache_audit.json'}`",
        f"- Aggregation script: `{REPO / 'scripts/analysis/audit_cached_sample_oracle.py'}`",
        f"- This report: `{REPORT_DIR / 'task_level_oracle_from_cached_sample_oracle.md'}`",
        "",
        "No model forward, generation, GPU computation, cache completion, or rerun was performed.",
        "",
    ]
)
report = "\n".join(lines)
atomic_text(REPORT_DIR / "task_level_oracle_from_cached_sample_oracle.md", report)
atomic_text(OUT_DIR / "task_level_oracle_from_cached_sample_oracle.md", report)

print(summary["status"])
print(summary["metric_status"])
print("AUDIT", audit["status"])
print("REPORT", REPORT_DIR / "task_level_oracle_from_cached_sample_oracle.md")
print("MATRIX", OUT_DIR / "combination_matrix.csv")
for row in task_rows:
    print(row["task"], row["loss_best_combo"], f"loss={row['loss_best_mean']:.6f}", "metric_status=" + row["task_metric_status"])

if audit_problems:
    raise SystemExit(1)
