#!/usr/bin/env python3
"""Aggregate Stage 04 summaries and emit machine-readable validation."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("experiments/runs/v6_ucit_staged/stage04_oracle_teacher")
DATA_ROOT = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/cache")
DIRECT_HASH = "10f2cdce7726e9f5cecdcebbd056555cd98393415e9665a5a80a4ce1f6b99fe5"
RMS_HASH = "12fa7623df4d392c748a61d8bf980c9610f1dcde25276eaae7b47701a24e0a8d"
TASK_IDS = {"ImageNet-R": 0, "ArxivQA": 1, "VizWiz": 2, "IconQA": 3, "CLEVR": 4, "Flickr30k": 5}
REQUIRED_METRICS = {
    "EmptyOracleRate", "SingleOracleRate", "PairOracleRate", "PairEvaluatedRate",
    "PairBetterThanBestSingleRate", "PairPassedThresholdRate", "PairSelectedRate",
    "mean_pair_synergy", "median_pair_synergy", "positive_pair_synergy_rate",
    "harmful_pair_rate", "worst_10_percent_pair_synergy", "pair_gain_ci95",
    "selected_set_mean_nll", "selected_set_exact_accuracy", "best_single_exact_accuracy",
    "selected_vs_best_single_accuracy_delta", "pair_selected_accuracy_delta_vs_best_single",
    "single_expert_metrics", "worst_tail_sample_ids", "collapse_checks", "performance",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def discover():
    items = []
    specifications = [
        ("controlled", ROOT / "controlled_regression", 8, 16),
        ("smoke", ROOT / "smoke", 4, 32),
        ("mini2", ROOT / "mini2", 8, 32),
        ("full_seed42", ROOT / "full_seed42", 24, 32),
    ]
    counts = {}
    for suite, base, expected, samples in specifications:
        paths = sorted(base.glob("**/direct_summary.json")) + sorted(base.glob("**/rms_summary.json"))
        counts[suite] = {"actual": len(paths), "expected": expected}
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            mode = "direct" if path.name.startswith("direct") else "rms"
            if suite == "controlled":
                task_name = path.parent.name
                scope = "post_task_diagnostic"
                cache = DATA_ROOT / "controlled" / task_name / (mode + ".json")
            else:
                task_name = path.parent.name
                scope = path.parent.parent.name
                cache = DATA_ROOT / suite / scope / task_name / (mode + ".json")
            miss_path = path.with_name(mode + "_cache_miss_summary.json")
            performance_data = json.loads(miss_path.read_text(encoding="utf-8")) if miss_path.exists() else data
            items.append({
                "suite": suite, "task_name": task_name, "scope": scope, "mode": mode,
                "path": str(path), "cache_path": str(cache), "expected_samples": samples,
                "summary": data, "performance_summary": performance_data,
            })
    return items, counts


def weighted(items, field):
    values = [(item["summary"].get(field), item["summary"]["samples"]) for item in items]
    values = [(float(value), count) for value, count in values if value is not None]
    return sum(value * count for value, count in values) / sum(count for _, count in values) if values else None


def aggregate(items):
    groups = defaultdict(list)
    for item in items:
        groups[(item["suite"], item["scope"], item["mode"])].append(item)
    fields = [
        "EmptyOracleRate", "SingleOracleRate", "PairOracleRate", "PairEvaluatedRate",
        "PairBetterThanBestSingleRate", "PairPassedThresholdRate", "mean_pair_synergy",
        "harmful_pair_rate", "selected_set_mean_nll", "selected_set_exact_accuracy",
        "best_single_exact_accuracy", "selected_vs_best_single_accuracy_delta",
        "average_selected_cardinality", "average_evaluated_singles", "average_evaluated_pairs",
    ]
    result = {}
    for key, values in sorted(groups.items()):
        result["/".join(key)] = {
            "runs": len(values), "samples": sum(item["summary"]["samples"] for item in values),
            **{field: weighted(values, field) for field in fields},
        }
    return result


def validate(items, counts):
    checks = []
    diagnostics = []

    def check(name, passed, details):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    check("formal_run_count", all(value["actual"] == value["expected"] for value in counts.values()), counts)
    check("sample_counts", all(item["summary"].get("samples") == item["expected_samples"] for item in items),
          {item["path"]: item["summary"].get("samples") for item in items})
    check("required_metrics", all(REQUIRED_METRICS <= set(item["summary"]) for item in items),
          {item["path"]: sorted(REQUIRED_METRICS - set(item["summary"])) for item in items
           if not REQUIRED_METRICS <= set(item["summary"])})
    check("finite_metrics", all(finite_tree(item["summary"]) for item in items), "all numeric summary values are finite")
    check("no_test_split", all("test" not in item["summary"]["provenance"]["split"] for item in items),
          sorted({item["summary"]["provenance"]["split"] for item in items}))
    check("composition_and_config_isolation", all(
        item["summary"]["composition_mode"] == ("direct_sum" if item["mode"] == "direct" else "rms_calibrated")
        and item["summary"]["config_hash"] == (DIRECT_HASH if item["mode"] == "direct" else RMS_HASH)
        for item in items
    ), {"direct": DIRECT_HASH, "rms": RMS_HASH})
    check("cache_files_and_checksums", all(Path(item["cache_path"]).is_file() for item in items),
          {item["cache_path"]: sha256_file(item["cache_path"]) for item in items if Path(item["cache_path"]).is_file()})
    check("no_duplicate_pairs", all(not item["summary"]["collapse_checks"]["duplicate_pair"] for item in items),
          "all formal summaries report duplicate_pair=false")

    temporal_failures = []
    for item in items:
        if item["suite"] == "controlled":
            continue
        task_id = TASK_IDS[item["task_name"]]
        visible = sorted(map(int, item["summary"]["provenance"]["expert_checkpoint_hashes"]))
        expected = list(range(task_id if item["scope"] == "historical_only" else task_id + 1))
        if visible != expected:
            temporal_failures.append({"path": item["path"], "visible": visible, "expected": expected})
    check("temporal_boundaries", not temporal_failures, temporal_failures or "historical/post-task pools match task chronology")

    rms_failures = []
    for item in items:
        visible_count = len(item["summary"]["provenance"]["expert_checkpoint_hashes"])
        rms_hash = item["summary"]["provenance"]["rms_statistics_hash"]
        if item["mode"] == "direct" and rms_hash != "none":
            rms_failures.append(item["path"])
        if item["mode"] == "rms" and visible_count >= 2 and rms_hash == "none":
            rms_failures.append(item["path"])
    check("rms_provenance", not rms_failures, rms_failures or "RMS statistics exist iff pair-capable RMS scoring requires them")

    manifest = json.loads((ROOT / "manifests/sample_subsets.json").read_text(encoding="utf-8"))
    check("sample_manifest_no_test", manifest.get("test_data_used") is False, {"test_data_used": manifest.get("test_data_used")})

    controlled = [item for item in items if item["suite"] == "controlled"]
    differences = [item["performance_summary"]["performance"].get("stage03_single_token_nll_max_absolute_difference") for item in controlled]
    differences = [float(value) for value in differences if value is not None]
    check("controlled_scorer_regression", len(differences) == 8 and max(differences) <= 2e-5,
          {"observations": len(differences), "max_absolute_difference": max(differences) if differences else None})

    smoke = [item for item in items if item["suite"] == "smoke"]
    smoke_repeat = []
    for item in smoke:
        miss = item["performance_summary"]
        same = all(item["summary"].get(field) == miss.get(field) for field in (
            "EmptyOracleRate", "SingleOracleRate", "PairOracleRate", "selected_set_mean_nll"
        ))
        smoke_repeat.append(item["summary"]["performance"].get("cache_hit") is True and miss["performance"].get("cache_hit") is False and same)
    check("smoke_cache_repeat", len(smoke_repeat) == 4 and all(smoke_repeat), smoke_repeat)

    peak = max(item["performance_summary"]["performance"].get("peak_cuda_memory_bytes", 0) for item in items)
    check("peak_memory_under_24gib", peak < 24 * 1024 ** 3, {"peak_cuda_memory_bytes": peak})
    for item in items:
        collapse = {key: value for key, value in item["summary"]["collapse_checks"].items() if value}
        if collapse:
            diagnostics.append({"path": item["path"], "collapse": collapse})
    check("direct_rms_paths_disjoint", len({item["cache_path"] for item in items}) == len(items),
          "every formal branch has an independent cache path")
    return {
        "status": "PASSED" if all(item["passed"] for item in checks) else "FAILED",
        "checks": checks, "non_blocking_collapse_diagnostics": diagnostics,
        "current_hyper_route_metric": "not available; emitted as null rather than inferred from Oracle candidates",
    }


def main():
    items, counts = discover()
    groups = aggregate(items)
    aggregate_payload = {
        "stage": "Stage 04", "formal_run_counts": counts, "groups": groups,
        "runs": [{
            "suite": item["suite"], "task_name": item["task_name"], "scope": item["scope"], "mode": item["mode"],
            "samples": item["summary"]["samples"], "summary_path": item["path"], "cache_path": item["cache_path"],
            "rates": {key: item["summary"][key] for key in ("EmptyOracleRate", "SingleOracleRate", "PairOracleRate")},
        } for item in items],
    }
    validation = validate(items, counts)
    (ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "validation").mkdir(parents=True, exist_ok=True)
    (ROOT / "metrics/aggregate_metrics.json").write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "validation/validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for mode, config_hash, composition_mode in (
        ("direct", DIRECT_HASH, "direct_sum"), ("rms", RMS_HASH, "rms_calibrated")
    ):
        branch_items = [item for item in items if item["mode"] == mode]
        branch_manifest = {
            "oracle_branch": mode, "composition_mode": composition_mode, "config_hash": config_hash,
            "run_count": len(branch_items), "cache_root": str(DATA_ROOT),
            "runs": [{"summary": item["path"], "cache": item["cache_path"]} for item in branch_items],
        }
        branch_summary = {key: value for key, value in groups.items() if key.endswith("/" + mode)}
        (ROOT / "manifests" / ("oracle_" + mode + "_runs.json")).write_text(
            json.dumps(branch_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (ROOT / "metrics" / ("oracle_" + mode + "_final_summary.json")).write_text(
            json.dumps(branch_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"runs": len(items), "validation": validation["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
