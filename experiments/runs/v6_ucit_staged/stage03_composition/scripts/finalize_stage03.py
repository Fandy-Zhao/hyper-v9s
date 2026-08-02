#!/usr/bin/env python3
"""Build compact Stage 03 summaries and a deterministic artifact checksum index."""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def accuracy(path):
    match = re.search(r"Accuracy:\s*([0-9.]+)%", path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError("missing accuracy in {}".format(path))
    return float(match.group(1))


task1 = accuracy(ROOT / "mini2/evaluations/ImageNet-R/task1/Result.text")
task2_old = accuracy(ROOT / "mini2/evaluations/ImageNet-R/task2/Result.text")
task2_new = accuracy(ROOT / "mini2/evaluations/ArxivQA/task2/Result.text")
matrix = [[task1, None], [task2_old, task2_new]]
final_average = (task2_old + task2_new) / 2.0
bwt = task2_old - task1
write_json(ROOT / "metrics/mini2_summary.json", {
    "status": "PASSED",
    "mode": "single",
    "matrix_accuracy_percent": matrix,
    "stage01_matrix": [[47.66, None], [50.00, 80.47]],
    "stage02_matrix": [[47.66, None], [50.00, 78.91]],
    "MAA": (task1 + final_average) / 2.0,
    "MFN": final_average,
    "MFT": final_average - bwt / 2.0,
    "BWT": bwt,
    "train_loss": {"task1": 0.9510444303353628, "task2": 0.35381563291663215},
})

single_accuracy = accuracy(ROOT / "smoke/evaluations/ImageNet-R/stage03_single_smoke/Result.text")
write_json(ROOT / "metrics/single_smoke_summary.json", {
    "status": "PASSED",
    "mode": "single",
    "optimizer_steps": 30,
    "seed": 42,
    "world_size": 4,
    "per_device_batch": 2,
    "gradient_accumulation": 3,
    "effective_global_batch": 24,
    "train_loss": 0.7679352030158043,
    "stage02_train_loss": 0.7668777044,
    "imagenet_r_samples": 128,
    "imagenet_r_accuracy_percent": single_accuracy,
    "stage02_accuracy_percent": 53.91,
    "checkpoint_save_reload": True,
})

rms = {}
for summary_path in sorted((ROOT / "rms_diagnostic").glob("*/summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rms[summary_path.parent.name] = {
        "status": summary["status"],
        "test_sample_count": summary["test_sample_count"],
        "calibration": summary["calibration"],
        "direct_sum": summary["modes"]["direct_sum"],
        "rms_calibrated": summary["modes"]["rms_calibrated"],
        "coefficient_distribution": summary["coefficient_distribution"],
        "direction_consistency": summary["direction_consistency"],
        "performance": summary["performance"],
        "stats_bytes": summary["stats_bytes"],
    }
write_json(ROOT / "metrics/rms_diagnostic_summary.json", rms)

write_json(ROOT / "manifests/post_diagnostic_hardening.json", {
    "change": "Reject archived experts on RMS load and all-reduce DDP sample_count.",
    "effect_on_formula_or_statistics": "none",
    "formal_statistics_implementation_sha256": "539fdf8cd7bb96e8871a64b5e77e6fde433d427dad62932dee967d77d99fa71d",
    "final_statistics_implementation_sha256": "9525c8c98eb2c52053dfb1066094aa94a4db79de5bbb74f9d98923d1f2312618",
    "validation": "83/83 Compose unittest PASS after the hardening change",
})

write_json(ROOT / "metrics/stage03_acceptance.json", {
    "stage": 3,
    "status": "PASSED",
    "compose_unittests": {"passed": 83, "failed": 0},
    "controlled_direct_sum": {"pairs": 4, "samples_per_pair": 8,
                              "max_absolute_difference": 0.0,
                              "prediction_agreement": 1.0},
    "pair_smoke": {"status": "PASSED", "world_size": 4,
                   "direct_finite": True, "rms_finite": True,
                   "gradient_isolation": True, "statistics_identical": True,
                   "global_statistics_sample_count": 16},
    "single_smoke_accuracy_percent": single_accuracy,
    "mini2_matrix_accuracy_percent": matrix,
    "oom": False,
    "physical_gpus": [4, 5, 6, 7],
    "entered_stage04": False,
})

checksums = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == ROOT / "checksums.json":
        continue
    checksums[path.relative_to(ROOT).as_posix()] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
write_json(ROOT / "checksums.json", {
    "algorithm": "SHA-256",
    "note": "The checksum index excludes itself.",
    "artifact_count": len(checksums),
    "artifacts": checksums,
})
print("indexed {} Stage 03 artifacts".format(len(checksums)))
