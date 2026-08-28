"""Prepare and summarize the evaluation-only commit-frozen RMS ablation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from compose.eval.formal_ucit_summary import wrapper_metrics


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(formal: Path, output: Path):
    frozen_root = output / "frozen_run"
    stage_calibrations = []
    for stage in range(6):
        path = formal / f"task{stage}" / "rms" / "rms_calibration.json"
        stage_calibrations.append(json.loads(path.read_text())["calibration"])
    origin = {}
    origin_stage = {}
    for stage, calibration in enumerate(stage_calibrations):
        for layer, values in calibration.items():
            origin.setdefault(layer, {})
            for expert_id, value in values.items():
                if expert_id not in origin[layer]:
                    origin[layer][expert_id] = value
                    origin_stage.setdefault(expert_id, stage)
    provenance = {"schema_version": 1, "mode": "commit_frozen_rms", "experts": {}}
    for stage in range(6):
        source_snapshot = formal / f"task{stage}" / "snapshots" / f"task{stage}"
        snapshot_manifest = json.loads((source_snapshot / "manifest.json").read_text())
        source_pool = Path(snapshot_manifest["pool_checkpoint_dir"])
        source_manifest_path = source_pool / "compose_experts.json"
        pool_manifest = json.loads(source_manifest_path.read_text())
        dynamic = pool_manifest["rms_calibration"]
        frozen = {}
        for layer, values in dynamic.items():
            frozen[layer] = {expert_id: origin[layer][expert_id] for expert_id in values}
        target_pool = output / "frozen_pools" / f"t{stage}"
        target_pool.mkdir(parents=True, exist_ok=True)
        weights = target_pool / "compose_experts.bin"
        if not weights.exists():
            os.symlink(str((source_pool / "compose_experts.bin").resolve()), weights)
        pool_manifest["rms_calibration"] = frozen
        pool_manifest["rank_study_rms_mode"] = "commit_frozen"
        pool_manifest["rank_study_source_manifest"] = str(source_manifest_path)
        (target_pool / "compose_experts.json").write_text(json.dumps(pool_manifest, indent=2, sort_keys=True) + "\n")
        target_snapshot = frozen_root / f"task{stage}" / "snapshots" / f"task{stage}"
        target_snapshot.mkdir(parents=True, exist_ok=True)
        router_target = target_snapshot / "router_checkpoint.pt"
        if not router_target.exists():
            os.symlink(str((source_snapshot / "router_checkpoint.pt").resolve()), router_target)
        snapshot_manifest["pool_checkpoint_dir"] = str(target_pool.resolve())
        snapshot_manifest["rank_study_rms_mode"] = "commit_frozen"
        (target_snapshot / "manifest.json").write_text(json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n")
        for expert_id in dynamic[next(iter(dynamic))]:
            provenance["experts"].setdefault(expert_id, {"origin_stage": origin_stage[expert_id]})
        provenance.setdefault("stages", {})[str(stage)] = {
            "source_pool": str(source_pool), "frozen_pool": str(target_pool.resolve()),
            "source_manifest_sha256": _hash(source_manifest_path),
            "router_sha256": _hash(source_snapshot / "router_checkpoint.pt"),
            "experts": sorted(int(value) for value in dynamic[next(iter(dynamic))]),
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_rms_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PREPARED", "frozen_root": str(frozen_root), "experts": len(origin_stage)}))


def matrix(path):
    payload = json.loads((path / "evaluation" / "continual_matrix.json").read_text())
    rows = []
    for stage in range(6):
        row = []
        for task in range(6):
            item = payload["rows"].get(str(stage), {}).get(str(task))
            row.append(float(item["value"]) if item else None)
        rows.append(row)
    return rows


def summarize(formal: Path, output: Path):
    dynamic, frozen = matrix(formal), matrix(output / "frozen_run")
    rows = []
    for setting, values in (("dynamic", dynamic), ("commit_frozen", frozen)):
        for stage in range(6):
            for task in range(stage + 1):
                rows.append({"setting": setting, "stage": stage, "eval_task": task, "metric": values[stage][task]})
    csv_path = output / "D_rms_ablation_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = {
        "dynamic": wrapper_metrics(dynamic), "commit_frozen": wrapper_metrics(frozen),
        "delta_frozen_minus_dynamic": {key: wrapper_metrics(frozen)[key] - wrapper_metrics(dynamic)[key] for key in ("MFN", "MAA", "MFT", "BWT")},
        "imagenet_r_stagewise": {str(stage): {"dynamic": dynamic[stage][0], "commit_frozen": frozen[stage][0], "delta": frozen[stage][0] - dynamic[stage][0]} for stage in range(6)},
    }
    (output / "D_rms_ablation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    report = ["# Experiment D — Dynamic RMS vs Commit-Frozen RMS", "", "No experts, keys, routes, clusters, or generation settings were retrained or changed.", "", "| setting | MFN | MAA | MFT | BWT |", "|---|---:|---:|---:|---:|", "| Dynamic | {MFN:.2f} | {MAA:.2f} | {MFT:.2f} | {BWT:.2f} |".format(**metrics["dynamic"]), "| Commit-frozen | {MFN:.2f} | {MAA:.2f} | {MFT:.2f} | {BWT:.2f} |".format(**metrics["commit_frozen"]), "", "## ImageNet-R stage-wise", "", "| stage | dynamic | frozen | delta |", "|---:|---:|---:|---:|"]
    for stage, item in metrics["imagenet_r_stagewise"].items():
        report.append("| {} | {:.2f} | {:.2f} | {:+.2f} |".format(stage, item["dynamic"], item["commit_frozen"], item["delta"]))
    (output / "D_rms_ablation_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(metrics, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "summarize"))
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else summarize)(Path(args.formal_root), Path(args.output_root))


if __name__ == "__main__":
    main()
