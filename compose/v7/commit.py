"""Commit only retained V7 candidates into an inference-ready checkpoint."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import torch

from compose.experts.checkpoint import MANIFEST_NAME, WEIGHTS_NAME

from .pool import V7ExpertKeyPool


_EXPERT_KEY = re.compile(r"\.experts\.(\d+)\.")


def _fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _validate_commit_directory(path, selected):
    root = Path(path)
    required = (root / WEIGHTS_NAME, root / MANIFEST_NAME, root / "v7_keys.pt")
    if not all(value.is_file() and value.stat().st_size > 0 for value in required):
        raise RuntimeError("incomplete V7 commit transaction")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest_ids = {int(value["expert_id"]) for value in manifest["experts"]}
    if manifest_ids != set(selected):
        raise RuntimeError("committed manifest expert set mismatch")
    restored = V7ExpertKeyPool.from_state(
        torch.load(root / "v7_keys.pt", map_location="cpu", weights_only=False)
    )
    if set(restored.selectable_ids()) != set(selected):
        raise RuntimeError("committed key pool expert set mismatch")


def commit_retained_candidates(
    source_checkpoint: str,
    output_dir: str,
    key_pool: V7ExpertKeyPool,
    retained_ids: Iterable[int],
    candidate_metrics: Mapping[int, Mapping[str, object]],
) -> None:
    retained = {int(value) for value in retained_ids}
    historical = set(key_pool.historical_ids)
    if not retained.issubset(set(key_pool.current_ids)):
        raise ValueError("retained IDs must be current candidates")
    selected = historical | retained
    if len(selected) < 2:
        raise ValueError(
            "V7 commit requires at least two selectable experts for Global Top-2"
        )
    source = Path(source_checkpoint)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".{}-".format(target.name), dir=target.parent))
    with (source / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    state = torch.load(source / WEIGHTS_NAME, map_location="cpu", weights_only=False)
    filtered = {}
    for name, value in state.items():
        match = _EXPERT_KEY.search(name)
        if match and int(match.group(1)) in selected:
            filtered[name] = value
    experts = []
    for entry in manifest["experts"]:
        expert_id = int(entry["expert_id"])
        if expert_id not in selected:
            continue
        entry = dict(entry)
        entry["status"] = "frozen"
        entry["trainable"] = False
        entry["active"] = True
        entry["lifecycle_status"] = "formal"
        if expert_id in retained:
            extra = dict(entry.get("extra") or {})
            extra.update(
                {
                    "v7_lifecycle": "historical",
                    "v7_validation": dict(candidate_metrics[expert_id]),
                }
            )
            entry["extra"] = extra
        experts.append(entry)
    manifest["experts"] = experts
    manifest["metrics"] = {
        "adapter_tensor_count": len(filtered),
        "adapter_parameter_count": sum(value.numel() for value in filtered.values()),
    }
    calibration = manifest.get("rms_calibration")
    if calibration:
        manifest["rms_calibration"] = {
            layer: {
                str(expert_id): value
                for expert_id, value in values.items()
                if int(expert_id) in selected
            }
            for layer, values in calibration.items()
        }
        for expert_id in retained:
            key_pool.metadata[expert_id]["rms_state"] = {
                "mode": "commit_frozen",
                "per_layer_kappa": {
                    layer: float(values[str(expert_id)])
                    for layer, values in calibration.items()
                    if str(expert_id) in values
                },
            }
    committed_pool = V7ExpertKeyPool.from_state(key_pool.export_state())
    committed_pool.commit(retained, candidate_metrics)
    try:
        torch.save(filtered, temporary / WEIGHTS_NAME)
        manifest["metrics"]["checkpoint_bytes"] = os.path.getsize(
            temporary / WEIGHTS_NAME
        )
        with (temporary / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Pruned keys remain in audit state but are unselectable.
        torch.save(committed_pool.export_state(), temporary / "v7_keys.pt")
        _fsync_file(temporary / WEIGHTS_NAME)
        _fsync_file(temporary / "v7_keys.pt")
        _validate_commit_directory(temporary, selected)
        os.replace(temporary, target)
        try:
            descriptor = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            # Directory fsync is unavailable on some supported platforms;
            # same-filesystem rename remains the atomic visibility boundary.
            pass
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    key_pool.commit(retained, candidate_metrics)
