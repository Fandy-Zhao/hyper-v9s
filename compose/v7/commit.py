"""Commit only retained V7 candidates into an inference-ready checkpoint."""

import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import torch

from compose.experts.checkpoint import MANIFEST_NAME, WEIGHTS_NAME

from .pool import V7ExpertKeyPool


_EXPERT_KEY = re.compile(r"\.experts\.(\d+)\.")


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
    source = Path(source_checkpoint)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
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
    torch.save(filtered, target / WEIGHTS_NAME)
    manifest["metrics"]["checkpoint_bytes"] = os.path.getsize(target / WEIGHTS_NAME)
    with (target / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    key_pool.commit(retained, candidate_metrics)
    # Pruned keys remain in the audit state but are lifecycle=pruned and hence
    # unselectable. The inference loader additionally rejects current keys.
    torch.save(key_pool.export_state(), target / "v7_keys.pt")

