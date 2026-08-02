#!/usr/bin/env python3
"""Freeze Stage 04 configs before any formal Oracle generation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


root = Path(__file__).resolve().parents[1]
configs = {}
for name in ("oracle_direct", "oracle_rms"):
    path = root / "configs" / (name + ".yaml")
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    configs[name] = {
        "config": value,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
if configs["oracle_direct"]["config_sha256"] == configs["oracle_rms"]["config_sha256"]:
    raise AssertionError("Direct and RMS configs must have distinct hashes")
manifest = {
    "stage": 4,
    "status": "PREREGISTERED_BEFORE_FORMAL_RESULTS",
    "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "git_head": "4ce1c5322c785e2ddd7f58a07c1a356807937102",
    "test_answers_permitted": False,
    "future_experts_permitted": False,
    "direct_and_rms_labels_may_be_mixed": False,
    "configs": configs,
}
target = root / "manifests" / "preregistration.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({name: row["config_sha256"] for name, row in configs.items()}, sort_keys=True))
