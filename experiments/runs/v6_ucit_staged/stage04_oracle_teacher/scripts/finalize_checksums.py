#!/usr/bin/env python3
"""Create SHA-256 inventory for committed Stage 04 artifacts (excluding logs/caches)."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("experiments/runs/v6_ucit_staged/stage04_oracle_teacher")
OUTPUT = ROOT / "checksums.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    paths = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix not in {".log", ".pyc"} and path != OUTPUT
        and "failures" not in path.parts and "__pycache__" not in path.parts
    ]
    paths.extend(path for path in Path("compose/teacher").glob("*.py") if path.is_file())
    paths.append(Path("tests/compose/test_oracle_teacher_stage04.py"))
    paths.append(Path("docs/reports/v6_ucit_stage04_oracle_teacher.md"))
    entries = {
        str(path): {"sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in sorted(set(paths), key=str)
    }
    payload = {"algorithm": "sha256", "artifact_count": len(entries), "artifacts": entries}
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"artifacts": len(entries), "output": str(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
