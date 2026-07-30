"""Verify deterministic recovery data and create the missing compatibility link."""

import hashlib
import json
from pathlib import Path


REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
STAGE = REPO / "experiments/runs/0730_residual_expert_feasibility/stage_f2"
RECOVERY = Path("/data/dataset/zhaozhuofan/controlled_functional_v1")
COMPATIBILITY_ROOT = Path("/data/dataset/zhaozhuofan/Hyper-LlaVA")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (STAGE / "dataset_manifest.json").open(encoding="utf-8") as handle:
        original = json.load(handle)
    rows = []
    for relative, expected in sorted(original["files"].items()):
        path = RECOVERY / relative
        actual = digest(path)
        if actual != expected:
            raise AssertionError("recovery hash mismatch: {}".format(relative))
        rows.append("{}  {}".format(actual, path))
    (STAGE / "recovery_file_hash_check.log").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    if COMPATIBILITY_ROOT.exists():
        raise FileExistsError(
            "refusing existing compatibility root: {}".format(COMPATIBILITY_ROOT)
        )
    COMPATIBILITY_ROOT.mkdir()
    link = COMPATIBILITY_ROOT / "controlled_functional_v1"
    link.symlink_to(RECOVERY, target_is_directory=True)
    manifest_hash = digest(RECOVERY / "manifest.json")
    (STAGE / "recovery_manifest.sha256").write_text(
        "{}  {}/manifest.json\n".format(manifest_hash, RECOVERY), encoding="utf-8"
    )
    print(json.dumps({
        "checked_files": len(rows),
        "compatibility_link": str(link),
        "manifest_sha256": manifest_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
