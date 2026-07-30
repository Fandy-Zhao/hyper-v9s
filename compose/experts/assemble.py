"""Assemble selected experts from compatible Compose checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .checkpoint import MANIFEST_NAME, WEIGHTS_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(checkpoint_dir: str, expert_id: int):
    root = Path(checkpoint_dir)
    manifest_path = root / MANIFEST_NAME
    weights_path = root / WEIGHTS_NAME
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported Compose checkpoint format: {}".format(root))
    matches = [
        item for item in manifest.get("experts", [])
        if int(item["expert_id"]) == int(expert_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            "checkpoint {} must contain expert {} exactly once".format(root, expert_id)
        )
    state = torch.load(str(weights_path), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError("Compose checkpoint weights must be a tensor dictionary")
    marker = ".experts.{}.".format(int(expert_id))
    selected = {key: value for key, value in state.items() if marker in key}
    expected = 2 * len(manifest["adapter"]["layers"])
    if len(selected) != expected:
        raise ValueError(
            "expert {} in {} has {} tensors, expected {}".format(
                expert_id, root, len(selected), expected
            )
        )
    return manifest, matches[0], selected, {
        "checkpoint_dir": str(root.resolve()),
        "expert_id": int(expert_id),
        "manifest_sha256": _sha256(manifest_path),
        "weights_sha256": _sha256(weights_path),
    }


def assemble_expert_checkpoint(
    sources: Sequence[Tuple[str, int]], output_dir: str
) -> Dict[str, object]:
    """Create a checkpoint containing exactly one selected expert per source."""
    if not sources:
        raise ValueError("at least one source expert is required")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing non-empty output directory: {}".format(output))

    adapter = None
    experts: List[Dict[str, object]] = []
    provenance: List[Dict[str, object]] = []
    combined = {}
    seen_ids = set()
    for checkpoint_dir, expert_id in sources:
        if int(expert_id) in seen_ids:
            raise ValueError("duplicate expert id: {}".format(expert_id))
        manifest, metadata, state, source_info = _load_source(
            checkpoint_dir, int(expert_id)
        )
        if adapter is None:
            adapter = manifest["adapter"]
        elif manifest["adapter"] != adapter:
            raise ValueError("source checkpoints have incompatible adapter metadata")
        overlap = set(combined).intersection(state)
        if overlap:
            raise ValueError("source checkpoints contain duplicate tensor keys")
        seen_ids.add(int(expert_id))
        experts.append(metadata)
        provenance.append(source_info)
        combined.update(state)

    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / WEIGHTS_NAME
    torch.save(combined, str(weights_path))
    result = {
        "format_version": 1,
        "experts": sorted(experts, key=lambda item: int(item["expert_id"])),
        "adapter": adapter,
        "metrics": {
            "adapter_tensor_count": len(combined),
            "adapter_parameter_count": sum(value.numel() for value in combined.values()),
            "checkpoint_bytes": weights_path.stat().st_size,
        },
        "assembly": {"sources": provenance},
    }
    with (output / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", nargs=2, metavar=("CHECKPOINT", "EXPERT_ID"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sources = [(path, int(expert_id)) for path, expert_id in args.source]
    result = assemble_expert_checkpoint(sources, args.output_dir)
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
