"""Measure per-layer LoRA delta magnitude and pairwise direction similarity."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch

from .checkpoint import MANIFEST_NAME, WEIGHTS_NAME


def _load(checkpoint_dir: str, expert_id: int):
    root = Path(checkpoint_dir)
    with (root / MANIFEST_NAME).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    state = torch.load(str(root / WEIGHTS_NAME), map_location="cpu")
    marker = ".experts.{}.".format(expert_id)
    selected = {key: value.float() for key, value in state.items() if marker in key}
    if len(selected) != 2 * len(manifest["adapter"]["layers"]):
        raise ValueError("checkpoint does not contain a complete expert {}".format(expert_id))
    scale = float(manifest["adapter"]["alpha"]) / int(manifest["adapter"]["rank"])
    return manifest, selected, scale


def _factors(state: Dict[str, torch.Tensor], expert_id: int, layer: str):
    prefix = "{}.experts.{}.".format(layer, expert_id)
    return state[prefix + "lora_A.weight"], state[prefix + "lora_B.weight"]


def factorized_delta_stats(
    first: Tuple[torch.Tensor, torch.Tensor, float],
    second: Tuple[torch.Tensor, torch.Tensor, float],
) -> Dict[str, float]:
    a1, b1, scale1 = first
    a2, b2, scale2 = second
    if a1.shape[1] != a2.shape[1] or b1.shape[0] != b2.shape[0]:
        raise ValueError("LoRA deltas must have matching input/output dimensions")
    gram_b1 = b1.T @ b1
    gram_a1 = a1 @ a1.T
    gram_b2 = b2.T @ b2
    gram_a2 = a2 @ a2.T
    norm1_sq = float(torch.sum(gram_b1 * gram_a1)) * scale1 * scale1
    norm2_sq = float(torch.sum(gram_b2 * gram_a2)) * scale2 * scale2
    cross_b = b1.T @ b2
    cross_a = a1 @ a2.T
    inner = float(torch.sum(cross_b * cross_a)) * scale1 * scale2
    norm1 = math.sqrt(max(norm1_sq, 0.0))
    norm2 = math.sqrt(max(norm2_sq, 0.0))
    denominator = norm1 * norm2
    cosine = inner / denominator if denominator else 0.0
    elements = b1.shape[0] * a1.shape[1]
    return {
        "first_delta_rms": norm1 / math.sqrt(elements),
        "second_delta_rms": norm2 / math.sqrt(elements),
        "cosine": max(-1.0, min(1.0, cosine)),
    }


def analyze_geometry(
    first_checkpoint: str,
    first_expert_id: int,
    second_checkpoint: str,
    second_expert_id: int,
) -> Dict[str, object]:
    manifest1, state1, scale1 = _load(first_checkpoint, first_expert_id)
    manifest2, state2, scale2 = _load(second_checkpoint, second_expert_id)
    layers1 = manifest1["adapter"]["layers"]
    layers2 = manifest2["adapter"]["layers"]
    if layers1 != layers2:
        raise ValueError("checkpoint layer lists differ")
    rows = []
    for layer in layers1:
        a1, b1 = _factors(state1, first_expert_id, layer)
        a2, b2 = _factors(state2, second_expert_id, layer)
        rows.append({
            "layer": layer,
            **factorized_delta_stats((a1, b1, scale1), (a2, b2, scale2)),
        })
    return {
        "first": {"checkpoint": first_checkpoint, "expert_id": first_expert_id},
        "second": {"checkpoint": second_checkpoint, "expert_id": second_expert_id},
        "layers": rows,
        "summary": {
            "layer_count": len(rows),
            "mean_first_delta_rms": sum(row["first_delta_rms"] for row in rows) / len(rows),
            "mean_second_delta_rms": sum(row["second_delta_rms"] for row in rows) / len(rows),
            "mean_cosine": sum(row["cosine"] for row in rows) / len(rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-checkpoint", required=True)
    parser.add_argument("--first-expert-id", type=int, required=True)
    parser.add_argument("--second-checkpoint", required=True)
    parser.add_argument("--second-expert-id", type=int, required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    result = analyze_geometry(
        args.first_checkpoint, args.first_expert_id,
        args.second_checkpoint, args.second_expert_id,
    )
    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError("refusing existing output: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
