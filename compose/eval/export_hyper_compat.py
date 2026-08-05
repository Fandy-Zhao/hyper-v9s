"""V6 snapshot -> original Hyper-LLaVA eval compatibility export (E12).

The original eval entry (``llava.eval.model_answer`` -> Hyper PEFT
``PeftModel.from_pretrained``) reads ``adapter_config.json`` +
``adapter_model.bin``. This script converts a compose expert checkpoint
directory (``compose_experts.json`` + ``compose_experts.bin``) into that
layout, so the original Hyper eval can load a V6 snapshot's experts.

Key mapping: ``model.layers.N.<module>.experts.<id>.lora_{A,B}.weight`` ->
``base_model.model.model.layers.N.<module>.lora_{A,B}.weight``.
"""

import argparse
import json
import os
from pathlib import Path

import torch


def _compose_keys(manifest: dict) -> list:
    return manifest.get("layers", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--expert-id", required=True, type=int)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    manifest_path = checkpoint_dir / "compose_experts.json"
    weights_path = checkpoint_dir / "compose_experts.bin"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "compose checkpoint missing in {}".format(checkpoint_dir)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = torch.load(weights_path, map_location="cpu", weights_only=False)

    prefix = "base_model.model."
    converted = {}
    for key, tensor in state.items():
        # key: model.layers.N.self_attn.q_proj.experts.<id>.lora_A.weight
        parts = key.split(".")
        expert_id = int(parts[parts.index("experts") + 1])
        if expert_id != args.expert_id:
            continue
        layer_path = ".".join(parts[: parts.index("experts")])
        parameter = ".".join(parts[-2:])  # lora_A.weight | lora_B.weight
        # Standard PEFT key (readable by the repo's peft eval path):
        # base_model.model.<layer_path>.lora_A.weight
        converted[
            "{}{}.{}".format(prefix, layer_path, parameter)
        ] = tensor
    if not converted:
        raise ValueError(
            "no tensors for expert {} in {}".format(args.expert_id, checkpoint_dir)
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(converted, str(output / "adapter_model.bin"))
    # Full decoder-projection paths (matches decoder_projection_names).
    layer_names = manifest.get("adapter", {}).get("layers", [])
    target_modules = sorted(layer_names)  # already model.layers.N.<module>
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(args.rank),
        "lora_alpha": float(args.alpha),
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
        "base_model_name_or_path": args.base_model,
    }
    (output / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PEFT-compatible export written to {} ({} tensors, expert {})".format(
            output, len(converted), args.expert_id
        )
    )


if __name__ == "__main__":
    main()
