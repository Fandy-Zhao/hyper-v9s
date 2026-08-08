"""Compose expert assembly: cluster state dict -> standard compose
checkpoint (commit-time artifact writer).

Loads the base model, injects compose adapters, registers the old
experts (optional) and the new expert id, copies the cluster LoRA
weights from an ``expert_<id>.pt`` payload (written by
``compose.train.train_compose --compose-mode cluster_expert``), then
writes a standard compose expert checkpoint (compose_experts.json +
compose_experts.bin) whose directory can be consumed by
``load_compose_model`` and the original eval entry.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.model import ComposeLlavaForCausalLM, load_compose_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--expert-state-dict", required=True)
    parser.add_argument("--expert-id", required=True, type=int)
    parser.add_argument("--old-expert-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()

    config = load_compose_config(args.model_path)
    config.mm_vision_tower = args.vision_tower
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_projector_type = "mlp2x_gelu"
    model = ComposeLlavaForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16
    )
    model.get_model().initialize_vision_modules(
        SimpleNamespace(
            vision_tower=args.vision_tower,
            mm_vision_select_layer=-2,
            mm_vision_select_feature="patch",
            mm_projector_type="mlp2x_gelu",
            pretrain_mm_mlp_adapter=args.projector_path,
        )
    )
    adapter_config = ComposeAdapterConfig(rank=args.rank, alpha=args.alpha)
    inject_compose_adapters(model, adapter_config)
    pool = ExpertPool(ExpertManager(model))
    if args.old_expert_checkpoint:
        load_expert_checkpoint(pool, args.old_expert_checkpoint)
    pool.register(
        args.expert_id,
        name="expert_{:04d}".format(args.expert_id),
        source_checkpoint=args.old_expert_checkpoint,
    )

    payload = torch.load(args.expert_state_dict, map_location="cpu", weights_only=False)
    if int(payload["expert_id"]) != args.expert_id:
        raise ValueError(
            "state dict expert id {} != requested expert id {}".format(
                payload["expert_id"], args.expert_id
            )
        )
    with torch.no_grad():
        for layer_name, layer in pool.manager.layers.items():
            expert = layer.experts[str(args.expert_id)]
            for name in ("lora_A", "lora_B"):
                key = "{}.{}.weight".format(layer_name, name)
                if key not in payload["state_dict"]:
                    raise KeyError("missing weight {}".format(key))
                getattr(expert, name).weight.copy_(
                    payload["state_dict"][key].to(expert.lora_A.weight.dtype)
                )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_expert_checkpoint(pool, str(output))

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    summary = {
        "expert_id": args.expert_id,
        "output_dir": str(output),
        "compose_experts_bin_sha256": sha256(str(output / "compose_experts.bin")),
        "compose_experts_json_sha256": sha256(str(output / "compose_experts.json")),
        "checkpoint_bytes": (output / "compose_experts.bin").stat().st_size,
    }
    (output / "assembly.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
