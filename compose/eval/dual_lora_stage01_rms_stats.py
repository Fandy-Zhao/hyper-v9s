"""Stage 01: per-layer delta-RMS calibration statistics for one expert pair.

Computes the formal Compose RMS-calibrated combination statistics on the
VAL split of the pair's dataset (never test): for every ComposeLinear layer
and every active expert, the RMS of the expert's LoRA delta on the val
hidden states, plus the frozen per-layer kappa coefficients.

The provenance hash binds the statistics to the exact checkpoint, val-split
question file and composition rule, so loading fails loudly if any of them
change.

Usage:
  python -m compose.eval.dual_lora_stage01_rms_stats --pair a_independent_b \
      --device cuda:4 --output-root artifacts/dual_lora_stage01
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict

import torch

from compose.adapters.lora import ComposeLinear
from compose.eval.load_compose import load_compose_model
from compose.lora.rms_composition import RMSCompositionConfig, frozen_coefficients
from compose.lora.statistics import RMSStatistics, StatisticKey, stable_hash
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch

BATCH_SIZE = 8

PAIRS = {
    "a_independent_b": ("assembled/seed42/a_independent_b", "A_plus_B", (0, 1)),
    "a_residual_b": ("seed42/residual_b", "A_plus_B", (0, 1)),
    "independent_b_c": ("assembled/seed42/independent_b_c", "B_plus_C", (1, 2)),
    "residual_b_c": ("assembled/seed42/residual_b_c", "B_plus_C", (1, 2)),
}
CHECKPOINT_ROOT = "/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1"
DATA_ROOT = "experiments/data/controlled_format_v1_training/instructions"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--projector-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="artifacts/dual_lora_stage01")
    args = parser.parse_args()

    checkpoint_name, dataset, pair_ids = PAIRS[args.pair]
    checkpoint_dir = os.path.join(args.checkpoint_root, checkpoint_name)
    image_folder = "experiments/data/controlled_format_v1"
    val_file = os.path.join(DATA_ROOT, dataset, "val.json")
    with open(val_file, encoding="utf-8") as handle:
        records = json.load(handle)
    checkpoint_hash = sha256_file(os.path.join(checkpoint_dir, "compose_experts.bin"))
    dataset_hash = sha256_file(val_file)
    config = RMSCompositionConfig()
    config_hash = stable_hash({
        "pair_scale": config.pair_scale, "coefficient_min": config.coefficient_min,
        "coefficient_max": config.coefficient_max, "expert_scalars": list(config.expert_scalars),
        "composition_rule": "frozen_arithmetic_mean_rms",
    })
    provenance = {
        "calibration_split": "val", "checkpoint_hash": checkpoint_hash,
        "dataset_manifest_hash": dataset_hash, "composition_config_hash": config_hash,
    }

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16, model_max_length=2048,
    )
    stats = RMSStatistics(provenance)
    layer_by_name: Dict[str, ComposeLinear] = {
        name: module for name, module in bundle.model.named_modules()
        if isinstance(module, ComposeLinear)
    }
    module_types = {name: name.rsplit(".", 1)[-1] for name in layer_by_name}

    # Per-layer deltas are consumed inside the forward hook (post-activation)
    # so at most one layer's hidden states are alive at any time: keeping all
    # 224 captured inputs simultaneously would exceed GPU memory.
    def make_post_hook(name: str, module: ComposeLinear):
        def post_hook(module, inputs, output):
            hidden = inputs[0]
            for expert_id in pair_ids:
                delta = module.experts[str(expert_id)](hidden).to(torch.bfloat16)
                key = StatisticKey(
                    expert_id=expert_id, layer_name=name, module_name=name,
                    target_module_type=module_types[name],
                )
                stats.update(key, delta.detach(), delta.detach(), None)
        return post_hook

    hooks = []
    for name, module in layer_by_name.items():
        hooks.append(module.register_forward_hook(make_post_hook(name, module)))

    bundle.expert_pool.manager.set_default_selection(list(pair_ids), [1.0, 1.0], normalization="none")
    try:
        with torch.inference_mode():
            for offset in range(0, len(records), BATCH_SIZE):
                batch_records = records[offset: offset + BATCH_SIZE]
                raw = _collate(batch_records, bundle, image_folder, args.device)
                prepared = _prepare_multimodal_batch(bundle, raw)
                bundle.model(**prepared)
    finally:
        for hook in hooks:
            hook.remove()

    # Per-layer frozen kappa (formal rule) for the report.
    kappa_rows = []
    for name in sorted(layer_by_name):
        rms_a = stats.delta_rms(pair_ids[0], name)
        rms_b = stats.delta_rms(pair_ids[1], name)
        kappa, audit = frozen_coefficients(rms_a, rms_b, config)
        kappa_rows.append({"layer": name, "rms_A": rms_a, "rms_B": rms_b,
                           "kappa_A": kappa[0], "kappa_B": kappa[1],
                           "clipped": audit["clipped"]})

    output_root = Path(args.output_root) / "rms_stats"
    output_root.mkdir(parents=True, exist_ok=True)
    stats.save_json(str(output_root / (args.pair + ".json")))
    summary = stats.summary()
    summary["kappa"] = kappa_rows
    with (output_root / (args.pair + "_kappa.json")).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "pair": args.pair, "dataset": dataset, "val_samples": len(records),
        "layers": len(layer_by_name), "provenance": provenance,
        "kappa_mean_A": sum(r["kappa_A"] for r in kappa_rows) / len(kappa_rows),
        "kappa_mean_B": sum(r["kappa_B"] for r in kappa_rows) / len(kappa_rows),
        "saved": str(output_root / (args.pair + ".json")),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
