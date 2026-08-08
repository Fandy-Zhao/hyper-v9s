#!/usr/bin/env python3
"""Single-expert per-layer removal ablation (L3 layer-group selection).

For each LoRA layer of one expert, evaluate the expert on its own val set
with that layer's delta zeroed; layers whose removal increases answer-token
NLL (or hurts accuracy) are the expert's "important" layers. The kept sets
are emitted as a layer-mask JSON for the L0-L3 diagnostics.

Usage:
  python -m compose.eval.ablate_layers_real \
      --checkpoint <assembled pair> --expert-id 1 \
      --questions <B_val.json> --images <data root> \
      --output <ablation.json> --layer-masks <out_masks.json>
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import AdapterBridge
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expert-id", type=int, required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-masks", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-threshold", type=float, default=0.0,
                        help="min NLL-removal gain (nats) to keep a layer")
    parser.add_argument("--block-size", type=int, default=1,
                        help="ablate blocks of N consecutive layers instead of "
                             "single layers (faster; kept layers = whole blocks)")
    args = parser.parse_args()
    expert_id = args.expert_id

    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower,
        projector_path=str(Path(args.model_path) / "mm_projector.bin"),
        expert_id=expert_id,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    records = json.loads(Path(args.questions).read_text(encoding="utf-8"))[: args.max_samples]
    for record in records:
        if "text" not in record and "question" in record:
            record["text"] = record["question"]

    def evaluate(masked_layers=None):
        masked_layers = masked_layers or set()
        hooks = []
        for layer_name, module in bridge.named_layers:
            if layer_name not in masked_layers:
                continue

            def hook(current_module, inputs, base_output, name=layer_name):
                if not inputs:
                    return base_output
                # remove this layer's expert delta: post-hooks see the
                # module output (base + delta), so subtract the delta back
                delta = bridge.compute_expert_delta(
                    current_module, expert_id, inputs[0]).to(base_output.dtype)
                return base_output - delta

            hooks.append(module.register_forward_hook(hook))
        nlls = []
        with torch.inference_mode():
            for offset in range(0, len(records), args.batch_size):
                batch = records[offset: offset + args.batch_size]
                prepared = _prepare_multimodal_batch(
                    bundle, _collate(batch, bundle, args.images, args.device))
                logits = bundle.model(**prepared).logits
                labels = prepared["labels"]
                for index, record in enumerate(batch):
                    supervised = torch.where(labels[index].ne(-100))[0]
                    positions = supervised.tolist()
                    gold = labels[index, supervised].tolist()
                    predicting = torch.tensor([p - 1 for p in positions], device=logits.device)
                    nlls.append(float(torch.nn.functional.cross_entropy(
                        logits[index, predicting],
                        torch.tensor(gold, device=logits.device)).item()))
        for hook in hooks:
            hook.remove()
        return statistics.fmean(nlls) if nlls else float("inf")

    # baseline full-expert NLL
    baseline = evaluate()
    names = [name for name, _ in bridge.named_layers]
    results = []
    if args.block_size <= 1:
        targets = [(name, name) for name in names]
    else:
        targets = [(name, names[min(i + args.block_size - 1, len(names) - 1)])
                   for i, name in enumerate(names[:: args.block_size])]
    for first, last in targets:
        removed = evaluate(masked_layers={first, last} if first == last else set(
            names[names.index(first): names.index(last) + 1]))
        results.append({
            "layer_name": first,
            "block_last": last,
            "baseline_nll": baseline,
            "removed_nll": removed,
            "removal_gain": baseline - removed,  # positive => block helps
        })
    kept = []
    for r in results:
        block = set(names[names.index(r["layer_name"]): names.index(r["block_last"]) + 1])
        if r["removal_gain"] >= args.keep_threshold:
            kept.extend(sorted(block))
    Path(args.output).write_text(json.dumps({
        "expert_id": expert_id,
        "baseline_nll": baseline,
        "samples": len(records),
        "keep_threshold": args.keep_threshold,
        "layers": results,
        "kept_layers": kept,
    }, indent=2, sort_keys=True), encoding="utf-8")

    # layer masks file: full expert on kept layers, zero elsewhere
    masks = {}
    for layer_name, _ in bridge.named_layers:
        key = "B" if expert_id == 1 else "C"
        masks[layer_name] = {other: 0 for other in ("B", "C")}
        if layer_name in kept:
            masks[layer_name][key] = 1
    Path(args.layer_masks).write_text(json.dumps(masks, indent=1), encoding="utf-8")
    print(json.dumps({
        "expert_id": expert_id,
        "baseline_nll": baseline,
        "kept_layers": len(kept),
        "total_layers": len(results),
    }))


if __name__ == "__main__":
    main()
