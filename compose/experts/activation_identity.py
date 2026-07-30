"""Compare residual-expert hidden-delta directions across controlled tasks."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from llava import conversation as conversation_lib
from llava.conversation import conv_templates

from compose.eval.load_compose import load_compose_model
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch


def vector_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = float(first.norm() * second.norm())
    if denominator == 0:
        return 0.0
    return float(torch.dot(first.flatten(), second.flatten()) / denominator)


def _signature(bundle, records, image_folder: str, expert_id: int, device: str):
    sums = {}
    counts = {}
    hooks = []
    for layer_name, layer in bundle.expert_pool.manager.layers.items():
        expert = layer.experts[str(expert_id)]

        def capture(_module, _inputs, output, name=layer_name):
            dimensions = tuple(range(output.ndim - 1))
            value = output.detach().float().sum(dim=dimensions)
            sums[name] = sums.get(name, torch.zeros_like(value)) + value
            count = 1
            for size in output.shape[:-1]:
                count *= size
            counts[name] = counts.get(name, 0) + count

        hooks.append(expert.register_forward_hook(capture))
    try:
        for record in records:
            raw = _collate([record], bundle, image_folder, device)
            prepared = _prepare_multimodal_batch(bundle, raw)
            with torch.inference_mode():
                bundle.model(**prepared)
    finally:
        for hook in hooks:
            hook.remove()
    return {name: (value / counts[name]).cpu() for name, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--expert-id", type=int, required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError("refusing existing output: {}".format(output))
    conversation_lib.default_conversation = conv_templates["vicuna_v1"]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bundle.expert_pool.manager.set_default_selection(
        [args.expert_id], [1.0], normalization="none"
    )
    signatures = {}
    for dataset in ("B_only", "A_plus_B", "B_plus_C"):
        path = Path(args.data_root) / "instructions" / dataset / "test.json"
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)[:args.samples_per_dataset]
        signatures[dataset] = _signature(
            bundle, records, args.image_folder, args.expert_id, args.device
        )
    if not all(set(value) == set(signatures["B_only"]) for value in signatures.values()):
        raise AssertionError("activation-signature layer sets differ")
    rows = []
    for layer in sorted(signatures["B_only"]):
        b = signatures["B_only"][layer]
        ab = signatures["A_plus_B"][layer]
        bc = signatures["B_plus_C"][layer]
        rows.append({
            "layer": layer,
            "B_only_rms": float(b.square().mean().sqrt()),
            "A_plus_B_rms": float(ab.square().mean().sqrt()),
            "B_plus_C_rms": float(bc.square().mean().sqrt()),
            "cosine_B_only_A_plus_B": vector_cosine(b, ab),
            "cosine_B_only_B_plus_C": vector_cosine(b, bc),
            "cosine_A_plus_B_B_plus_C": vector_cosine(ab, bc),
        })
    cosine_fields = (
        "cosine_B_only_A_plus_B", "cosine_B_only_B_plus_C",
        "cosine_A_plus_B_B_plus_C",
    )
    result = {
        "checkpoint": args.checkpoint_dir,
        "expert_id": args.expert_id,
        "samples_per_dataset": args.samples_per_dataset,
        "layers": rows,
        "summary": {
            field: {
                "mean": statistics.fmean(row[field] for row in rows),
                "median": statistics.median(row[field] for row in rows),
            }
            for field in cosine_fields
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
