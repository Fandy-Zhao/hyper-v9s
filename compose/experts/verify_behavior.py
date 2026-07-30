"""Verify frozen-expert teacher-forcing logits across two checkpoints."""

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.conversation import conv_templates

from compose.eval.load_compose import load_compose_model
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fixed_target_logits(args, checkpoint_dir: str, records):
    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bundle.expert_pool.manager.set_default_selection(
        [args.expert_id], [1.0], normalization="none"
    )
    raw = _collate(records, bundle, args.image_folder, args.device)
    prepared = _prepare_multimodal_batch(bundle, raw)
    with torch.inference_mode():
        logits = bundle.model(**prepared).logits[:, :-1]
    mask = prepared["labels"][:, 1:].ne(IGNORE_INDEX)
    selected = logits[mask].detach().cpu().contiguous()
    del logits, prepared, raw, bundle
    gc.collect()
    torch.cuda.empty_cache()
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--before-checkpoint", required=True)
    parser.add_argument("--after-checkpoint", required=True)
    parser.add_argument("--expert-id", type=int, required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError("refusing existing output: {}".format(output))
    with open(args.question_file, encoding="utf-8") as handle:
        records = json.load(handle)[:args.sample_count]
    conversation_lib.default_conversation = conv_templates["vicuna_v1"]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    before = _fixed_target_logits(args, args.before_checkpoint, records)
    after = _fixed_target_logits(args, args.after_checkpoint, records)
    if before.shape != after.shape:
        raise AssertionError("fixed-logit shapes differ")
    exact = torch.equal(before, after)
    result = {
        "before_checkpoint": args.before_checkpoint,
        "after_checkpoint": args.after_checkpoint,
        "expert_id": args.expert_id,
        "sample_ids": [str(record["question_id"]) for record in records],
        "target_position_count": before.shape[0],
        "vocabulary_size": before.shape[1],
        "dtype": str(before.dtype),
        "before_sha256": tensor_sha256(before),
        "after_sha256": tensor_sha256(after),
        "exactly_equal": exact,
        "max_absolute_difference": float((before.float() - after.float()).abs().max()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    if not exact:
        raise AssertionError("frozen expert fixed-input logits changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
