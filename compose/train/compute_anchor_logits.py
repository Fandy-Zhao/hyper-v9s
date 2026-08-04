"""Stage 04 B2: precompute frozen B-only anchor logits.

Runs the seed B expert (independent_b checkpoint) over the B_only TRAIN
split and saves, per sample, the full-vocabulary logits at the answer-token
position (the position predicted by logits[answer_pos - 1], matching the
training shift). Used as the L_preserve anchor during B2 compatibility
training.

Output: anchor_logits.pt dict {sample_id: fp32 tensor (vocab,)}.

Usage:
  python -m compose.train.compute_anchor_logits --checkpoint <independent_b> \
      --question-file experiments/data/controlled_format_v1_training/instructions/B_only/train.json \
      --output artifacts/dual_lora_stage04/anchor_b_only.pt
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from compose.eval.load_compose import load_compose_model
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch

BATCH_SIZE = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", default="experiments/data/controlled_format_v1")
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--projector-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=1, device=args.device, dtype=torch.bfloat16, model_max_length=2048,
    )
    with open(args.question_file, encoding="utf-8") as handle:
        records = json.load(handle)
    anchors = {}
    with torch.inference_mode():
        for offset in range(0, len(records), BATCH_SIZE):
            batch_records = records[offset: offset + BATCH_SIZE]
            raw = _collate(batch_records, bundle, args.image_folder, args.device)
            prepared = _prepare_multimodal_batch(bundle, raw)
            logits = bundle.model(**prepared).logits
            labels = prepared["labels"]
            for index, record in enumerate(batch_records):
                label_row = labels[index].tolist()
                positions = [i for i, value in enumerate(label_row) if value != -100]
                answer_pos = positions[0] - 1
                anchors[str(record["question_id"])] = logits[index, answer_pos].float().cpu()
    assert len(anchors) == len(records), "anchor count mismatch"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchors, str(output))
    print(json.dumps({
        "samples": len(anchors), "output": str(output),
        "checkpoint": args.checkpoint,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
