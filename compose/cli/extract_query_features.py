#!/usr/bin/env python3
"""Extract frozen CLIP image/question features without encoding answer text."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from compose.data.records import question_text


CLIP_PATH = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id(row, fallback):
    return str(row.get("question_id", row.get("id", fallback)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--oracle-cache", required=True)
    parser.add_argument("--post-task-oracle-cache", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    cache = json.loads(Path(args.oracle_cache).read_text(encoding="utf-8"))
    post_cache = json.loads(Path(args.post_task_oracle_cache).read_text(encoding="utf-8"))
    oracle = cache["records"]
    post_oracle = post_cache["records"]
    if len(rows) != len(oracle) or len(rows) != len(post_oracle) or cache["provenance"]["composition_mode"] != "direct_sum" or post_cache["provenance"]["composition_mode"] != "direct_sum":
        raise ValueError("question/Direct Oracle cache mismatch")
    if cache["provenance"]["split"] != args.split:
        raise ValueError("Oracle split mismatch")
    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    image_features, text_features = [], []
    prompt_hashes, image_hashes = [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            prompts = [question_text(row) for row in batch]
            image_paths = [os.path.join(args.images, str(row["image"])) for row in batch]
            images = [Image.open(path).convert("RGB") for path in image_paths]
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            image_inputs = processor(images=images, return_tensors="pt")
            text = model.get_text_features(**{key: value.to(device) for key, value in text_inputs.items()}).float().cpu()
            image = model.get_image_features(**{key: value.to(device) for key, value in image_inputs.items()}).float().cpu()
            text_features.append(torch.nn.functional.normalize(text, dim=-1))
            image_features.append(torch.nn.functional.normalize(image, dim=-1))
            prompt_hashes.extend(hashlib.sha256(value.encode()).hexdigest() for value in prompts)
            image_hashes.extend(sha256(path) for path in image_paths)
    payload = {
        "schema_version": 1, "feature_source": "frozen_clip_image_and_question_only", "clip_path": CLIP_PATH,
        "test_data_used": False, "task_id": args.task_id, "task_name": args.task_name, "split": args.split,
        "sample_ids": [source_id(row, index) for index, row in enumerate(rows)],
        "image_features": torch.cat(image_features), "text_features": torch.cat(text_features),
        "image_available": torch.ones(len(rows)), "text_available": torch.ones(len(rows)),
        "prompt_hashes": prompt_hashes, "image_hashes": image_hashes,
        "oracle_sets": [tuple(record["selected_expert_ids"]) for record in oracle],
        "post_task_oracle_sets": [tuple(record["selected_expert_ids"]) for record in post_oracle],
        "oracle_records": oracle, "dataset_manifest_hash": sha256(args.questions),
        "oracle_cache_hash": sha256(args.oracle_cache), "post_task_oracle_cache_hash": sha256(args.post_task_oracle_cache),
    }
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, target)
    print(json.dumps({"status": "EXTRACTED", "samples": len(rows), "split": args.split, "task": args.task_name, "answer_features_used": False}))


if __name__ == "__main__":
    main()
