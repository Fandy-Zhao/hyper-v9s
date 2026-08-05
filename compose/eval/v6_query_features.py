"""V6 frozen CLIP query features (image + question only, no answers) for
candidate key initialization and Router input (Stage E12).

Outputs ``{sample_id: {"image": [768 floats], "text": [768 floats]}}`` with
both modalities L2-normalized, plus a feature-set hash for provenance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_PATH = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"


def _sample_text(record):
    if "conversations" in record:
        for message in record["conversations"]:
            if message["from"] == "human":
                return message["value"]
    return record.get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    records = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    model = CLIPModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float16).to(
        torch.device(args.device)
    ).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_PATH)

    result = {}
    with torch.inference_mode():
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset: offset + args.batch_size]
            images = []
            texts = []
            sample_ids = []
            for record in batch:
                image_path = os.path.join(args.images, record["image"])
                if not os.path.isfile(image_path):
                    raise ValueError("missing image: {}".format(image_path))
                images.append(Image.open(image_path).convert("RGB"))
                texts.append(_sample_text(record))
                sample_ids.append(str(record.get("id", record.get("question_id"))))
            inputs = processor(
                text=texts, images=images, return_tensors="pt",
                padding=True, truncation=True,
            ).to(torch.device(args.device))
            outputs = model(**inputs)
            image_feats = torch.nn.functional.normalize(
                outputs.image_embeds.float(), dim=-1
            )
            text_feats = torch.nn.functional.normalize(
                outputs.text_embeds.float(), dim=-1
            )
            for index, sample_id in enumerate(sample_ids):
                result[sample_id] = {
                    "image": image_feats[index].cpu().tolist(),
                    "text": text_feats[index].cpu().tolist(),
                }

    payload = {
        "schema_version": 1,
        "feature_source": "frozen_clip_l14_336",
        "records": result,
        "feature_hash": hashlib.sha256(
            json.dumps(result, sort_keys=True).encode()
        ).hexdigest(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("features written to {} ({} samples)".format(args.output, len(result)))


if __name__ == "__main__":
    main()
