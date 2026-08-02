#!/usr/bin/env python3
"""Produce answer-free per-sample 0/1/2 expert decisions for UCIT test data."""

import argparse
import fcntl
import hashlib
import json
import os
import time
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from compose.data.records import question_text
from compose.router.checkpoint import load_router_checkpoint, load_set_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.inference import predict_sets
from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs
from compose.router.set_router import ExpertSetRouter


CLIP_PATH = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--query-key-checkpoint", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--visible-experts", required=True,
                        help="Comma-separated deployment-visible expert IDs; never inferred from sample task identity.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-cache", help="Optional answer-free frozen CLIP feature cache on /data.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    visible_ids = tuple(int(value) for value in args.visible_experts.split(",") if value != "")
    if len(visible_ids) != len(set(visible_ids)):
        raise ValueError("deployment-visible experts contain duplicates")
    device = torch.device(args.device)
    raw = torch.load(args.query_key_checkpoint, map_location="cpu")
    metadata = [ExpertKeyMetadata(**item) for item in raw["key_metadata"]["experts"]]
    keys = ExpertKeyStore(metadata).to(device)
    encoder = MultimodalQueryEncoder(768, 768).to(device)
    load_router_checkpoint(args.query_key_checkpoint, encoder, keys, map_location=device)
    missing = sorted(set(visible_ids) - set(keys.expert_ids))
    if missing:
        raise ValueError(f"deployment-visible experts are absent from key store: {missing}")
    router = ExpertSetRouter().to(device)
    extra = load_set_router_checkpoint(args.router_checkpoint, router, map_location=device)
    threshold = float(extra["pair_threshold"])
    encoder.eval(); keys.eval(); router.eval()

    manifest_hash = _sha256(args.questions)
    cached = None
    lock_handle = None
    if args.feature_cache:
        cache_path = Path(args.feature_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = open(str(cache_path) + ".lock", "a+", encoding="utf-8")
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            expected_ids = [str(row.get("question_id", row.get("id", index))) for index, row in enumerate(rows)]
            if cached.get("question_manifest_sha256") != manifest_hash or cached.get("sample_ids") != expected_ids:
                raise ValueError("test feature cache provenance mismatch")
    if cached is None:
        clip = CLIPModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float16).to(device).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_PATH)
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                prompts = [question_text(row) for row in batch]
                images = [Image.open(os.path.join(args.images, str(row["image"]))).convert("RGB") for row in batch]
                text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
                image_inputs = processor(images=images, return_tensors="pt")
                all_text_features.append(torch.nn.functional.normalize(
                    clip.get_text_features(**{key: value.to(device) for key, value in text_inputs.items()}).float(), dim=-1).cpu())
                all_image_features.append(torch.nn.functional.normalize(
                    clip.get_image_features(**{key: value.to(device) for key, value in image_inputs.items()}).float(), dim=-1).cpu())
        cached = {"schema_version": 1, "feature_source": "frozen_clip_image_and_question_only",
                  "question_manifest_sha256": manifest_hash,
                  "sample_ids": [str(row.get("question_id", row.get("id", index))) for index, row in enumerate(rows)],
                  "image_features": torch.cat(all_image_features), "text_features": torch.cat(all_text_features),
                  "answer_features_used": False, "oracle_used": False, "task_id_lookup_used": False}
        if args.feature_cache:
            descriptor, temporary = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".tmp", dir=str(cache_path.parent))
            os.close(descriptor)
            try:
                torch.save(cached, temporary)
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    if lock_handle is not None:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
    decisions = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            text_features = cached["text_features"][offset:offset + len(batch)].to(device)
            image_features = cached["image_features"][offset:offset + len(batch)].to(device)
            queries = encoder(QueryInputs(image_features, text_features,
                                          torch.ones(len(batch), device=device),
                                          torch.ones(len(batch), device=device)))
            visible = torch.tensor([[expert_id in visible_ids for expert_id in keys.expert_ids]] * len(batch),
                                   dtype=torch.bool, device=device)
            output = router(queries, keys.normalized(), keys.expert_ids, visible)
            predictions = predict_sets(output, threshold)
            for index, (row, prediction) in enumerate(zip(batch, predictions)):
                if not set(prediction).issubset(visible_ids) or len(prediction) > 2:
                    raise RuntimeError("router emitted an invalid deployment selection")
                decisions.append({
                    "question_id": str(row.get("question_id", row.get("id", offset + index))),
                    "expert_ids": list(prediction),
                    "cardinality": len(prediction),
                })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "routes": decisions,
        "visible_expert_ids": list(visible_ids),
        "pair_threshold": threshold,
        "question_manifest_sha256": manifest_hash,
        "feature_cache": str(args.feature_cache) if args.feature_cache else None,
        "query_key_checkpoint_sha256": _sha256(args.query_key_checkpoint),
        "router_checkpoint_sha256": _sha256(args.router_checkpoint),
        "feature_source": "frozen_clip_image_and_question_only",
        "oracle_used": False,
        "answer_features_used": False,
        "task_id_lookup_used": False,
        "route_seconds_per_sample": elapsed / max(1, len(rows)),
        "route_throughput_samples_per_second": len(rows) / max(elapsed, 1e-12),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ROUTED", "samples": len(rows), "rates": {
        str(size): sum(len(item["expert_ids"]) == size for item in decisions) / max(1, len(decisions))
        for size in range(3)}, "answer_features_used": False, "oracle_used": False,
        "task_id_lookup_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
