"""Functional query features (image + instruction only, no answers).

For every sample computes:

    z_v = frozen CLIP visual embedding (L2-normalized)
    z_s = frozen CLIP instruction embedding (L2-normalized)
    q   = ComposeQueryEncoder(z_v, z_s)   # 128-D L2-normalized

Output is ``{sample_id: {"visual_feature": [...], "text_feature": [...],
"query": [...]}}`` plus three provenance hashes:

- ``query_encoder_hash``: the encoder's deterministic parameter hash
  (see ``ComposeQueryEncoder.provenance``),
- ``feature_hash``: hash of the raw visual/text features,
- ``query_hash``: hash of the encoder outputs.

The query encoder is loaded from ``--query-encoder`` when provided
(later tasks reuse the task-0 checkpoint) and created deterministically
from ``--seed`` otherwise; it is never trained or re-randomized here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from compose.data.records import question_text
from compose.eval.sharding import partial_path, shard_records
from compose.router.functional_query import (
    ComposeQueryEncoder,
    load_query_encoder_checkpoint,
)

FEATURE_SCHEMA_VERSION = 1


def query_backbone_provenance(path):
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ValueError("missing query vision model: {}".format(source))
    names = (
        "config.json", "preprocessor_config.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
    )
    files = {}
    for name in names:
        candidate = source / name
        if candidate.is_file():
            files[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if not files:
        raise ValueError("query vision model has no hashable configuration files")
    payload = {"resolved_path": str(source), "configuration_files": files}
    payload["backbone_hash"] = _sha256(payload)
    return payload


def _sample_text(record):
    """Canonical CLIP query text: the question only, placeholder-free.

    Cached queries must be byte-identical to the live test-time routing text
    (compose.data.records.question_text; used by compose.eval.eval_task).
    The raw LLaVA human value embeds a literal <image> placeholder; CLIP
    tokenizes that placeholder as ordinary text and shifts the text embedding
    (measured cosine ~0.94 against the stripped form), putting train/val
    keys, centers and pruning in a different coordinate space than final test
    routing. Delegating to question_text keeps both modes on one contract.
    """
    return question_text(record)


def _sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-encoder", default=None,
                        help="task-0 query encoder checkpoint to reuse")
    parser.add_argument("--query-vision-model", required=True)
    parser.add_argument(
        "--query-mode", choices=("v6_functional", "v7_fixed"),
        default="v6_functional",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    records = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    # 4-GPU execution (spec §15): shard by SAMPLE only. The frozen CLIP
    # features are deterministic per sample (the text embedding is read at
    # the eos position and the visual embedding is per-image), so the
    # merged shards reproduce the single-GPU payload exactly.
    records = shard_records(records, args.num_shards, args.shard_index)
    backbone = query_backbone_provenance(args.query_vision_model)
    clip = CLIPModel.from_pretrained(
        args.query_vision_model, torch_dtype=torch.float16
    ).to(
        torch.device(args.device)
    ).eval()
    processor = CLIPProcessor.from_pretrained(args.query_vision_model)

    # The query encoder is stable across the whole run: load the task-0
    # checkpoint when one exists, otherwise create it deterministically.
    if args.query_mode == "v7_fixed" and args.query_encoder:
        raise ValueError("V7 fixed query cannot load a query-encoder checkpoint")
    if args.query_mode == "v7_fixed":
        from compose.v7.query import FixedMultimodalQuery

        encoder = FixedMultimodalQuery()
    elif args.query_encoder and Path(args.query_encoder).is_file():
        info = load_query_encoder_checkpoint(args.query_encoder)
        encoder = ComposeQueryEncoder(
            visual_dim=int(info["visual_dim"]),
            text_dim=int(info["text_dim"]),
            query_dim=int(info["query_dim"]),
            seed=int(info["init_seed"]),
            initialize=True,
        )
        load_query_encoder_checkpoint(args.query_encoder, encoder)
    else:
        encoder = ComposeQueryEncoder(seed=args.seed)
    if hasattr(encoder, "freeze"):
        encoder.freeze()
    encoder.to(torch.device(args.device)).eval()
    provenance = encoder.provenance()

    records_out = {}
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
            outputs = clip(**inputs)
            z_v = torch.nn.functional.normalize(outputs.image_embeds.float(), dim=-1)
            z_s = torch.nn.functional.normalize(outputs.text_embeds.float(), dim=-1)
            queries = encoder(z_v, z_s)
            for index, sample_id in enumerate(sample_ids):
                records_out[sample_id] = {
                    "visual_feature": z_v[index].cpu().tolist(),
                    "text_feature": z_s[index].cpu().tolist(),
                    "query": queries[index].cpu().tolist(),
                }

    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_source": "frozen_clip_l14_336",
        "query_backbone_provenance": backbone,
        "query_mode": args.query_mode,
        "query_encoder_provenance": provenance.to_dict(),
        "query_encoder_hash": provenance.module_hash,
        "feature_hash": _sha256(
            {
                sample_id: (record["visual_feature"], record["text_feature"])
                for sample_id, record in sorted(records_out.items())
            }
        ),
        "query_hash": _sha256(
            {
                sample_id: record["query"]
                for sample_id, record in sorted(records_out.items())
            }
        ),
        "records": records_out,
    }
    # A sharded worker writes only its partial payload; the orchestrator
    # merges the partials and recomputes the provenance hashes over the
    # full record set (the shard-level hashes cover only this shard).
    target = partial_path(args.output, args.shard_index) if args.num_shards > 1 else args.output
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(
        "features written to {} ({} samples, query_dim={}, shard {}/{})".format(
            target, len(records_out), encoder.query_dim,
            args.shard_index, args.num_shards,
        )
    )


if __name__ == "__main__":
    main()
