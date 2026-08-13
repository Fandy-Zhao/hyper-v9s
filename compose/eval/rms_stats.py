"""Compose RMS statistics subprocess (pipeline stage S9).

Loads the assembled compose checkpoint (full expert pool), runs the
frozen backbone forward over the calibration split (validation only;
``ComposeRMSConfig`` rejects any test split), and collects per-layer
expert delta RMS through forward hooks. Then it builds the runtime
kappa calibration (reference = arithmetic mean over ALL active experts
in the layer, never the expert's own RMS alone) and persists:

- ``rms_statistics.json``  -- raw statistics + provenance,
- ``rms_calibration.json`` -- runtime kappa map, bound to the
  checkpoint/dataset/config hashes,
- ``rms_report.json``      -- raw/calibrated RMS, clip/dominance/pair
  diagnostics.

Finally it patches the assembled checkpoint manifest with the additive
``rms_calibration`` key (JSON only; ``compose_experts.bin`` bytes are
untouched, so the registry-bound checkpoint hash stays valid) and
verifies the bin hash is unchanged. Downstream inference then applies
kappa automatically through ``ComposeLinear.set_expert_calibration``.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from compose.eval.sharding import shard_records

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

from compose.eval.load_compose import load_compose_model
from compose.experts.checkpoint import MANIFEST_NAME, WEIGHTS_NAME
from compose.lora.rms import (
    ComposeRMSConfig,
    build_kappa_calibration,
    build_rms_provenance,
    compute_expert_rms,
    rms_report,
    save_calibration,
    validate_rms_freshness,
    merge_commit_frozen_calibration,
)

CALIBRATION_SPLIT = "validation"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _question_text(record: Dict[str, Any]) -> str:
    if "conversations" in record:
        for message in record["conversations"]:
            if message["from"] == "human":
                return message["value"]
    return record.get("text", "")


def _image_question(text: str) -> str:
    """Ensure exactly one image placeholder per question.

    The UCIT instruction files embed the <image> placeholder in the human
    message themselves (LLaVA v1 convention); prepending a second token
    leaves the batch with more image tokens than images (IndexError in
    prepare_inputs_labels_for_multimodal). Only inject the placeholder for
    questions that lack one; multiple placeholders are malformed data and
    fail loudly here rather than with an opaque index error mid-forward.
    """
    count = text.count(DEFAULT_IMAGE_TOKEN)
    if count > 1:
        raise ValueError(
            "question has {} image placeholders; expected at most one".format(count)
        )
    if count == 1:
        return text
    return DEFAULT_IMAGE_TOKEN + "\n" + text


def _build_batches(
    records: List[Dict[str, Any]],
    bundle,
    image_folder: str,
    device: str,
    batch_size: int,
) -> List[Dict[str, torch.Tensor]]:
    tokenizer = bundle.tokenizer
    conversation = conv_templates["vicuna_v1"].copy()
    batches = []
    for offset in range(0, len(records), batch_size):
        chunk = records[offset: offset + batch_size]
        ids_list = []
        attention_masks = []
        images_list = []
        for record in chunk:
            image_path = os.path.join(image_folder, str(record["image"]))
            if not os.path.isfile(image_path):
                raise ValueError("missing image: {}".format(image_path))
            image = Image.open(image_path).convert("RGB")
            conversation.messages = []
            conversation.append_message(
                conversation.roles[0], _image_question(_question_text(record))
            )
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            image_tensor = process_images(
                [image], bundle.image_processor, bundle.model.config
            )[0].unsqueeze(0)
            ids_list.append(input_ids)
            images_list.append(image_tensor)
        max_length = max(tensor.shape[0] for tensor in ids_list)
        padded_ids = []
        for tensor in ids_list:
            width = max_length - tensor.shape[0]
            padded = torch.cat(
                [tensor, torch.full((width,), tokenizer.pad_token_id, dtype=tensor.dtype)]
            )
            padded_ids.append(padded)
            attention_masks.append(
                torch.ones(max_length, dtype=torch.long)
                if width == 0 else
                torch.cat(
                    [
                        torch.ones(tensor.shape[0], dtype=torch.long),
                        torch.zeros(width, dtype=torch.long),
                    ]
                )
            )
        batches.append(
            {
                "input_ids": torch.stack(padded_ids).to(device),
                "images": torch.stack(images_list).to(device, dtype=torch.bfloat16),
                "attention_mask": torch.stack(attention_masks).to(device),
            }
        )
    return batches


def _init_distributed() -> int:
    """Return the local rank (0 when not launched under torchrun).

    torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; this module initializes the
    process group itself (HF Trainer's accelerate init is not available
    here). Each rank processes a shard of the calibration split, the
    per-layer moments are all-reduced by SUM (``RMSStatistics.all_reduce_``,
    exact fp64 aggregation — spec §17 forbids averaging), and only local
    rank 0 builds the kappa calibration and patches the manifest.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group(backend="nccl")
        return int(os.environ.get("LOCAL_RANK", "0"))
    return 0


def main() -> None:
    local_rank = _init_distributed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="assembled compose expert checkpoint (full pool)")
    parser.add_argument("--question-file", required=True,
                        help="calibration split question file (validation)")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--checkpoint-hash", required=True,
                        help="sha256 of compose_experts.bin (registry binding)")
    parser.add_argument("--dataset-manifest-hash", default="")
    parser.add_argument("--composition-config-hash", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--frozen-calibration", default=None)
    parser.add_argument("--new-expert-ids", default="")
    args = parser.parse_args()

    records = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    if args.max_samples:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("calibration split is empty")
    total_samples = len(records)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Sample-shard per rank (spec §17); the moments all-reduce below
        # makes the aggregation mathematically identical to single-GPU.
        records = shard_records(
            records,
            torch.distributed.get_world_size(),
            local_rank,
        )
        if not records:
            raise ValueError("calibration split empty on rank {}".format(local_rank))
        args.device = "cuda:{}".format(local_rank % torch.cuda.device_count())

    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower,
        projector_path=args.projector_path,
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    expert_ids = sorted(int(value) for value in bundle.expert_pool.expert_ids())
    if not expert_ids:
        raise ValueError("checkpoint has no registered experts")
    config = ComposeRMSConfig(calibration_split=CALIBRATION_SPLIT)
    batches = _build_batches(
        records, bundle, args.image_folder, args.device, args.batch_size
    )
    provenance = build_rms_provenance(
        calibration_split=CALIBRATION_SPLIT,
        checkpoint_hash=args.checkpoint_hash,
        dataset_manifest_hash=args.dataset_manifest_hash,
        composition_config_hash=args.composition_config_hash,
    )
    stats, pair_moments = compute_expert_rms(
        bundle.model,
        expert_ids,
        batches,
        provenance,
        config,
        prepare_batch=lambda batch: batch,
        forward_fn=lambda inputs: bundle.model(**inputs, return_dict=True),
        device=args.device,
    )
    is_rank0 = not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ) or local_rank == 0
    if not is_rank0:
        # Non-root ranks only contributed moments; the kappa calibration,
        # manifest patch and all outputs are written by rank 0 (spec §17).
        # No barrier here: rank 0 is still running its own writes.
        return
    if not validate_rms_freshness(stats, args.checkpoint_hash):
        raise ValueError(
            "RMS statistics invalidated: checkpoint hash {} mismatch".format(
                args.checkpoint_hash
            )
        )
    dynamic_calibration = build_kappa_calibration(stats, expert_ids, config)
    frozen_payload = None
    new_expert_ids = [
        int(value) for value in args.new_expert_ids.split(",") if value.strip()
    ]
    if args.frozen_calibration:
        frozen_payload = json.loads(
            Path(args.frozen_calibration).read_text(encoding="utf-8")
        )
        frozen_map = frozen_payload.get("calibration", frozen_payload)
        calibration = merge_commit_frozen_calibration(
            frozen_map, dynamic_calibration, new_expert_ids
        )
    else:
        calibration = dynamic_calibration
    # Pair cross-term diagnostics cover rank 0's shard only (they feed the
    # diagnostic rms_report.json, never the calibration); the moments
    # themselves were all-reduced across all ranks.
    report = rms_report(stats, expert_ids, pair_moments, config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats.save_json(str(output / "rms_statistics.json"))
    calibration_sha256 = save_calibration(
        calibration, str(output / "rms_calibration.json"), provenance, config
    )
    _atomic_write_json(output / "rms_report.json", report)

    # Patch the assembled checkpoint manifest additively (JSON only);
    # verify the weights bytes are untouched so the registry binding
    # (checkpoint hash) remains valid.
    checkpoint_dir = Path(args.checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    bin_hash = _sha256(str(checkpoint_dir / WEIGHTS_NAME))
    if bin_hash != args.checkpoint_hash:
        raise ValueError(
            "checkpoint bin hash changed before RMS patch: {} != {}".format(
                bin_hash, args.checkpoint_hash
            )
        )
    manifest["rms_calibration"] = {
        layer: {str(expert_id): float(kappa) for expert_id, kappa in layer_map.items()}
        for layer, layer_map in sorted(calibration.items())
    }
    _atomic_write_json(manifest_path, manifest)
    if _sha256(str(checkpoint_dir / WEIGHTS_NAME)) != args.checkpoint_hash:
        raise ValueError("checkpoint bin hash changed after RMS patch")

    summary = {
        "calibration_split": CALIBRATION_SPLIT,
        "expert_ids": expert_ids,
        "samples": total_samples,
        "layers": report["layers"],
        "calibration_sha256": calibration_sha256,
        "checkpoint_hash": args.checkpoint_hash,
        "manifest_patched": True,
        "layers_with_kappa": len(calibration),
        "rms_mode": "commit_frozen",
        "new_expert_ids": new_expert_ids or expert_ids,
        "historical_expert_ids": sorted(set(expert_ids) - set(new_expert_ids)),
        "frozen_calibration_source": args.frozen_calibration,
        "output_dir": str(output),
        "execution": {
            "mode": (
                "4gpu_torchrun"
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else "single_gpu"
            ),
            "world_size": (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            ),
            "pair_diagnostics_shard": "rank0_only",
            "aggregation": "sum_all_reduce_fp64",
        },
    }
    _atomic_write_json(output / "rms_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
