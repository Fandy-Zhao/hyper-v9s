import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

from compose.data.records import question_text

from .load_compose import load_compose_model, load_peft_model


def _chunk(records: List[Dict[str, object]], count: int, index: int):
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid chunk selection {}/{}".format(index, count))
    size = int(math.ceil(len(records) / count))
    return records[index * size : (index + 1) * size]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prompt(record, model_config, conv_mode):
    question = question_text(record)
    # The UCIT instruction files embed the <image> placeholder in the human
    # message themselves (LLaVA v1 convention); prepending another token
    # leaves the batch with more image tokens than images (IndexError in
    # prepare_inputs_labels_for_multimodal). Only inject the placeholder for
    # records that lack one (mm_use_im_start_end wrapper included).
    if DEFAULT_IMAGE_TOKEN not in question:
        image_token = DEFAULT_IMAGE_TOKEN
        if model_config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        question = image_token + "\n" + question
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-kind", choices=("compose", "peft"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--run-summary-file", required=True)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--expert-ids")
    parser.add_argument("--gates")
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("none", "l1", "l2"), default="none")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument(
        "--router-checkpoint", default=None,
        help="compose router checkpoint; when given, selection is per-sample "
             "ComposeRouter.select() (frozen query encoder + expert keys) "
             "instead of a fixed --expert-ids/--gates",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.time()
    common = dict(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower,
        projector_path=args.projector_path,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=args.model_max_length,
    )
    if args.adapter_kind == "compose":
        if args.expert_ids is None and args.router_checkpoint is None:
            bundle = load_compose_model(
                expert_id=args.expert_id,
                gate=args.gate,
                normalization=args.normalization,
                **common
            )
        else:
            bundle = load_compose_model(expert_id=None, **common)
            # --router-checkpoint alone means per-sample selection: no fixed
            # expert ids (None), and the default gates follow.
            expert_ids = (
                [int(value.strip()) for value in args.expert_ids.split(",") if value.strip()]
                if args.expert_ids is not None
                else []
            )
            gates = (
                [float(value.strip()) for value in args.gates.split(",") if value.strip()]
                if args.gates is not None
                else [1.0] * len(expert_ids)
            )
            if len(expert_ids) > 4 or len(gates) != len(expert_ids):
                raise ValueError("explicit selection requires zero through four experts")
            if expert_ids:
                bundle.expert_pool.manager.set_default_selection(
                    expert_ids, gates, normalization=args.normalization
                )
            else:
                bundle.expert_pool.manager.clear_default_selection()
            bundle.load_summary["evaluation_selection"] = {
                "expert_ids": expert_ids,
                "gates": gates,
                "normalization": args.normalization,
            }
    else:
        bundle = load_peft_model(**common)

    # Runtime kappa calibration, when the checkpoint carries one.
    calibration = bundle.load_summary.get("rms_calibration")
    if calibration:
        from compose.lora.rms import apply_kappa_calibration

        apply_kappa_calibration(bundle.model, calibration)
        bundle.load_summary["rms_calibration_applied"] = True

    # Router-based inference: per-sample ComposeRouter.select() with the
    # frozen query encoder and the expert keys; no answers, no oracle,
    # no task-id lookup, no clustering at test time (spec §22).
    router = None
    clip = None
    clip_processor = None
    if args.router_checkpoint:
        # --expert-id defaults to 0 and is a legacy fixed-selection flag; it
        # is unused (and ignored) in router mode. Only an explicit fixed
        # --expert-ids list conflicts with per-sample router selection.
        if args.expert_ids is not None:
            raise ValueError(
                "--router-checkpoint cannot be combined with fixed --expert-ids"
            )
        from transformers import CLIPModel, CLIPProcessor

        from compose.router.router import ComposeRouter, load_compose_router_checkpoint
        from compose.router.functional_query import load_query_encoder_checkpoint

        router = ComposeRouter()
        router_extra = load_compose_router_checkpoint(args.router_checkpoint, router)
        # Move the whole router (frozen query encoder AND expert keys) to the
        # device: load_state_dict_extra rebuilds the key ParameterDict on CPU
        # (map_location="cpu"), and select() matmuls query @ keys.T.
        router.to(torch.device(args.device)).eval()
        clip = CLIPModel.from_pretrained(
            "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336",
            torch_dtype=torch.float16,
        ).to(torch.device(args.device)).eval()
        clip_processor = CLIPProcessor.from_pretrained(
            "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
        )
        bundle.expert_pool.manager.clear_default_selection()
        bundle.load_summary["evaluation_selection"] = {
            "mode": "compose_router",
            "router_checkpoint": args.router_checkpoint,
            "visible_expert_ids": list(router.expert_ids),
            "router_pool_version": router_extra.get("pool_version"),
        }

    with open(args.question_file, "r", encoding="utf-8") as handle:
        all_records = json.load(handle)
    records = _chunk(all_records, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    with open(args.answers_file, "w", encoding="utf-8") as output:
        selection_histogram = {0: 0, 1: 0, 2: 0}
        for record in tqdm(records):
            prompt = _prompt(record, bundle.model.config, args.conv_mode)
            input_ids = tokenizer_image_token(
                prompt, bundle.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(args.device)
            image = Image.open(
                os.path.join(args.image_folder, str(record["image"]))
            ).convert("RGB")
            image_tensor = process_images(
                [image], bundle.image_processor, bundle.model.config
            )[0].unsqueeze(0).to(device=args.device, dtype=torch.bfloat16)
            selection_meta = None
            if router is not None:
                clip_inputs = clip_processor(
                    text=[question_text(record)],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(torch.device(args.device))
                with torch.inference_mode():
                    clip_outputs = clip(**clip_inputs)
                    z_v = torch.nn.functional.normalize(
                        clip_outputs.image_embeds.float(), dim=-1
                    )
                    z_s = torch.nn.functional.normalize(
                        clip_outputs.text_embeds.float(), dim=-1
                    )
                    query = router.query_encoder(z_v, z_s)
                    selection = router.select(query, router.expert_ids)
                ids = list(selection.sets[0])
                if ids:
                    bundle.expert_pool.manager.set_default_selection(
                        ids, [1.0] * len(ids)
                    )
                else:
                    bundle.expert_pool.manager.clear_default_selection()
                selection_histogram[len(ids)] = selection_histogram.get(len(ids), 0) + 1
                selection_meta = {
                    "expert_ids": ids,
                    "selection_source": "compose_router",
                    "answer_features_used": bool(selection.answer_features_used),
                    "oracle_used": bool(selection.oracle_used),
                    "task_id_lookup_used": bool(selection.task_id_lookup_used),
                    "clustering_used_at_test": bool(selection.clustering_used_at_test),
                }
            with torch.inference_mode():
                output_ids = bundle.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            input_length = input_ids.shape[1]
            text = bundle.tokenizer.batch_decode(
                output_ids[:, input_length:], skip_special_tokens=True
            )[0].strip()
            output.write(
                json.dumps(
                    {
                        "question_id": str(
                            record.get("question_id", record.get("id"))
                        ),
                        "prompt": question_text(record),
                        "text": text,
                        "model_id": args.adapter_kind,
                        "metadata": {
                            "checkpoint": args.checkpoint_dir,
                            "git_commit": _git_commit(),
                            "selection": selection_meta,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    duration = time.time() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(torch.device(args.device))
        if torch.cuda.is_available()
        else 0
    )
    summary = {
        "adapter_kind": args.adapter_kind,
        "checkpoint": args.checkpoint_dir,
        "command": sys.argv,
        "git_commit": _git_commit(),
        "seed": 42,
        "samples": len(records),
        "duration_seconds": duration,
        "samples_per_second": len(records) / duration if duration else 0.0,
        "peak_memory_bytes": peak_memory,
        "load_summary": bundle.load_summary,
        "selection_mode": (
            "compose_router" if router is not None else "fixed"
        ),
        "router_selection_histogram": (
            selection_histogram if router is not None else None
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.run_summary_file)), exist_ok=True)
    with open(args.run_summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
