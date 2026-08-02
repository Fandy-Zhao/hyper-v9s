#!/usr/bin/env python3
"""Generate UCIT answers under audited per-sample Compose route decisions."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from compose.experts import ExpertMetadata, ExpertRegistry, ExpertStatus
from compose.lora import AdapterBridge, CompositionRuntime, ExpertComposer
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def _chunks(values, count):
    size = math.ceil(len(values) / count)
    return [values[index:index + size] for index in range(0, len(values), size)]


class RoutedDataset(Dataset):
    def __init__(self, rows, image_folder, tokenizer, processor, config, conv_mode):
        self.rows, self.image_folder = rows, image_folder
        self.tokenizer, self.processor, self.config, self.conv_mode = tokenizer, processor, config, conv_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        question = row["text"]
        prefix = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if self.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        conversation = conv_templates[self.conv_mode].copy()
        conversation.append_message(conversation.roles[0], prefix + "\n" + question)
        conversation.append_message(conversation.roles[1], None)
        image = Image.open(os.path.join(self.image_folder, row["image"])).convert("RGB")
        image_tensor = process_images([image], self.processor, self.config)[0]
        input_ids = tokenizer_image_token(conversation.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor


def _registry(bridge, checkpoint):
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        digest = hashlib.sha256(f"{checkpoint}:{expert_id}".encode()).hexdigest()
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=bridge.adapter_name,
                                         rank=8, alpha=96, status=ExpertStatus.FROZEN,
                                         creation_task=expert_id, checkpoint_path=checkpoint,
                                         checkpoint_sha256=digest))
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--text-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--route-decisions", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--metrics-file")
    parser.add_argument("--reuse-answers",
                        help="Frozen deterministic Stage-01 JSONL answers reusable only for one exact expert selection.")
    parser.add_argument("--reuse-expert-id", type=int)
    parser.add_argument("--reuse-expert-answer", action="append", default=[], metavar="EXPERT_ID=JSONL",
                        help="Frozen Stage-01 answer file for an exact immutable Single expert; may be repeated.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, text_tower=args.text_tower,
        eval_modality_routing_mode="task")
    model.eval()
    bridge = AdapterBridge(model, verify_ddp=False)
    bridge.freeze_all_experts()
    registry = _registry(bridge, args.model_path)
    composer = ExpertComposer(bridge)

    all_rows = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    routes = json.loads(Path(args.route_decisions).read_text(encoding="utf-8"))
    if routes.get("oracle_used") is not False or routes.get("answer_features_used") is not False or routes.get("task_id_lookup_used") is not False:
        raise ValueError("route manifest failed answer/Oracle/task lookup audit")
    route_map = {str(item["question_id"]): tuple(int(value) for value in item["expert_ids"]) for item in routes["routes"]}
    if len(route_map) != len(routes["routes"]):
        raise ValueError("route manifest contains duplicate question IDs")
    pairs = [(row, route_map[str(row["question_id"])]) for row in all_rows]
    pairs = _chunks(pairs, args.num_chunks)[args.chunk_idx]
    rows = [item[0] for item in pairs]
    selections = [item[1] for item in pairs]
    visible = set(int(value) for value in routes["visible_expert_ids"])
    if any(len(value) > 2 or not set(value).issubset(visible) for value in selections):
        raise ValueError("route selection violates deployment visibility or cardinality")
    reuse_by_expert = {}
    if args.reuse_answers:
        if args.reuse_expert_id is None:
            raise ValueError("--reuse-answers requires --reuse-expert-id")
        args.reuse_expert_answer.append(f"{args.reuse_expert_id}={args.reuse_answers}")
    for specification in args.reuse_expert_answer:
        expert_text, separator, answer_path = specification.partition("=")
        if not separator:
            raise ValueError("--reuse-expert-answer must be EXPERT_ID=JSONL")
        expert_id = int(expert_text)
        if expert_id in reuse_by_expert:
            raise ValueError("duplicate reuse answer source for expert")
        reuse = {}
        with open(answer_path, "r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                key = str(value["question_id"])
                if key in reuse:
                    raise ValueError("reuse answers contain duplicate question IDs")
                reuse[key] = value
        reuse_by_expert[expert_id] = reuse

    cached_rows, pending_rows, pending_selections = {}, [], []
    reused_by_expert = {str(value): 0 for value in reuse_by_expert}
    for row, selected in zip(rows, selections):
        cached = reuse_by_expert.get(selected[0], {}).get(str(row["question_id"])) if len(selected) == 1 else None
        if cached is None:
            pending_rows.append(row)
            pending_selections.append(selected)
        else:
            if cached.get("prompt") != row["text"]:
                raise ValueError("reuse answer prompt mismatch")
            cached_rows[str(row["question_id"])] = str(cached["text"])
            reused_by_expert[str(selected[0])] += 1
    loader = DataLoader(RoutedDataset(pending_rows, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode),
                        batch_size=1, num_workers=min(4, max(1, len(pending_rows))), shuffle=False)
    target = Path(args.answers_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated_rows = {}
    for (input_ids, image_tensor), row, selected in tqdm(zip(loader, pending_rows, pending_selections), total=len(pending_rows)):
        mode = ("base_only", "single", "direct_sum")[len(selected)]
        with torch.inference_mode(), CompositionRuntime(registry, bridge, composer, selected, (), mode):
            output_ids = model.generate(input_ids.to("cuda", non_blocking=True),
                                        images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                                        do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True)
        input_length = input_ids.shape[1]
        generated_rows[str(row["question_id"])] = tokenizer.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True)[0].strip()
    reused = len(cached_rows)
    with target.open("w", encoding="utf-8") as handle:
        for row, selected in zip(rows, selections):
            mode = ("base_only", "single", "direct_sum")[len(selected)]
            question_id = str(row["question_id"])
            output = cached_rows.get(question_id, generated_rows[question_id] if question_id in generated_rows else None)
            if output is None:
                raise RuntimeError("answer assembly lost a routed sample")
            handle.write(json.dumps({"question_id": row["question_id"], "prompt": row["text"], "text": output,
                                     "answer_id": shortuuid.uuid(), "model_id": model_name,
                                     "metadata": {"expert_ids": list(selected), "composition_mode": mode,
                                                  "oracle_used": False, "answer_features_used": False,
                                                  "task_id_lookup_used": False}}) + "\n")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {"samples": len(rows), "chunk": args.chunk_idx,
               "model_seconds_per_sample": elapsed / max(1, len(rows)),
               "model_throughput_samples_per_second": len(rows) / max(elapsed, 1e-12),
               "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
               "reused_frozen_stage01_answers": reused,
               "reused_by_expert": reused_by_expert,
               "route_counts": {str(size): sum(len(value) == size for value in selections) for size in range(3)},
               "oracle_used": False, "answer_features_used": False, "task_id_lookup_used": False}
    if args.metrics_file:
        metrics_target = Path(args.metrics_file)
        metrics_target.parent.mkdir(parents=True, exist_ok=True)
        metrics_target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bridge.close()
    print(json.dumps({"status": "GENERATED", **metrics}))


if __name__ == "__main__":
    main()
