"""Compare frozen legacy ComposeLinear direct sum with Stage-03 runtime."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import AdapterBridge, CompositionRuntime, ExpertComposer
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--expert-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    ids = tuple(int(item) for item in args.expert_ids.split(","))
    common = {"model_path": "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b",
              "vision_tower": "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336",
              "projector_path": "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin"}
    bundle = load_compose_model(checkpoint_dir=args.checkpoint, expert_id=None, device="cuda:0",
                                dtype=torch.bfloat16, model_max_length=2048, **common)
    with open(args.questions, encoding="utf-8") as handle:
        records = json.load(handle)[:args.max_samples]
    sample_ids = [str(row["question_id"]) for row in records]
    raw = _collate(records, bundle, args.images, "cuda:0")
    prepared = _prepare_multimodal_batch(bundle, raw)
    a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
    b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
    labels = prepared["labels"]
    answer_positions = []
    targets = []
    for index in range(len(records)):
        positions = torch.where(labels[index].ne(-100))[0]
        answer_positions.append(int(positions[0].item()) - 1)
        targets.append(a_id if records[index]["answer"] == "A" else b_id)
    bundle.expert_pool.manager.set_default_selection(ids, [1.0, 1.0], normalization="none")
    with torch.inference_mode():
        old = bundle.model(**prepared).logits
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    with torch.inference_mode(), CompositionRuntime(registry, bridge, ExpertComposer(bridge), ids, [], "direct_sum"):
        new = bundle.model(**prepared).logits
    difference = (new.float() - old.float()).abs()
    old_answers, new_answers, old_nll, new_nll = [], [], [], []
    for row, position, target in zip(range(len(records)), answer_positions, targets):
        old_token, new_token = old[row, position].float(), new[row, position].float()
        old_answers.append([float(old_token[a_id]), float(old_token[b_id])])
        new_answers.append([float(new_token[a_id]), float(new_token[b_id])])
        old_nll.append(float(torch.logsumexp(old_token, 0) - old_token[target]))
        new_nll.append(float(torch.logsumexp(new_token, 0) - new_token[target]))
    old_pair = torch.tensor(old_answers)
    new_pair = torch.tensor(new_answers)
    old_predictions = old_pair.argmax(1)
    new_predictions = new_pair.argmax(1)
    result = {"status": "PASSED" if float(difference.max()) == 0.0 else "FAILED",
              "checkpoint": args.checkpoint, "question_file": args.questions, "expert_ids": list(ids),
              "sample_ids": sample_ids, "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
              "samples": len(records), "max_absolute_logit_difference": float(difference.max()),
              "mean_absolute_logit_difference": float(difference.mean()),
              "answer_logit_max_absolute_difference": float((new_pair - old_pair).abs().max()),
              "prediction_agreement": float(old_predictions.eq(new_predictions).float().mean()),
              "max_absolute_nll_difference": max(abs(a-b) for a, b in zip(old_nll, new_nll)),
              "old_answer_logits": old_answers, "new_answer_logits": new_answers,
              "old_nll": old_nll, "new_nll": new_nll,
              "tolerance": {"exact_bf16": 0.0}}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "samples", "max_absolute_logit_difference", "prediction_agreement", "max_absolute_nll_difference")}, sort_keys=True))
    if result["status"] != "PASSED":
        raise SystemExit("STAGE03_BLOCKED_BY_DIRECT_SUM_REGRESSION")


if __name__ == "__main__":
    main()
