#!/usr/bin/env python3
"""Generate one temporally bounded Direct or RMS Oracle cache with one backbone."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers

from Hyper.peft import PeftModel
from compose.eval.load_compose import EvaluationBundle, _foundation, load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry, ExpertStatus
from compose.lora import AdapterBridge, CompositionRuntime, ExpertComposer, RMSStatistics, StatisticKey
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch
from compose.teacher import (
    AnswerNLL, OracleConfig, answer_token_nll, pair_candidates, select_oracle_set, stable_hash,
    summarize_oracles, validate_oracle_split, write_shard,
)
from llava import conversation as conversation_lib
from llava.conversation import conv_templates
from llava.model import LlavaLlamaForCausalLM


BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin"
COMPOSER_CONFIG_HASH = "29eacd7c25718e085a864a742b3c0d8ab343759b2be958202b11cc846923d0f2"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path):
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    selected = []
    for name in ("adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin", "config.json", "mm_projector.bin", "compose_experts.json", "compose_experts.bin", "stats.json"):
        candidate = root / name
        if candidate.is_file():
            selected.append((name, sha256_file(candidate)))
    if not selected:
        raise FileNotFoundError("checkpoint has no hashable model files: {}".format(path))
    return stable_hash({"files": selected})


def source_hashes():
    files = [
        "compose/lora/composer.py", "compose/lora/runtime.py", "compose/lora/statistics.py",
        "compose/teacher/scorer.py", "compose/teacher/oracle_set.py",
    ]
    return stable_hash({path: sha256_file(path) for path in files})


def _load_hyper(checkpoint, device):
    config = transformers.AutoConfig.from_pretrained(checkpoint)
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model = LlavaLlamaForCausalLM.from_pretrained(
        BASE_MODEL, low_cpu_mem_usage=True, config=config, torch_dtype=torch.bfloat16,
    )
    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
        VISION_TOWER, model_max_length=77, padding_side="right", use_fast=True,
    )
    model.set_clip_tokenizer(clip_tokenizer)
    model.set_tokenizer(tokenizer)
    if hasattr(model, "initialize_instance_router"):
        model.initialize_instance_router()
    non_lora = torch.load(Path(checkpoint) / "non_lora_trainables.bin", map_location="cpu")
    non_lora = {(key[11:] if key.startswith("base_model.") else key): value for key, value in non_lora.items()}
    if any(key.startswith("model.model.") for key in non_lora):
        non_lora = {(key[6:] if key.startswith("model.") else key): value for key, value in non_lora.items()}
    model.load_state_dict(non_lora, strict=False)
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model.config.eval_modality_routing_mode = "task"
    model.config.modality_routing_mode = "task"
    vision = model.get_vision_tower()
    if not vision.is_loaded:
        vision.load_model()
    vision.to(device=device, dtype=torch.bfloat16)
    text = model.get_text_tower()
    if not text.is_loaded:
        text.load_model()
    text.to(device=device, dtype=torch.bfloat16)
    model.config.image_aspect_ratio = "pad"
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = False
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return EvaluationBundle(model, tokenizer, vision.image_processor, 2048, {"adapter_kind": "hyper", "checkpoint": checkpoint})


def load_bundle(kind, checkpoint, device):
    if kind == "base":
        model, tokenizer, image_processor = _foundation(
            BASE_MODEL, VISION_TOWER, PROJECTOR, device, torch.bfloat16, 2048,
        )
        return EvaluationBundle(model, tokenizer, image_processor, 2048, {"adapter_kind": "base"})
    if kind == "compose":
        return load_compose_model(
            BASE_MODEL, checkpoint, VISION_TOWER, PROJECTOR, expert_id=None,
            device=device, dtype=torch.bfloat16, model_max_length=2048,
        )
    if kind == "hyper":
        return _load_hyper(checkpoint, device)
    raise ValueError("unsupported checkpoint kind: {}".format(kind))


def build_registry(bridge, checkpoint, checkpoint_sha):
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        registry.register(ExpertMetadata(
            expert_id=expert_id, adapter_name=bridge.adapter_name, rank=8, alpha=96,
            status=ExpertStatus.FROZEN, creation_task=expert_id,
            checkpoint_path=checkpoint,
            checkpoint_sha256=stable_hash({"checkpoint": checkpoint_sha, "logical_expert_id": expert_id}),
        ))
    return registry


def prepare(bundle, records, images, device):
    return _prepare_multimodal_batch(bundle, _collate(records, bundle, images, device))


def load_or_collect_rms(path, bundle, bridge, registry, visible, calibration_path, images, batch_size, device, checkpoint_sha, config_hash_value):
    target = Path(path)
    dataset_sha = sha256_file(calibration_path)
    provenance = {
        "calibration_split": "train_calibration", "checkpoint_hash": checkpoint_sha,
        "dataset_manifest_hash": dataset_sha, "composition_config_hash": COMPOSER_CONFIG_HASH,
        "oracle_config_hash": config_hash_value, "visible_expert_ids": list(visible),
    }
    if target.exists():
        return RMSStatistics.load_json(target, provenance, registry), True
    records = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    stats = RMSStatistics(provenance)
    hooks = []
    snapshot = bridge.snapshot_runtime_state()
    bridge.set_active_experts([])
    try:
        for layer_name, module in bridge.named_layers:
            def collect(current_module, inputs, base_output, name=layer_name):
                for expert_id in visible:
                    delta = bridge.compute_expert_delta(current_module, expert_id, inputs[0])
                    stats.update(StatisticKey(expert_id, name, name, type(current_module).__name__), delta,
                                 base_output + delta.to(base_output.dtype), base_output)
            hooks.append(module.register_forward_hook(collect))
        with torch.inference_mode():
            for offset in range(0, len(records), batch_size):
                bundle.model(**prepare(bundle, records[offset:offset + batch_size], images, device))
    finally:
        for hook in hooks:
            hook.remove()
        bridge.restore_runtime_state(snapshot)
    target.parent.mkdir(parents=True, exist_ok=True)
    stats.save_json(target)
    return stats, False


def convert(details, index):
    per_token = details.get("per_token_nll", ())
    return AnswerNLL(
        float(details["sum_nll"][index]), float(details["mean_nll"][index]),
        int(details["token_count"][index]), bool(details["exact_teacher_forced"][index]),
        tuple(float(value) for value in per_token[index]) if per_token else (),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kind", choices=("base", "compose", "hyper"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--calibration-questions")
    parser.add_argument("--images", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--temporal-scope", choices=("historical_only", "post_task_diagnostic"), required=True)
    parser.add_argument("--visible-experts", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--rms-statistics")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage03-single-token-regression", action="store_true")
    args = parser.parse_args()
    split = validate_oracle_split(args.split)
    visible = tuple(int(value) for value in args.visible_experts.split(",") if value != "")
    if len(visible) != len(set(visible)) or any(value > args.task_id for value in visible):
        raise ValueError("visible experts violate uniqueness or future boundary")
    if args.temporal_scope == "historical_only" and any(value >= args.task_id for value in visible):
        raise ValueError("historical-only Oracle includes a current/future expert")
    raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = OracleConfig.from_mapping(raw_config)
    if cfg.composition_mode == "rms_calibrated" and len(visible) >= 2 and not args.rms_statistics:
        raise ValueError("RMS Oracle with pairs requires an isolated RMS statistics path")
    records = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    sample_ids = ["{}:{}".format(args.task_name, row.get("question_id", index)) for index, row in enumerate(records)]
    checkpoint_sha = checkpoint_hash(args.checkpoint)
    dataset_sha = sha256_file(args.questions)
    tokenizer_sha = stable_hash({"identifier": BASE_MODEL, "eos": 2, "answer_mask": cfg.answer_mask_version})
    logical_hashes = {str(value): stable_hash({"checkpoint": checkpoint_sha, "logical_expert_id": value}) for value in visible}
    rms_hash = sha256_file(args.rms_statistics) if args.rms_statistics and Path(args.rms_statistics).exists() else "none"
    provenance = {
        "sample_id": stable_hash({"sample_ids": sample_ids}), "dataset_manifest_hash": dataset_sha,
        "split": split, "tokenizer_hash": tokenizer_sha, "model_identifier": BASE_MODEL,
        "base_checkpoint_hash": checkpoint_hash(BASE_MODEL), "expert_registry_hash": stable_hash({"visible": visible, "checkpoint": checkpoint_sha}),
        "expert_checkpoint_hashes": logical_hashes, "composition_mode": cfg.composition_mode,
        "rms_statistics_hash": rms_hash, "oracle_config_hash": args.config_hash,
        "answer_mask_version": cfg.answer_mask_version, "answer_template_hash": stable_hash({"template": "vicuna_v1", "eos": cfg.include_eos}),
        "target_averaging": cfg.target_averaging, "composer_version": COMPOSER_CONFIG_HASH,
        "code_version": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() + ":" + source_hashes(),
    }
    cache_target = Path(args.cache_file)
    started = time.perf_counter()
    if cache_target.exists():
        from compose.teacher import load_shard
        envelope = load_shard(cache_target, provenance)
        rows = envelope["records"]
        cache_hit = True
        cache_latency = time.perf_counter() - started
        peak_memory = 0
        timings = {}
        rms_cache_hit = None
    else:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        bundle = load_bundle(args.checkpoint_kind, args.checkpoint, args.device)
        conversation_lib.default_conversation = conv_templates["vicuna_v1"]
        bridge = None if args.checkpoint_kind == "base" else AdapterBridge(bundle.model, verify_ddp=False)
        if bridge is None and visible:
            raise ValueError("base-only checkpoint cannot expose experts")
        missing = sorted(set(visible) - set(bridge.expert_ids)) if bridge is not None else []
        if missing:
            raise KeyError("checkpoint is missing visible experts {}".format(missing))
        registry = build_registry(bridge, args.checkpoint, checkpoint_sha) if bridge is not None else None
        rms_stats, rms_cache_hit = (None, None)
        if cfg.composition_mode == "rms_calibrated" and len(visible) >= 2:
            if not args.calibration_questions:
                raise ValueError("RMS statistics collection requires train calibration questions")
            rms_stats, rms_cache_hit = load_or_collect_rms(
                args.rms_statistics, bundle, bridge, registry, visible, args.calibration_questions,
                args.images, args.batch_size, args.device, checkpoint_sha, args.config_hash,
            )
            rms_hash = sha256_file(args.rms_statistics)
            provenance["rms_statistics_hash"] = rms_hash
        composer = ExpertComposer(bridge, rms_stats) if bridge is not None else None
        timings = {"empty": [], "single": [], "pair": []}
        stage03_differences = []
        rank = int(os.environ.get("RANK", "0"))
        partial_target = cache_target.with_name(cache_target.name + ".rank{}.partial".format(rank))
        rows = []
        if partial_target.exists():
            from compose.teacher import load_shard
            rows = list(load_shard(partial_target, provenance)["records"])
        completed_ids = {str(row["sample_id"]) for row in rows}
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        with torch.inference_mode():
            for offset in range(0, len(records), args.batch_size):
                batch_records = records[offset:offset + args.batch_size]
                batch_sample_ids = sample_ids[offset:offset + len(batch_records)]
                completed_in_batch = [sample_id in completed_ids for sample_id in batch_sample_ids]
                if all(completed_in_batch):
                    continue
                if any(completed_in_batch):
                    raise ValueError("partial Oracle cache ended inside a scoring batch")
                batch = prepare(bundle, batch_records, args.images, args.device)
                nll_maps = [dict() for _ in batch_records]
                for ids, mode, category in [((), "base_only", "empty"), *[((value,), "single", "single") for value in visible]]:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    if bridge is None:
                        if ids:
                            raise AssertionError("base-only bundle received an expert candidate")
                        logits = bundle.model(**batch).logits
                    else:
                        with CompositionRuntime(registry, bridge, composer, ids, [], mode):
                            logits = bundle.model(**batch).logits
                    details = answer_token_nll(logits, batch["labels"], eos_token_id=bundle.tokenizer.eos_token_id, include_eos=cfg.include_eos)
                    if args.stage03_single_token_regression:
                        if not torch.all(details["token_count"].eq(1)):
                            raise AssertionError("controlled Stage 03 regression expects one non-EOS answer token")
                        for index in range(len(batch_records)):
                            target_positions = torch.where(
                                batch["labels"][index].ne(-100) & batch["labels"][index].ne(bundle.tokenizer.eos_token_id)
                            )[0]
                            position = int(target_positions[0])
                            target = int(batch["labels"][index, position])
                            manual = torch.logsumexp(logits[index, position - 1].float(), 0) - logits[index, position - 1, target].float()
                            stage03_differences.append(abs(float(manual) - float(details["mean_nll"][index])))
                    torch.cuda.synchronize()
                    timings[category].append((time.perf_counter() - tick) / len(batch_records))
                    for index in range(len(batch_records)):
                        nll_maps[index][ids] = convert(details, index)
                    del logits, details
                allowed_pairs = []
                for values in nll_maps:
                    allowed_pairs.append(pair_candidates(visible, {ids[0]: score.mean_nll for ids, score in values.items() if len(ids) == 1}, top_k_for_pair=cfg.top_k_for_pair, max_pairs=cfg.max_pairs))
                union_pairs = sorted(set(pair for values in allowed_pairs for pair in values))
                for pair in union_pairs:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    with CompositionRuntime(registry, bridge, composer, pair, [], cfg.composition_mode):
                        logits = bundle.model(**batch).logits
                    details = answer_token_nll(logits, batch["labels"], eos_token_id=bundle.tokenizer.eos_token_id, include_eos=cfg.include_eos)
                    if args.stage03_single_token_regression:
                        if not torch.all(details["token_count"].eq(1)):
                            raise AssertionError("controlled Stage 03 regression expects one non-EOS answer token")
                        for index in range(len(batch_records)):
                            target_positions = torch.where(
                                batch["labels"][index].ne(-100) & batch["labels"][index].ne(bundle.tokenizer.eos_token_id)
                            )[0]
                            position = int(target_positions[0])
                            target = int(batch["labels"][index, position])
                            manual = torch.logsumexp(logits[index, position - 1].float(), 0) - logits[index, position - 1, target].float()
                            stage03_differences.append(abs(float(manual) - float(details["mean_nll"][index])))
                    torch.cuda.synchronize()
                    timings["pair"].append((time.perf_counter() - tick) / len(batch_records))
                    for index in range(len(batch_records)):
                        if pair in allowed_pairs[index]:
                            nll_maps[index][pair] = convert(details, index)
                    del logits, details
                for index, values in enumerate(nll_maps):
                    record = select_oracle_set(
                        sample_id=sample_ids[offset + index], task_id=args.task_id, task_name=args.task_name,
                        split=split, candidate_expert_ids=visible, nll_by_set=values, config=cfg,
                        config_hash=args.config_hash, expert_pool_hash=provenance["expert_registry_hash"],
                        dataset_manifest_hash=dataset_sha, temporal_scope=args.temporal_scope,
                        diagnostics={"checkpoint": args.checkpoint, "rms_statistics_hash": rms_hash},
                    )
                    rows.append(record.to_dict())
                completed_ids.update(batch_sample_ids)
                write_shard(partial_target, rows, provenance, rank=rank)
        peak_memory = int(torch.cuda.max_memory_allocated(torch.device(args.device)))
        expected_ids = set(sample_ids)
        if completed_ids != expected_ids or len(rows) != len(sample_ids):
            raise ValueError("partial Oracle cache completion mismatch")
        order = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        rows.sort(key=lambda row: order[str(row["sample_id"])])
        write_shard(partial_target, rows, provenance, rank=rank)
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_target, cache_target)
        cache_hit = False
        cache_latency = 0.0
    duration = time.perf_counter() - started
    metrics = summarize_oracles(rows)
    metrics["performance"] = {
        "duration_seconds": duration, "samples_per_second": len(rows) / duration if duration else 0,
        "peak_cuda_memory_bytes": peak_memory, "cache_hit": cache_hit,
        "cache_hit_latency_seconds": cache_latency, "cache_bytes": cache_target.stat().st_size,
        "mean_forward_seconds_per_sample": {key: statistics.fmean(value) if value else None for key, value in timings.items()},
        "candidate_sets_per_sample": statistics.fmean(1 + len(row["singles"]) + len(row["pairs"]) for row in rows),
        "rms_statistics_cache_hit": rms_cache_hit,
        "stage03_single_token_nll_max_absolute_difference": max(stage03_differences) if not cache_hit and stage03_differences else None,
    }
    metrics["provenance"] = provenance
    target = Path(args.summary_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "samples": len(rows), "mode": cfg.composition_mode,
                      "scope": args.temporal_scope, "rates": [metrics["EmptyOracleRate"], metrics["SingleOracleRate"], metrics["PairOracleRate"]],
                      "cache_hit": cache_hit}, sort_keys=True))


if __name__ == "__main__":
    main()
