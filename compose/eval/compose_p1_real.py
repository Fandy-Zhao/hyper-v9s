"""Compose P1-Real: free-form real-data composition evaluation (C0-C5).

Modes (identical to the synthetic P1 protocol):
  base       = backbone only
  single_B / single_C = each expert alone (gate 1)
  c0         = direct sum of the two expert deltas
  c1         = sum with 1/sqrt(2) per expert
  c2         = per-layer RMS calibration (arithmetic-mean reference, clip
               [0.25, 4.0]) + 1/sqrt(2)  [frozen_coefficients / rms_compose]
  c3         = c2 + validation-only scalar search (g_B, g_C in [0,2]) on the
               B+C calibration split

The official TextVQA/VQA2.0 metric (VQA Accuracy over the 10 human answers)
is the primary correctness metric; normalized exact match vs the majority
answer is reported alongside. NLL synergy is computed per sample from
teacher-forced answer-token NLL.

Optional layer-group diagnostics (spec section 11): pass --layer-masks with
a JSON mapping layer_name -> {"B": 0|1, "C": 0|1} to evaluate a masked
composition (L1/L2/L3). L0 is the ordinary c1 mode.

Usage:
  python -m compose.eval.compose_p1_real \
      --model-path ... --vision-tower ... --checkpoint <assembled pair> \
      --calibration-questions <BC_calib.json> --test-questions <BC_test.json> \
      --images <data root> --expert-ids 1,2 --pair-name independent_b_c \
      --checkpoint-seed 0 --analysis-seed 0 --output-dir ... --device cuda:0
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from PIL import Image

from compose.data.real_p1.official_metric import normalize, vqa_accuracy
from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import (
    AdapterBridge,
    CompositionRuntime,
    ExpertComposer,
    RMSCompositionConfig,
    RMSStatistics,
    StatisticKey,
)
from compose.lora.rms_composition import frozen_coefficients
from compose.oracle.evaluator import _prepare_multimodal_batch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def ece(rows, bins=15):
    total = len(rows)
    if not total:
        return 0.0
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            row for row in rows
            if lower <= row["confidence"] < upper or (index == bins - 1 and row["confidence"] == 1.0)
        ]
        if selected:
            result += len(selected) / total * abs(
                statistics.fmean(row["confidence"] for row in selected)
                - statistics.fmean(float(row["correct"]) for row in selected)
            )
    return result


def summarize(rows, best_single_nll=None):
    result = {
        "samples": len(rows),
        "accuracy": statistics.fmean(float(row["correct"]) for row in rows),
        "mean_vqa_score": statistics.fmean(row["vqa_score"] for row in rows),
        "mean_em": statistics.fmean(row["em"] for row in rows),
        "answer_token_nll": statistics.fmean(row["nll"] for row in rows),
        "brier_score": statistics.fmean(row["brier"] for row in rows),
        "ece": ece(rows),
    }
    if best_single_nll is not None:
        synergy = [single - row["nll"] for single, row in zip(best_single_nll, rows)]
        ordered = sorted(synergy)
        tail = max(1, math.ceil(0.1 * len(ordered)))
        result.update({
            "mean_synergy": statistics.fmean(synergy),
            "median_synergy": statistics.median(synergy),
            "positive_synergy_rate": statistics.fmean(float(value > 0) for value in synergy),
            "worst_10_percent_synergy": statistics.fmean(ordered[:tail]),
        })
    return result


def _prompt(record, model_config, conv_mode):
    question = str(record.get("text", record.get("question", "")))
    image_token = DEFAULT_IMAGE_TOKEN
    if model_config.mm_use_im_start_end:
        image_token = "<im_start>" + image_token + "<im_end>"
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def build_generation_batch(records, bundle, image_folder, device):
    """Right-padded input_ids/attention_mask + images for generation."""
    encoded = []
    images = []
    for record in records:
        prompt = _prompt(record, bundle.model.config, "v1")
        input_ids = tokenizer_image_token(
            prompt, bundle.tokenizer, return_tensors="pt"
        )
        encoded.append(input_ids)
        image = Image.open(os.path.join(image_folder, str(record["image"]))).convert("RGB")
        images.append(image)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        encoded, batch_first=True, padding_value=bundle.tokenizer.pad_token_id
    )[:, : bundle.context_length].to(device)
    attention_mask = input_ids.ne(bundle.tokenizer.pad_token_id).to(device)
    image_tensor = process_images(images, bundle.image_processor, bundle.model.config).to(
        device=device, dtype=torch.bfloat16
    )
    return input_ids, attention_mask, image_tensor


def generate_rows(bundle, records, image_folder, device, max_new_tokens):
    rows = []
    for record in records:
        input_ids, attention_mask, image_tensor = build_generation_batch(
            [record], bundle, image_folder, device
        )
        with torch.inference_mode():
            output_ids = bundle.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        input_length = input_ids.shape[1]
        prediction = bundle.tokenizer.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True
        )[0].strip()
        rows.append(prediction)
    return rows


@contextmanager
def selection(bundle, registry, bridge, rms_stats, ids, mode, scalars=(1.0, 1.0)):
    manager = bundle.expert_pool.manager
    try:
        if mode == "base":
            manager.clear_default_selection()
            yield
        elif mode == "single_left":
            manager.set_default_selection((ids[0],), (1.0,), normalization="none")
            yield
        elif mode == "single_right":
            manager.set_default_selection((ids[1],), (1.0,), normalization="none")
            yield
        elif mode in ("c0", "c1"):
            gate = 1.0 if mode == "c0" else 1.0 / math.sqrt(2.0)
            manager.set_default_selection(ids, (gate, gate), normalization="none")
            yield
        elif mode in ("c2", "c3"):
            config = RMSCompositionConfig(expert_scalars=tuple(map(float, scalars)))
            composer = ExpertComposer(bridge, rms_stats, config)
            with CompositionRuntime(registry, bridge, composer, ids, (), "rms_calibrated"):
                yield
        else:
            raise ValueError("unknown mode {}".format(mode))
    finally:
        manager.clear_default_selection()


class LayerMaskedComposer:
    """C1-style composition with per-layer per-expert masks (diagnostics)."""

    def __init__(self, bridge, masks, scale=1.0 / math.sqrt(2.0)):
        self.bridge = bridge
        self.masks = masks
        self.scale = scale
        self.hooks = []

    def __enter__(self):
        self.bridge.enable_pair_execution()
        for layer_name, module in self.bridge.named_layers:
            def hook(current_module, inputs, base_output, name=layer_name):
                if not inputs:
                    raise ValueError("layer hook received no hidden states")
                masks = self.masks.get(name, {"B": 1, "C": 1})
                output = base_output
                for expert_key, expert_id in (("B", self.expert_b), ("C", self.expert_c)):
                    coefficient = float(masks.get(expert_key, 1)) * self.scale
                    if coefficient:
                        delta = self.bridge.compute_expert_delta(
                            current_module, expert_id, inputs[0]
                        ).to(base_output.dtype)
                        output = output + delta * coefficient
                return output
            self.hooks.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in reversed(self.hooks):
            hook.remove()
        self.bridge.disable_pair_execution()
        self.hooks = []
        return False


def answer_rows(logits, labels, records):
    """Teacher-forced per-sample NLL over the gold answer tokens."""
    rows = []
    for index, record in enumerate(records):
        supervised = torch.where(labels[index].ne(IGNORE_INDEX))[0]
        if not supervised.numel():
            raise ValueError("sample {} has no answer tokens".format(record["question_id"]))
        positions = supervised.tolist()
        gold_tokens = labels[index, supervised].tolist()
        # logits at position p predict token p+1 (causal shift)
        predicting = torch.tensor([p - 1 for p in positions], device=logits.device)
        token_logits = logits[index, predicting].float()
        nlls = torch.nn.functional.cross_entropy(
            token_logits, torch.tensor(gold_tokens, device=logits.device), reduction="none"
        )
        nll = float(nlls.mean().item())
        conf = math.exp(-nll)
        rows.append({
            "sample_id": str(record["question_id"]),
            "question": str(record.get("text", record.get("question", ""))),
            "gold": str(record["answer"]),
            "answers": list(record.get("answers", [str(record["answer"])])),
            "nll": nll,
            "confidence": conf,
            "prediction": "",
            "vqa_score": 0.0,
            "em": 0.0,
            "correct": False,
            "brier": 0.0,
        })
    return rows


def fill_predictions(rows, predictions):
    for row, prediction in zip(rows, predictions):
        score = vqa_accuracy(prediction, row["answers"])
        em = 1.0 if normalize(prediction) == normalize(row["gold"]) else 0.0
        correct = score >= 2.0 / 3.0
        row.update({
            "prediction": prediction,
            "vqa_score": float(score),
            "em": float(em),
            "correct": bool(correct),
            "brier": (row["confidence"] - float(correct)) ** 2,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-questions", required=True)
    parser.add_argument("--test-questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--expert-ids", required=True)
    parser.add_argument("--pair-name", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--analysis-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--c3-grid", default="0.5,1.0,1.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer-masks", default="")
    parser.add_argument("--mode-filter", default="")
    parser.add_argument("--skip-c3", action="store_true")
    args = parser.parse_args()
    ids = tuple(map(int, args.expert_ids.split(",")))
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("P1-Real requires two distinct expert IDs")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing nonempty output directory {}".format(output))
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.analysis_seed)
    torch.cuda.manual_seed_all(args.analysis_seed)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.time()
    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower,
        projector_path=str(Path(args.model_path) / "mm_projector.bin"),
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    missing = sorted(set(ids) - set(bridge.expert_ids))
    if missing:
        raise KeyError("checkpoint is missing experts {}".format(missing))

    checkpoint_hash = hashlib.sha256(
        (sha256(Path(args.checkpoint) / "compose_experts.json")
         + sha256(Path(args.checkpoint) / "compose_experts.bin")).encode()
    ).hexdigest()
    provenance = {
        "calibration_split": "BC_calib",
        "checkpoint_hash": checkpoint_hash,
        "dataset_manifest_hash": sha256(Path(args.calibration_questions)),
        "composition_config_hash": "compose-p1-real-arithmetic-rms-v1",
        "analysis_seed": args.analysis_seed,
        "checkpoint_seed": args.checkpoint_seed,
    }
    rms_stats = RMSStatistics(provenance)
    cosine = {}
    hooks = []
    for layer_name, module in bridge.named_layers:
        def collect(current_module, inputs, base_output, name=layer_name):
            deltas = []
            for expert_id in ids:
                delta = bridge.compute_expert_delta(current_module, expert_id, inputs[0])
                deltas.append(delta)
                rms_stats.update(
                    StatisticKey(expert_id, name, name, type(current_module).__name__),
                    delta,
                    base_output + delta.to(base_output.dtype),
                    base_output,
                )
            left, right = (item.detach().float().reshape(-1) for item in deltas)
            denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            value = float(torch.dot(left, right) / denominator) if denominator.item() else 0.0
            state = cosine.setdefault(name, [0.0, 0])
            state[0] += value
            state[1] += 1
        hooks.append(module.register_forward_hook(collect))

    calibration = json.loads(Path(args.calibration_questions).read_text(encoding="utf-8"))
    if args.calibration_samples > 0:
        calibration = calibration[: args.calibration_samples]
    with torch.inference_mode():
        for offset in range(0, len(calibration), args.batch_size):
            records = calibration[offset: offset + args.batch_size]
            prepared = _prepare_multimodal_batch(bundle, _collate_real(bundle, records, args.images, args.device))
            bundle.model(**prepared)
    for hook in hooks:
        hook.remove()
    rms_stats.save_json(output / "rms_statistics.json")

    test = json.loads(Path(args.test_questions).read_text(encoding="utf-8"))
    if args.test_samples > 0:
        test = test[: args.test_samples]

    modes = ("base", "single_left", "single_right", "c0", "c1", "c2", "c3")
    if args.mode_filter:
        modes = tuple(m for m in modes if m in args.mode_filter.split(","))

    # C3 scalar search on the calibration split only
    selected_scalars = (1.0, 1.0)
    grid_loss = {}
    if not args.skip_c3 and "c3" in modes:
        grid = tuple(float(value) for value in args.c3_grid.split(","))
        if not grid or any(not 0.0 <= value <= 2.0 for value in grid):
            raise ValueError("C3 grid values must lie in [0, 2]")
        with torch.inference_mode():
            for left in grid:
                for right in grid:
                    losses = []
                    for offset in range(0, len(calibration), args.batch_size):
                        records = calibration[offset: offset + args.batch_size]
                        prepared = _prepare_multimodal_batch(bundle, _collate_real(bundle, records, args.images, args.device))
                        with selection(bundle, registry, bridge, rms_stats, ids, "c3", (left, right)):
                            logits = bundle.model(**prepared).logits
                        rows = answer_rows(logits, prepared["labels"], records)
                        losses.extend(row["nll"] for row in rows)
                    grid_loss[(left, right)] = statistics.fmean(losses)
        selected_scalars = min(grid_loss, key=lambda pair: (grid_loss[pair], pair))

    rows_by_mode = {mode: [] for mode in modes}
    timings = {mode: [] for mode in modes}
    layer_masks = None
    if args.layer_masks:
        layer_masks = json.loads(Path(args.layer_masks).read_text())
        if "c0" in modes or "c1" in modes:
            pass
    masked_composer = None
    if layer_masks:
        masked_composer = LayerMaskedComposer(bridge, layer_masks)
        masked_composer.expert_b, masked_composer.expert_c = ids

    with torch.inference_mode():
        for offset in range(0, len(test), args.batch_size):
            records = test[offset: offset + args.batch_size]
            prepared = _prepare_multimodal_batch(bundle, _collate_real(bundle, records, args.images, args.device))
            for mode in modes:
                torch.cuda.synchronize()
                step_started = time.perf_counter()
                if layer_masks and mode in ("c1", "c0"):
                    with masked_composer:
                        logits = bundle.model(**prepared).logits
                        predictions = generate_rows(bundle, records, args.images, args.device, args.max_new_tokens)
                else:
                    scalars = selected_scalars if mode == "c3" else (1.0, 1.0)
                    with selection(bundle, registry, bridge, rms_stats, ids, mode, scalars):
                        logits = bundle.model(**prepared).logits
                        predictions = generate_rows(bundle, records, args.images, args.device, args.max_new_tokens)
                torch.cuda.synchronize()
                rows = answer_rows(logits, prepared["labels"], records)
                fill_predictions(rows, predictions)
                rows_by_mode[mode].extend(rows)
                timings[mode].append((time.perf_counter() - step_started) / len(records))

    has_both_singles = "single_left" in rows_by_mode and "single_right" in rows_by_mode
    best_single = None
    if has_both_singles:
        best_single = [
            min(left["nll"], right["nll"])
            for left, right in zip(rows_by_mode["single_left"], rows_by_mode["single_right"])
        ]
    mode_metrics = {}
    for mode in modes:
        mode_metrics[mode] = summarize(
            rows_by_mode[mode], best_single if mode in ("c0", "c1", "c2", "c3") else None
        )
        mode_metrics[mode]["mean_latency_seconds"] = statistics.fmean(timings[mode])
    for mode in ("c0", "c1", "c2", "c3"):
        if mode not in modes or not has_both_singles:
            continue
        rows = rows_by_mode[mode]
        mode_metrics[mode]["left_given_right_mean_gain"] = statistics.fmean(
            right["nll"] - pair["nll"] for right, pair in zip(rows_by_mode["single_right"], rows)
        )
        mode_metrics[mode]["right_given_left_mean_gain"] = statistics.fmean(
            left["nll"] - pair["nll"] for left, pair in zip(rows_by_mode["single_left"], rows)
        )
        mode_metrics[mode]["accuracy_delta_vs_best_single"] = mode_metrics[mode]["accuracy"] - max(
            mode_metrics["single_left"]["accuracy"], mode_metrics["single_right"]["accuracy"]
        )
        mode_metrics[mode]["vqa_score_delta_vs_best_single"] = mode_metrics[mode]["mean_vqa_score"] - max(
            mode_metrics["single_left"]["mean_vqa_score"], mode_metrics["single_right"]["mean_vqa_score"]
        )
        mode_metrics[mode]["conditional_gain_B_given_C"] = mode_metrics[mode]["left_given_right_mean_gain"]
        mode_metrics[mode]["conditional_gain_C_given_B"] = mode_metrics[mode]["right_given_left_mean_gain"]

    layers = []
    config = RMSCompositionConfig(expert_scalars=selected_scalars)
    for layer_name, _ in bridge.named_layers:
        raw = [rms_stats.delta_rms(expert_id, layer_name) for expert_id in ids]
        coefficients, audit = frozen_coefficients(raw[0], raw[1], config)
        calibrated = [raw[index] * coefficients[index] for index in range(2)]
        effective = [calibrated[index] * selected_scalars[index] / math.sqrt(2.0) for index in range(2)]
        denominator = sum(effective)
        shares = [value / denominator if denominator else 0.0 for value in effective]
        layers.append({
            "layer_name": layer_name,
            "expert_ids": list(ids),
            "raw_rms": raw,
            "rms_coefficients": list(coefficients),
            "calibrated_rms": calibrated,
            "c3_effective_rms": effective,
            "c3_contribution_shares": shares,
            "delta_cosine": cosine[layer_name][0] / cosine[layer_name][1],
            **audit,
        })

    with (output / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for index, record in enumerate(test):
            handle.write(json.dumps({
                "sample_id": str(record["question_id"]),
                "checkpoint_seed": args.checkpoint_seed,
                "analysis_seed": args.analysis_seed,
                "pair_name": args.pair_name,
                "question": str(record.get("text", record.get("question", ""))),
                "function_label": record.get("function_label"),
                "question_type": record.get("question_type"),
                "operation": record.get("operation"),
                "gold": str(record["answer"]),
                "answers": list(record.get("answers", [str(record["answer"])])),
                "modes": {mode: {k: rows_by_mode[mode][index][k] for k in (
                    "nll", "confidence", "prediction", "vqa_score", "em", "correct", "brier"
                )} for mode in modes},
            }, sort_keys=True) + "\n")

    summary = {
        "status": "COMPLETED",
        "stage": "p1_real",
        "pair_name": args.pair_name,
        "expert_ids": list(ids),
        "checkpoint_seed": args.checkpoint_seed,
        "analysis_seed": args.analysis_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_hash": checkpoint_hash,
        "calibration_questions": args.calibration_questions,
        "test_questions": args.test_questions,
        "calibration_sample_count": len(calibration),
        "test_sample_count": len(test),
        "test_used_for_c3_search": False,
        "c3_selected_scalars": list(selected_scalars),
        "c3_grid": [{"scalars": list(pair), "validation_nll": loss}
                    for pair, loss in sorted(grid_loss.items())],
        "layer_masks": args.layer_masks,
        "modes": mode_metrics,
        "layers": layers,
        "max_c3_expert_contribution_share": max(max(row["c3_contribution_shares"]) for row in layers),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device(args.device))),
        "duration_seconds": time.time() - started,
        "git_commit": git_commit(),
        "command": sys.argv,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "pair_name": args.pair_name,
        "checkpoint_seed": args.checkpoint_seed,
        "c3_selected_scalars": summary["c3_selected_scalars"],
        "modes": {mode: {k: mode_metrics[mode].get(k) for k in (
            "accuracy", "mean_vqa_score", "answer_token_nll", "mean_synergy",
            "accuracy_delta_vs_best_single", "vqa_score_delta_vs_best_single")}
            for mode in modes},
    }, sort_keys=True))


def _collate_real(bundle, records, image_folder, device):
    """LLaVA multimodal batch with labels for teacher-forced NLL.

    Built records carry the question under ``question`` (and ``text`` where
    available); the oracle collator reads ``text``, so normalize here.
    """
    from compose.oracle.evaluator import _collate
    normalized = []
    for record in records:
        if "text" not in record and "question" in record:
            record = dict(record)
            record["text"] = record["question"]
        normalized.append(record)
    return _collate(normalized, bundle, image_folder, device)


if __name__ == "__main__":
    main()
