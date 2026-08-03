"""Formal Compose P1 evaluation for one expert pair and checkpoint seed."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from llava.constants import IGNORE_INDEX

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
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ece(rows, bins: int = 15) -> float:
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


def answer_rows(logits, labels, records, a_id: int, b_id: int):
    rows = []
    for index, record in enumerate(records):
        supervised = torch.where(labels[index].ne(IGNORE_INDEX))[0]
        if not supervised.numel():
            raise ValueError("sample {} has no answer token".format(record["question_id"]))
        position = int(supervised[0].item()) - 1
        token = logits[index, position].float()
        target_a = str(record["answer"]) == "A"
        target_id = a_id if target_a else b_id
        nll = float(torch.logsumexp(token, dim=0) - token[target_id])
        legal = torch.softmax(token[[a_id, b_id]], dim=0)
        p_a, p_b = float(legal[0]), float(legal[1])
        predicted_a = p_a > p_b
        correct = predicted_a == target_a
        confidence = max(p_a, p_b)
        rows.append({
            "sample_id": str(record["question_id"]),
            "target": "A" if target_a else "B",
            "prediction": "A" if predicted_a else "B",
            "correct": bool(correct),
            "nll": nll,
            "probability_A": p_a,
            "probability_B": p_b,
            "confidence": confidence,
            "brier": (p_a - float(target_a)) ** 2,
        })
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
            raise ValueError("unknown P1 mode {}".format(mode))
    finally:
        manager.clear_default_selection()


def run_mode(bundle, prepared, records, registry, bridge, rms_stats, ids, mode, a_id, b_id, scalars=(1.0, 1.0)):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with selection(bundle, registry, bridge, rms_stats, ids, mode, scalars):
        logits = bundle.model(**prepared).logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return answer_rows(logits, prepared["labels"], records, a_id, b_id), elapsed


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
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--c3-grid", default="0.5,1.0,1.5")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    ids = tuple(map(int, args.expert_ids.split(",")))
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("P1 requires two distinct expert IDs")
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
        (sha256(Path(args.checkpoint) / "compose_experts.json") + sha256(Path(args.checkpoint) / "compose_experts.bin")).encode()
    ).hexdigest()
    provenance = {
        "calibration_split": "validation",
        "checkpoint_hash": checkpoint_hash,
        "dataset_manifest_hash": sha256(Path(args.calibration_questions)),
        "composition_config_hash": "compose-p1-arithmetic-rms-v1",
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
            left, right = (delta.detach().float().reshape(-1) for delta in deltas)
            denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            value = float(torch.dot(left, right) / denominator) if denominator.item() else 0.0
            state = cosine.setdefault(name, [0.0, 0])
            state[0] += value
            state[1] += 1
        hooks.append(module.register_forward_hook(collect))

    calibration = json.loads(Path(args.calibration_questions).read_text(encoding="utf-8"))[:args.calibration_samples]
    with torch.inference_mode():
        for offset in range(0, len(calibration), args.batch_size):
            records = calibration[offset:offset + args.batch_size]
            prepared = _prepare_multimodal_batch(bundle, _collate(records, bundle, args.images, args.device))
            bundle.model(**prepared)
    for hook in hooks:
        hook.remove()
    rms_stats.save_json(output / "rms_statistics.json")

    a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
    b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
    grid = tuple(float(value) for value in args.c3_grid.split(","))
    if not grid or any(not 0.0 <= value <= 2.0 for value in grid):
        raise ValueError("C3 grid values must lie in [0, 2]")
    grid_loss = {}
    with torch.inference_mode():
        for left in grid:
            for right in grid:
                losses = []
                for offset in range(0, len(calibration), args.batch_size):
                    records = calibration[offset:offset + args.batch_size]
                    prepared = _prepare_multimodal_batch(bundle, _collate(records, bundle, args.images, args.device))
                    rows, _ = run_mode(bundle, prepared, records, registry, bridge, rms_stats, ids, "c3", a_id, b_id, (left, right))
                    losses.extend(row["nll"] for row in rows)
                grid_loss[(left, right)] = statistics.fmean(losses)
    selected_scalars = min(grid_loss, key=lambda pair: (grid_loss[pair], pair))

    test = json.loads(Path(args.test_questions).read_text(encoding="utf-8"))[:args.test_samples]
    modes = ("base", "single_left", "single_right", "c0", "c1", "c2", "c3")
    rows_by_mode = {mode: [] for mode in modes}
    timings = {mode: [] for mode in modes}
    with torch.inference_mode():
        for offset in range(0, len(test), args.batch_size):
            records = test[offset:offset + args.batch_size]
            prepared = _prepare_multimodal_batch(bundle, _collate(records, bundle, args.images, args.device))
            for mode in modes:
                scalars = selected_scalars if mode == "c3" else (1.0, 1.0)
                rows, elapsed = run_mode(bundle, prepared, records, registry, bridge, rms_stats, ids, mode, a_id, b_id, scalars)
                rows_by_mode[mode].extend(rows)
                timings[mode].append(elapsed / len(records))

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
                "modes": {mode: rows_by_mode[mode][index] for mode in modes},
            }, sort_keys=True) + "\n")
    summary = {
        "status": "COMPLETED",
        "stage": "p1",
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
        "c3_grid": [{"scalars": list(pair), "validation_nll": loss} for pair, loss in sorted(grid_loss.items())],
        "modes": mode_metrics,
        "layers": layers,
        "max_c3_expert_contribution_share": max(max(row["c3_contribution_shares"]) for row in layers),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device(args.device))),
        "duration_seconds": time.time() - started,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "pair_name": args.pair_name,
        "checkpoint_seed": args.checkpoint_seed,
        "c3_selected_scalars": summary["c3_selected_scalars"],
        "modes": mode_metrics,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
