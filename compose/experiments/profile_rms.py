"""Phase timer for the RMS calibration stage (audit tool, no production path).

Measures where ``compose.eval.rms_stats`` spends its wall-clock, with each
phase timed separately and the hook cost isolated from the backbone forward.

It reuses the *real* helpers (``_build_batches``, ``load_compose_model``,
``compute_expert_rms``) so the numbers describe the shipped code path rather
than a re-implementation of it.  ``--mode`` selects what is timed:

``load``      model load only
``batches``   model load + batch construction (I/O, tokenisation, images)
``forward``   + the backbone forward with no hooks registered
``hooks``     + the real ``compute_expert_rms`` (hooks + statistics)
``stats``     the statistics update in isolation, replaying recorded deltas

Usage::

    python -m compose.experiments.profile_rms --checkpoint-dir CKPT \\
        --question-file val.json --image-folder IMAGES \\
        --checkpoint-hash HASH --mode hooks --max-samples 8 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

from compose.eval.load_compose import load_compose_model
from compose.eval.rms_stats import _build_batches
from compose.lora.rms import (
    ComposeRMSConfig,
    compute_expert_rms,
    compute_expert_rms_accelerated,
)
from compose.lora.statistics import RMSStatistics, StatisticKey


def _timed(label: str, fn: Callable[[], Any], timings: Dict[str, float]) -> Any:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    timings[label] = timings.get(label, 0.0) + (time.perf_counter() - start)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument(
        "--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
    )
    parser.add_argument("--projector-path", default=None)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--mode",
        default="hooks",
        choices=["load", "batches", "forward", "hooks", "hooks_fast", "stats"],
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="repeat the timed section; reports every run so noise is visible",
    )
    args = parser.parse_args()
    projector = args.projector_path or os.path.join(args.model_path, "mm_projector.bin")

    timings: Dict[str, float] = {}
    records = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    records = records[: args.max_samples]

    bundle = _timed(
        "load_model",
        lambda: load_compose_model(
            model_path=args.model_path,
            checkpoint_dir=args.checkpoint_dir,
            vision_tower=args.vision_tower,
            projector_path=projector,
            expert_id=None,
            device=args.device,
            dtype=torch.bfloat16,
            model_max_length=2048,
        ),
        timings,
    )
    expert_ids = sorted(int(value) for value in bundle.expert_pool.expert_ids())

    if args.mode == "load":
        print(json.dumps({"timings": timings, "experts": len(expert_ids)}, indent=2))
        return

    batches = _timed(
        "build_batches",
        lambda: _build_batches(records, bundle, args.image_folder, args.device, args.batch_size),
        timings,
    )

    if args.mode in ("forward", "batches"):
        # Backbone forward only: no hooks, no delta recomputation.  This is the
        # floor any RMS implementation must pay, since the deltas are read off
        # the activations the forward itself produces.
        from compose.adapters.lora import ComposeLinear

        probe = [name for name, module in bundle.model.named_modules()
                 if isinstance(module, ComposeLinear)]
        for _ in range(args.runs):
            def _forward_all() -> None:
                bundle.model.eval()
                with torch.inference_mode():
                    for batch in batches:
                        bundle.model(**batch, return_dict=True)

            _timed("forward_no_hooks", _forward_all, timings)
        print(json.dumps({
            "timings": timings,
            "experts": len(expert_ids),
            "compose_layers": len(probe),
            "samples": len(records),
            "batches": len(batches),
        }, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(timings, indent=2))
        return

    provenance = {
        "calibration_split": "validation",
        "checkpoint_hash": args.checkpoint_hash,
        "dataset_manifest_hash": "",
        "composition_config_hash": "",
    }
    config = ComposeRMSConfig(calibration_split="validation")

    if args.mode in ("hooks", "hooks_fast"):
        collector = (
            compute_expert_rms_accelerated
            if args.mode == "hooks_fast"
            else compute_expert_rms
        )
        per_run: List[float] = []
        stats = None
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            stats, _pairs = collector(
                bundle.model,
                expert_ids,
                batches,
                provenance,
                config,
                prepare_batch=lambda batch: batch,
                forward_fn=lambda inputs: bundle.model(**inputs, return_dict=True),
                device=args.device,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            per_run.append(elapsed)
            timings["hooks_total"] = timings.get("hooks_total", 0.0) + elapsed
        n = len(records)
        payload = {
            "timings": timings,
            "hooks_per_run_s": per_run,
            "hooks_total_s": sum(per_run),
            "hooks_mean_s": statistics.mean(per_run),
            "experts": len(expert_ids),
            "samples": n,
            "seconds_per_sample": statistics.mean(per_run) / n if n else None,
            "entries": len(stats.entries) if stats is not None else 0,
        }
        print(json.dumps(payload, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(payload, indent=2))
        return

    if args.mode == "stats":
        # Isolate the accumulator: replay real deltas through the real
        # ``RMSStatistics.update`` so the per-call synchronisation cost is
        # measured without any model in the loop.
        from compose.adapters.lora import ComposeLinear

        layers = [
            (name, module)
            for name, module in bundle.model.named_modules()
            if isinstance(module, ComposeLinear)
        ]
        hidden = torch.randn(1, 4096, dtype=torch.bfloat16, device=args.device)
        deltas: List[tuple] = []
        with torch.inference_mode():
            for name, module in layers:
                for expert_id in expert_ids:
                    if str(expert_id) not in module.experts:
                        continue
                    delta = module.experts[str(expert_id)](hidden).to(torch.bfloat16)
                    deltas.append((name, expert_id, delta, delta, module.__class__.__name__))
        torch.cuda.synchronize()
        start = time.perf_counter()
        acc = RMSStatistics(provenance)
        for name, expert_id, delta, output, kind in deltas:
            acc.update(
                StatisticKey(
                    expert_id=int(expert_id),
                    layer_name=name,
                    module_name=name,
                    target_module_type=kind,
                ),
                delta,
                output,
                None,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        payload = {
            "timings": timings,
            "updates": len(deltas),
            "stats_update_s": elapsed,
            "per_update_us": 1e6 * elapsed / len(deltas) if deltas else None,
        }
        print(json.dumps(payload, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()
