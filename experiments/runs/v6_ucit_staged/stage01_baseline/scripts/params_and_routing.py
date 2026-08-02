#!/usr/bin/env python3
"""Stage 01: per-task LoRA parameter counts and routing diagnostics.

Reads each task checkpoint (adapter_model.bin + stats.json) and reports:
  - total adapter params per task (per expert i = 0..cur_task)
  - incremental params per task
  - image/text gaussian stats coverage
  - adaptive_w_img routing priors
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

CK = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints")
TASKS = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)  # e.g. full_gb24 / full_orig
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    runs = []
    for t in range(1, 7):
        ck = CK / f"{args.tag}_task{t}_llava_lora_ours"
        sd = torch.load(ck / "adapter_model.bin", map_location="cpu")
        # expert params: keys ...loraA.{i}.mlp.weight / loraB.{i}.mlp.weight
        expert_params = {}
        for k, v in sd.items():
            m = re.search(r"lora([AB])\.(\d+)\.weight", k)
            if m:
                ab, i = m.group(1), int(m.group(2))
                expert_params.setdefault(i, {"A": 0, "B": 0})
                expert_params[i][ab] += v.numel()
        per_expert = {i: d["A"] + d["B"] for i, d in sorted(expert_params.items())}
        total = sum(per_expert.values())
        stats = json.loads((ck / "stats.json").read_text())
        runs.append({
            "task": t, "dataset": TASKS[t - 1],
            "checkpoint": str(ck),
            "total_adapter_params": total,
            "per_expert_params": per_expert,
            "n_experts": len(per_expert),
            "adaptive_w_img": stats.get("adaptive_w_img"),
            "image_count": stats.get("image_count"),
            "text_count": stats.get("text_count"),
        })
        print(f"task{t} {TASKS[t-1]}: {total} params, experts={len(per_expert)}, w_img={stats.get('adaptive_w_img')}")

    # incremental growth
    cumulative = 0
    for r in runs:
        cumulative += r["total_adapter_params"]
        r["cumulative_params"] = cumulative
        r["incremental_params"] = r["total_adapter_params"]
    payload = {"run": args.tag, "tasks": runs, "final_cumulative_params": cumulative}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"final cumulative adapter params: {cumulative}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
