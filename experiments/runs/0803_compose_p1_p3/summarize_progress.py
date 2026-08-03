#!/usr/bin/env python3
"""Print compact progress records for the fixed Compose run."""

import glob
import json


ROOT = "/home/zhaozhuofan/Hyper-LlaVA/outputs/compose_p1_p3_20260803T090000Z"

states = [json.load(open(path)) for path in sorted(glob.glob(ROOT + "/logs/scheduler_state/*.json"))]
print(json.dumps([
    {
        "id": state["id"],
        "status": state["status"],
        "gpu": state["attempts"][-1]["cuda_visible_devices"],
        "seconds": round(state["attempts"][-1]["elapsed_seconds"], 1),
        "exit": state["attempts"][-1]["exit_code"],
    }
    for state in states
], sort_keys=True))

summaries = [json.load(open(path)) for path in sorted(glob.glob(ROOT + "/predictions/p1/*/summary.json"))]
print(json.dumps([
    {
        "pair": summary["pair_name"],
        "analysis_seed": summary["analysis_seed"],
        "scalars": summary["c3_selected_scalars"],
        "peak_gib": round(summary["peak_memory_bytes"] / 2**30, 2),
        "accuracy": {mode: summary["modes"][mode]["accuracy"] for mode in ("c0", "c1", "c2", "c3")},
    }
    for summary in summaries
], sort_keys=True))
