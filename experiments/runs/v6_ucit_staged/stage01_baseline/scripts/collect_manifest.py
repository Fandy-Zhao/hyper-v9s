#!/usr/bin/env python3
"""Stage 01: collect per-run manifest and checksums for a finished run.

Sources: make_launch records (generated/run_<tag>_record.json), checkpoint dirs
on /data, training logs, and the pinned conda env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
BASE = REPO / "experiments/runs/v6_ucit_staged/stage01_baseline"
CK = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints")
PY = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"

TASKS = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions() -> dict:
    code = (
        "import sys, torch, peft, transformers; "
        "print(sys.version.split()[0]); print(torch.__version__); print(peft.__version__); "
        "print(transformers.__version__)"
    )
    raw = subprocess.run([PY, "-W", "ignore", "-c", code], capture_output=True, text=True).stdout
    out = [ln for ln in raw.splitlines() if ln and not ln.startswith(("[", "\x1b", " "))]
    ds_out = subprocess.run(
        [PY, "-W", "ignore", "-c",
         "import deepspeed; print(deepspeed.__version__ if hasattr(deepspeed,'__version__') else 'n/a')"],
        capture_output=True, text=True).stdout.strip().splitlines()
    ds = ds_out[-1] if ds_out else "n/a"
    return {
        "python": out[0], "torch": out[1], "peft": out[2], "transformers": out[3],
        "deepspeed": ds, "host": platform.node(),
    }


def collect(config: str, tag: str, task_count: int) -> dict:
    env = env_versions()
    tasks = []
    for t in range(1, task_count + 1):
        record_file = BASE / "generated" / f"run_{tag}_task{t}_record.json"
        if not record_file.exists():
            print(f"missing record: {record_file}")
            continue
        rec = json.loads(record_file.read_text())
        ckpt_dir = Path(rec["output_dir"])
        files = {}
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.iterdir()):
                if f.is_file():
                    files[f.name] = {"sha256": sha256(f), "size_bytes": f.stat().st_size}
        log_file = BASE / "logs" / f"{tag}_task{t}.log"
        timing = {}
        if log_file.exists():
            text = log_file.read_text(errors="ignore")
            m = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
            timing = {"first_ts": m[0] if m else None, "last_ts": m[-1] if m else None}
        tasks.append({**rec, "checkpoint_files": files, "timing": timing})
    return {"run": tag, "config": config, "env": env, "tasks": tasks,
            "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["original_batch", "gb24_matched"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-tasks", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    payload = collect(args.config, args.tag, args.num_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
