#!/usr/bin/env python3
"""Stage 01: generate a launch script for one UCIT task training run.

Copies the official Task<N>.sh verbatim and overrides only:
  launcher gpus/master_port, per_device_train_batch_size,
  gradient_accumulation_steps, output_dir, data_path (optional),
  seed (fixed 42), max_steps (optional).

Enforces: 2 <= n_gpus <= 4, effective global batch == expected from config,
per-stage "previous task" env var, and records the actual command.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
BASE = REPO / "experiments/runs/v6_ucit_staged/stage01_baseline"
SEED = 42


def parse_args(argv):
    out = {}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--" and i + 1 < len(argv):
            out["--"] = argv[i + 1:]
            break
        if k.startswith("--") and i + 1 < len(argv):
            out[k] = argv[i + 1]
            i += 2
        else:
            out[k] = True
            i += 1
    return out


def main() -> int:
    args = parse_args(sys.argv[1:])
    task = int(args["--task"])
    config = args["--config"]
    gpu_list = args["--gpus"]
    tag = args.get("--tag", f"task{task}_{config}")
    max_steps = args.get("--max-steps")
    data_path_override = args.get("--data-path")
    output_dir = args.get("--output-dir")
    port = int(args.get("--port", str(29600 + task * 10)))

    n_gpus = len(gpu_list.split(","))
    if not 2 <= n_gpus <= 4:
        print(f"ERROR: n_gpus={n_gpus} not in [2,4]", file=sys.stderr)
        return 1

    params = json.loads((BASE / "configs" / config / "task_params.json").read_text())
    p = params[str(task)]
    expected_gb = int(p["global_batch"])
    per_dev = int(p["per_device_train_batch_size"])
    grad_accum = expected_gb // (n_gpus * per_dev)
    if grad_accum * n_gpus * per_dev != expected_gb:
        print(f"ERROR: cannot split global_batch {expected_gb} over {n_gpus} gpus x bs {per_dev}", file=sys.stderr)
        return 1

    src = REPO / "scripts" / "Hyper" / "Train_UCIT" / f"Task{task}.sh"
    text = src.read_text()

    def sub(old, new):
        nonlocal text
        if old not in text:
            print(f"ERROR: pattern not found in {src.name}: {old}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(old, new)

    # launcher line: pin the hyper-env deepspeed binary (conda env not on PATH)
    text = re.sub(r"^deepspeed\s", "/home/zhaozhuofan/miniconda3/envs/hyper/bin/deepspeed ", text, flags=re.MULTILINE)
    text = re.sub(r"--include\s+localhost:[\d,]+", f"--include localhost:{gpu_list}", text)
    text = re.sub(r"--master_port\s+\d+", f"--master_port {port}", text)
    # batch / accum / output
    sub("--per_device_train_batch_size", f"--per_device_train_batch_size_NEW")
    text = re.sub(r"--per_device_train_batch_size_NEW\s+\d+", f"--per_device_train_batch_size {per_dev}", text)
    sub("--gradient_accumulation_steps", "--gradient_accumulation_steps_NEW")
    text = re.sub(r"--gradient_accumulation_steps_NEW\s+\d+", f"--gradient_accumulation_steps {grad_accum}", text)
    sub('--output_dir "$OUTPUT_DIR"', f'--output_dir "{output_dir}"')
    if '--output_dir "${OUTPUT_DIR}"' in text:
        sub('--output_dir "${OUTPUT_DIR}"', f'--output_dir "{output_dir}"')
    if data_path_override:
        sub(p["data_path"], data_path_override)
    # seed + max_steps appended before --report_to none end
    extra = f" --seed {SEED}"
    if max_steps:
        extra += f" --max_steps {max_steps}"
    text = text.replace("--report_to none", "--report_to none" + extra)

    # previous task env
    if task > 1:
        prev_dir = args.get("--prev-task-path")
        if prev_dir:
            text = re.sub(
                r'PREVIOUS_TASK_MODEL_PATH="\$\{PREVIOUS_TASK_MODEL_PATH[^}]*\}"',
                f'PREVIOUS_TASK_MODEL_PATH="{prev_dir}"',
                text,
            )

    out_dir = BASE / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"run_{tag}.sh"
    out_file.write_text(text)
    out_file.chmod(0o755)

    record = {
        "task": task,
        "config": config,
        "tag": tag,
        "n_gpus": n_gpus,
        "gpu_list": gpu_list,
        "per_device_train_batch_size": per_dev,
        "gradient_accumulation_steps": grad_accum,
        "global_batch": grad_accum * n_gpus * per_dev,
        "expected_global_batch": expected_gb,
        "seed": SEED,
        "max_steps": int(max_steps) if max_steps else None,
        "data_path": data_path_override or p["data_path"],
        "output_dir": output_dir,
        "master_port": port,
        "source_script": str(src.relative_to(REPO)),
        "source_script_sha256": p["script_sha256"],
    }
    (BASE / "generated" / f"run_{tag}_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"wrote {out_file.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
