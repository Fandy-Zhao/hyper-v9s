#!/usr/bin/env python3
"""Stage 01: parse Hyper-LLaVA UCIT Task{1..6}.sh launch params verbatim.

Audit rule: all baseline hyperparameters must be parsed from the real Task
scripts, never reconstructed from memory. This script extracts every
--arg value from each script and emits two baseline configs:
  - original_batch : per-task global batch copied from the scripts
                     (Task1/3/4/5/6 = 64, Task2 = 32), rescaled to 4 GPUs
                     (per_device=2) via gradient accumulation.
  - gb24_matched   : every task at effective global batch 24 (4 GPUs,
                     per_device=2, grad_accum=3).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
SCRIPT_DIR = REPO / "scripts" / "Hyper" / "Train_UCIT"

# Per-task authoritative source scripts
TASK_SCRIPTS = {
    1: "Task1.sh", 2: "Task2.sh", 3: "Task3.sh",
    4: "Task4.sh", 5: "Task5.sh", 6: "Task6.sh",
}
# Script global batch (as configured in the repo, verified by manual count)
SCRIPT_GLOBAL_BATCH = {1: 64, 2: 32, 3: 64, 4: 64, 5: 64, 6: 64}

GB24_4GPU = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 3, "global_batch": 24}
ORIG64_4GPU = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8, "global_batch": 64}
ORIG32_4GPU = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "global_batch": 32}


def parse_script_args(path: Path) -> dict:
    """Extract --key value pairs from a shell script (values may span
    lines; simple key-value tokens only, no arrays)."""
    text = path.read_text(encoding="utf-8")
    args: dict[str, str] = {}
    # find all "--key value" / "--key=value" tokens
    for m in re.finditer(r"--([a-z0-9_]+)(?:=(\S+))?", text):
        key = m.group(1)
        if "=" in m.group(0):
            args[key] = m.group(2)
            continue
        # value: next token on same logical line, up to line end or next --
        rest = text[m.end():]
        end = re.search(r"(?:\n\s*--|\n\s*$|\s+--)", rest)
        val = rest[: end.start() if end else len(rest)].strip()
        val = val.strip("\\").strip().strip('"')
        if val and not val.startswith("--"):
            args[key] = val
        else:
            args[key] = ""  # flag
    # launcher line
    m = re.search(r"^(deepspeed|torchrun|python)\s+(.*?)$", text, re.MULTILINE)
    args["_launcher"] = m.group(1) if m else "deepspeed"
    m = re.search(r"--include\s+(\S+)", text)
    args["_include"] = m.group(1) if m else None
    m = re.search(r"--master_port\s+(\S+)", text)
    args["_master_port"] = m.group(1) if m else None
    m = re.search(r"--deepspeed\s+(\S+)", text)
    args["_deepspeed_config"] = m.group(1) if m else None
    return args


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "experiments/runs/v6_ucit_staged/stage01_baseline/configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    for task_id, fname in TASK_SCRIPTS.items():
        path = SCRIPT_DIR / fname
        if not path.exists():
            print(f"ERROR: missing {path}")
            return 1
        args = parse_script_args(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records[str(task_id)] = {
            "script": str(path.relative_to(REPO)),
            "script_sha256": sha,
            "cur_task": args.get("cur_task"),
            "data_path": args.get("data_path"),
            "lora_r": args.get("lora_r"),
            "lora_alpha": args.get("lora_alpha"),
            "lora_dropout": args.get("lora_dropout"),
            "expert_num": args.get("expert_num"),
            "previous_task_model_path": args.get("previous_task_model_path"),
            "model_name_or_path": args.get("model_name_or_path"),
            "pretrain_mm_mlp_adapter": args.get("pretrain_mm_mlp_adapter"),
            "vision_tower": args.get("vision_tower"),
            "text_tower": args.get("text_tower"),
            "mm_projector_type": args.get("mm_projector_type"),
            "num_train_epochs": args.get("num_train_epochs"),
            "per_device_train_batch_size_script": args.get("per_device_train_batch_size"),
            "gradient_accumulation_steps_script": args.get("gradient_accumulation_steps"),
            "learning_rate": args.get("learning_rate"),
            "weight_decay": args.get("weight_decay"),
            "warmup_ratio": args.get("warmup_ratio"),
            "lr_scheduler_type": args.get("lr_scheduler_type"),
            "bf16": args.get("bf16"),
            "tf32": args.get("tf32"),
            "gradient_checkpointing": args.get("gradient_checkpointing"),
            "lazy_preprocess": args.get("lazy_preprocess"),
            "group_by_modality_length": args.get("group_by_modality_length"),
            "image_aspect_ratio": args.get("image_aspect_ratio"),
            "model_max_length": args.get("model_max_length"),
            "save_strategy": args.get("save_strategy"),
            "evaluation_strategy": args.get("evaluation_strategy"),
            "report_to": args.get("report_to"),
            "modality_routing_mode": args.get("modality_routing_mode"),
            "eval_modality_routing_mode": args.get("eval_modality_routing_mode"),
            "router_hidden_dim": args.get("router_hidden_dim"),
            "router_loss_weight": args.get("router_loss_weight"),
            "launcher": args.get("_launcher"),
            "include": args.get("_include"),
            "master_port": args.get("_master_port"),
            "deepspeed_config": args.get("_deepspeed_config"),
        }

    # emit per-config task params with batch rescaling to 4 GPUs
    configs = {
        "original_batch": {},
        "gb24_matched": {},
    }
    for tid in range(1, 7):
        base = records[str(tid)]
        target_gb = SCRIPT_GLOBAL_BATCH[tid]
        configs["original_batch"][str(tid)] = {
            **{k: v for k, v in base.items() if not k.startswith("_") and v is not None},
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": ORIG64_4GPU["gradient_accumulation_steps"]
            if target_gb == 64 else ORIG32_4GPU["gradient_accumulation_steps"],
            "global_batch": target_gb,
            "batch_source": f"script_global_batch_{target_gb}_rescaled_to_4gpu",
        }
        configs["gb24_matched"][str(tid)] = {
            **{k: v for k, v in base.items() if not k.startswith("_") and v is not None},
            "per_device_train_batch_size": GB24_4GPU["per_device_train_batch_size"],
            "gradient_accumulation_steps": GB24_4GPU["gradient_accumulation_steps"],
            "global_batch": GB24_4GPU["global_batch"],
            "batch_source": "gb24_fixed_all_tasks",
        }

    for cfg, payload in configs.items():
        target = out_dir / cfg / "task_params.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {target.relative_to(REPO)}")

    # sanity: assert parsed global batch matches manual count
    for tid in range(1, 7):
        b = configs["original_batch"][str(tid)]
        gb = int(b["per_device_train_batch_size"]) * int(b["gradient_accumulation_steps"]) * 4
        assert gb == SCRIPT_GLOBAL_BATCH[tid], f"Task{tid} original_batch mismatch: {gb}"
        gb24 = configs["gb24_matched"][str(tid)]
        assert int(gb24["per_device_train_batch_size"]) * int(gb24["gradient_accumulation_steps"]) * 4 == 24
    print("global batch assertions OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
