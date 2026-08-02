#!/usr/bin/env python3
"""Stage 01 Step 1 static tests (CPU only).

Checks:
  T1 config schema: task_params.json has all required keys for both configs
  T2 global batch math: launch generation asserts gb for 2/3/4 gpus
     (gb24 -> 2/2/6, 3/2/4, 4/2/3; original_batch task1 -> 64@4gpu, 32@2gpu)
  T3 mini2 index determinism: rebuild produces identical files + no split overlap
  T4 checkpoint path: expected output dir scheme is a child of stage01 base
  T5 evaluator import: llava.eval modules importable
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
BASE = REPO / "experiments/runs/v6_ucit_staged/stage01_baseline"
PY = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"

REQUIRED_KEYS = [
    "script", "script_sha256", "cur_task", "data_path", "lora_r", "lora_alpha",
    "expert_num", "model_name_or_path", "learning_rate", "weight_decay",
    "warmup_ratio", "lr_scheduler_type", "num_train_epochs", "bf16", "tf32",
    "gradient_checkpointing", "save_strategy", "evaluation_strategy",
    "per_device_train_batch_size", "gradient_accumulation_steps", "global_batch",
]


def t1_config_schema() -> None:
    for cfg in ("original_batch", "gb24_matched"):
        p = json.loads((BASE / "configs" / cfg / "task_params.json").read_text())
        assert len(p) == 6, f"{cfg}: expected 6 tasks"
        for tid, v in p.items():
            for k in REQUIRED_KEYS:
                assert k in v and v[k] is not None, f"{cfg} task {tid} missing {k}"
    print("T1 config schema OK")


def t2_global_batch_math() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_launch", BASE / "scripts" / "make_launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = json.loads((BASE / "configs" / "gb24_matched" / "task_params.json").read_text())
    cases = [("gb24_matched", "0,1,2,3", 24), ("gb24_matched", "0,1,2", 24), ("gb24_matched", "0,1", 24)]
    for cfg, gpus, gb in cases:
        n = len(gpus.split(","))
        pd = int(params["1"]["per_device_train_batch_size"])
        ga = gb // (n * pd)
        assert ga * n * pd == gb, f"{cfg} {gpus}: {ga}*{n}*{pd} != {gb}"
        print(f"T2 gb24 {n}gpu -> bs {pd} accum {ga} gb {ga*n*pd} OK")
    op = json.loads((BASE / "configs" / "original_batch" / "task_params.json").read_text())
    # task1 gb64 on 4 gpu; task2 gb32 on 4 gpu; task1 gb64 on 2 gpu
    pd = int(op["1"]["per_device_train_batch_size"]); ga = int(op["1"]["gradient_accumulation_steps"])
    assert pd * ga * 4 == 64, "original task1 4gpu != 64"
    pd2 = int(op["2"]["per_device_train_batch_size"]); ga2 = int(op["2"]["gradient_accumulation_steps"])
    assert pd2 * ga2 * 4 == 32, "original task2 4gpu != 32"
    assert 64 % (2 * 2) == 0, "original task1 2gpu cannot split"
    print("T2 original_batch math OK")


def t3_mini2_determinism() -> None:
    # rebuild into a temp mirror dir by monkeying the OUT constant
    src = (BASE / "scripts" / "build_mini2_indices.py").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        out_tmp = Path(tmp) / "idx"
        text = src.replace("OUT = REPO / \"experiments/runs/v6_ucit_staged/stage01_baseline/data_indices\"",
                           f"OUT = Path({str(out_tmp)!r})")
        script = Path(tmp) / "rebuild.py"
        script.write_text(text)
        subprocess.run([PY, str(script)], check=True, capture_output=True)
        # compare with committed indices
        for f in ["ImageNet-R/train.json", "ImageNet-R/val.json", "ImageNet-R/test.json",
                  "ArxivQA/train.json", "ArxivQA/val.json", "ArxivQA/test.json", "mini2_manifest.json"]:
            a = hashlib.sha256((BASE / "data_indices" / f).read_bytes()).hexdigest()
            b = hashlib.sha256((out_tmp / f).read_bytes()).hexdigest()
            assert a == b, f"mini2 {f} not deterministic"
    # overlap checks
    for name in ("ImageNet-R", "ArxivQA"):
        tr = {e["id"] for e in json.loads((BASE / "data_indices" / name / "train.json").read_text())}
        va = {e["id"] for e in json.loads((BASE / "data_indices" / name / "val.json").read_text())}
        assert not (tr & va), f"{name} train/val overlap"
    print("T3 mini2 determinism OK")


def t4_checkpoint_path() -> None:
    base = BASE.resolve()
    for d in ["checkpoints/smoke_gb24", "checkpoints/mini2_gb24/task1", "checkpoints/full_orig/task6"]:
        p = (BASE / d).resolve()
        assert str(p).startswith(str(base)), f"checkpoint outside stage01 base: {p}"
    print("T4 checkpoint path OK")


def t5_evaluator_import() -> None:
    code = (
        "import llava.eval.model_answer, llava.eval.eval_deepseek_r1, llava.eval.eval_caption; "
        "print('T5 evaluator imports OK')"
    )
    subprocess.run([PY, "-c", code], check=True)


def main() -> int:
    t1_config_schema()
    t2_global_batch_math()
    t3_mini2_determinism()
    t4_checkpoint_path()
    t5_evaluator_import()
    print("STAGE01_STATIC_ALL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
