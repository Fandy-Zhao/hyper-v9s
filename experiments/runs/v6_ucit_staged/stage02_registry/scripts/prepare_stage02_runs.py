"""Create immutable Stage-02 launch configs from the frozen Stage-01 GB24 scripts."""

import json
from pathlib import Path


REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
SOURCE = REPO / "experiments/runs/v6_ucit_staged/stage01_baseline/generated"
ROOT = REPO / "experiments/runs/v6_ucit_staged/stage02_registry"
CHECKPOINT_ROOT = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage02_registry/checkpoints")
SOURCE_COMMIT = "5668cfd"
GPU_PHYSICAL = [4, 5, 6, 7]
GPU_LOGICAL = [0, 1, 2, 3]


for name in ("configs", "smoke", "mini2", "metrics", "manifests", "logs", "failures", "scripts"):
    (ROOT / name).mkdir(parents=True, exist_ok=True)


def transform(source_name, target_name, output_dir, artifact_dir, port, replacements=()):
    text = (SOURCE / source_name).read_text(encoding="utf-8")
    text = text.replace("--include localhost:4,5,6,7", "--include localhost:0,1,2,3")
    text = text.replace("llava/train/train_mem_MOE.py", str(ROOT / "scripts/stage02_train_entry.py"))
    text = text.replace("--master_port 29620", "--master_port {}".format(port))
    text = text.replace("--master_port 29651", "--master_port {}".format(port))
    text = text.replace("--master_port 29652", "--master_port {}".format(port))
    for old, new in replacements:
        text = text.replace(old, new)
    # Replace the one quoted output_dir regardless of its frozen Stage-01 suffix.
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("--output_dir "):
            line = '    --output_dir "{}" \\'.format(output_dir)
        lines.append(line)
    header = """# Stage 02 generated launch; source: {source}\nexport CUDA_VISIBLE_DEVICES=4,5,6,7\nexport V6_REGISTRY_ARTIFACT_DIR={artifact}\nexport V6_SOURCE_GIT_COMMIT={commit}\nexport V6_RUN_ID={run_id}\nexport V6_RUN_TIMESTAMP=2026-08-02T03:00:00+08:00\nmkdir -p \"$V6_REGISTRY_ARTIFACT_DIR\"\n""".format(
        source=source_name,
        artifact=artifact_dir,
        commit=SOURCE_COMMIT,
        run_id=target_name.removesuffix(".sh"),
    )
    target = ROOT / "configs" / target_name
    target.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o755)


transform(
    "run_smoke_gb24.sh",
    "run_smoke_gb24.sh",
    CHECKPOINT_ROOT / "smoke_gb24",
    ROOT / "smoke/runtime",
    29720,
)
transform(
    "run_mini2_gb24_task1.sh",
    "run_mini2_gb24_task1.sh",
    CHECKPOINT_ROOT / "mini2_gb24_task1_llava_lora_ours",
    ROOT / "mini2/task1_runtime",
    29751,
)
transform(
    "run_mini2_gb24_task2.sh",
    "run_mini2_gb24_task2.sh",
    CHECKPOINT_ROOT / "mini2_gb24_task2_llava_lora_ours",
    ROOT / "mini2/task2_runtime",
    29752,
    replacements=(
        (
            "/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints/mini2_gb24_task1_llava_lora_ours",
            str(CHECKPOINT_ROOT / "mini2_gb24_task1_llava_lora_ours"),
        ),
    ),
)

(ROOT / "configs/gpu_and_batch.json").write_text(
    json.dumps(
        {
            "physical_gpu_ids": GPU_PHYSICAL,
            "logical_gpu_ids": GPU_LOGICAL,
            "mapping": {str(logical): physical for logical, physical in zip(GPU_LOGICAL, GPU_PHYSICAL)},
            "selection_reason": "GPU 0-3 occupied by another user's openpi processes; GPU 4-7 idle",
            "world_size": 4,
            "per_device_batch": 2,
            "gradient_accumulation_steps": 3,
            "effective_global_batch": 24,
            "oom_fallback": {"per_device_batch": 1, "gradient_accumulation_steps": 6},
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
