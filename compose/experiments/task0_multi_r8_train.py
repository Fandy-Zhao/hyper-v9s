"""Task0 multi rank-8 expert experiment -- training launcher.

Launches one independent ``train_compose`` job per cluster expert on its own
physical GPU (never GPUs 0-3).  Every job replicates the V6.2 S6 training
recipe exactly (same trainer, same arguments, same seed, same optimizer and
schedule); only the data slice (the expert's own cluster) and the expert id
differ.  There is no parameter sharing and no joint update between experts.

GPU assignment (spec §2):
  GPU4 -> single_r8 (1 job)
  GPU5 -> two_r8    (2 jobs, sequential)
  GPU6 -> four_r8   (4 jobs, sequential)
  GPU7 -> rank48    (1 job)  + later smoke/validation work
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR_PATH = os.path.join(BASE_MODEL, "mm_projector.bin")
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
SEED = 42
LEARNING_RATE = 2.0e-4
WARMUP_RATIO = 0.03
SCHEDULER = "cosine"
BATCH_SIZE = 1
GRAD_ACCUM = 8
MODEL_MAX_LENGTH = 2048
CACHE_DIR = "/tmp/compose_hf_cache"

# config name -> physical GPU (spec §2; never 0-3)
GPU_PLAN = {
    "single_r8": 4,
    "two_r8": 5,
    "four_r8": 6,
    "rank48": 7,
}


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def gpu_free(physical: int) -> bool:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(physical)],
        text=True,
    ).strip()
    return int(output) < 1000


def train_command(job: Dict[str, object], root: Path, epochs: int, max_steps: int, seed: int) -> List[str]:
    output_dir = root / "checkpoints" / str(job["config"]) / "expert_{}".format(int(job["expert_id"]))
    command = [
        PYTHON, "-m", "compose.train.train_compose",
        "--model_name_or_path", BASE_MODEL,
        "--vision_tower", VISION_TOWER,
        "--pretrain_mm_mlp_adapter", PROJECTOR_PATH,
        "--version", "v1",
        "--data_path", str(job["data_path"]),
        "--image_folder", IMAGE_FOLDER,
        "--compose-mode", "cluster_expert",
        "--compose-selection-manifest", str(job["manifest_path"]),
        "--compose-cluster-expert-ids", str(int(job["expert_id"])),
        "--compose_rank", str(int(job["rank"])),
        "--compose_alpha", str(float(job["alpha"])),
        "--output_dir", str(output_dir),
        "--bf16", "True",
        "--tf32", "True",
        "--num_train_epochs", str(epochs),
        "--learning_rate", str(LEARNING_RATE),
        "--warmup_ratio", str(WARMUP_RATIO),
        "--lr_scheduler_type", SCHEDULER,
        "--logging_steps", "1",
        "--save_steps", "999999",
        "--model_max_length", str(MODEL_MAX_LENGTH),
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "0",
        "--cache_dir", CACHE_DIR,
        "--seed", str(seed),
        "--report_to", "none",
        "--per_device_train_batch_size", str(BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRAD_ACCUM),
    ]
    if max_steps > 0:
        command += ["--max_steps", str(max_steps)]
    return command, output_dir


def launch(root_arg: str, epochs: int, max_steps: int, seed: int, gpu: str, only: str,
           config: str) -> None:
    root = Path(root_arg)
    manifest = read_json(root / "experiment_manifest.json")
    jobs = manifest["jobs"]
    by_config = {}
    for job in jobs:
        by_config.setdefault(str(job["config"]), []).append(job)
    jobs_by_gpu = {}
    for config_name, gpu_id in GPU_PLAN.items():
        jobs_by_gpu[gpu_id] = by_config.get(config_name, [])
    if config:
        if config not in by_config:
            raise RuntimeError("unknown config: {}".format(config))
        jobs_by_gpu = {int(gpu): by_config[config]} if gpu else {
            GPU_PLAN[config]: by_config[config]}
    if only:
        wanted = int(only)
        jobs_by_gpu = {k: v for k, v in jobs_by_gpu.items() if k == wanted}
    if gpu:
        jobs_by_gpu = {int(gpu): jobs_by_gpu.get(int(gpu), [])}

    # Preflight: refuse physical GPUs 0-3 (spec §2).
    for gpu_id in jobs_by_gpu:
        if gpu_id in (0, 1, 2, 3):
            raise RuntimeError("refusing physical GPU {} (spec §2)".format(gpu_id))

    reports = []
    for gpu_id, job_list in sorted(jobs_by_gpu.items()):
        if not job_list:
            continue
        if not gpu_free(gpu_id):
            raise RuntimeError("GPU {} is busy; refusing to co-locate jobs".format(gpu_id))
        for job in job_list:
            log_dir = root / "logs" / str(job["config"]) / "expert_{}".format(int(job["expert_id"]))
            log_dir.mkdir(parents=True, exist_ok=True)
            marker = log_dir / "complete.txt"
            if marker.is_file():
                print("skip completed job: {} expert {}".format(job["config"], job["expert_id"]), flush=True)
                reports.append({"config": job["config"], "expert_id": job["expert_id"], "status": "SKIPPED"})
                continue
            command, output_dir = train_command(job, root, epochs, max_steps, seed)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(log_dir / "command.json", {
                "config": job["config"], "expert_id": job["expert_id"],
                "gpu": gpu_id, "seed": seed, "epochs": epochs,
                "command": command,
            })
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
            print("[{}] start: {} expert {} on GPU {} ({} samples, {} steps/epoch)".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), job["config"], job["expert_id"],
                gpu_id, int(job["cluster_size"]), int(job["steps_per_epoch"]),
            ), flush=True)
            started = time.time()
            stdout = (log_dir / "train.stdout.log").open("w", encoding="utf-8")
            stderr = (log_dir / "train.stderr.log").open("w", encoding="utf-8")
            try:
                result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
            finally:
                stdout.close()
                stderr.close()
            elapsed = time.time() - started
            expected = output_dir / "expert_{:04d}.pt".format(int(job["expert_id"]))
            if result.returncode != 0:
                raise RuntimeError(
                    "training failed: {} expert {} rc={} (see {}/train.stderr.log)".format(
                        job["config"], job["expert_id"], result.returncode, log_dir))
            if not expected.is_file():
                raise RuntimeError("training produced no checkpoint: {}".format(expected))
            write_json(marker, {
                "config": job["config"], "expert_id": job["expert_id"],
                "gpu": gpu_id, "seed": seed, "epochs": epochs,
                "duration_seconds": elapsed,
                "checkpoint": str(expected),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            print("[{}] done: {} expert {} in {:.1f} min".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), job["config"], job["expert_id"], elapsed / 60.0,
            ), flush=True)
            reports.append({"config": job["config"], "expert_id": job["expert_id"],
                            "gpu": gpu_id, "duration_seconds": elapsed, "status": "TRAINED"})
    write_json(root / "logs" / "training_reports.json", reports)
    print("training launch complete", flush=True)


def status(root_arg: str) -> None:
    root = Path(root_arg)
    manifest = read_json(root / "experiment_manifest.json")
    for job in manifest["jobs"]:
        marker = root / "logs" / str(job["config"]) / "expert_{}".format(int(job["expert_id"])) / "complete.txt"
        if marker.is_file():
            info = read_json(marker)
            print("{:>9s} expert {} DONE {:.0f} min".format(
                job["config"], job["expert_id"], info["duration_seconds"] / 60.0))
        else:
            print("{:>9s} expert {} PENDING".format(job["config"], job["expert_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "status"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke: cap optimizer steps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", default="", help="restrict to one physical GPU")
    parser.add_argument("--only", default="", help="restrict to one config by GPU id")
    parser.add_argument("--config", default="", help="train one named config (with --gpu)")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.root, args.epochs, args.max_steps, args.seed, args.gpu, args.only, args.config)
    else:
        status(args.root)


if __name__ == "__main__":
    main()
