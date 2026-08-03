"""Resumable one-process-per-GPU scheduler with an authorized 4-7 fallback."""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path


PRIMARY_GPUS = (0, 1, 2, 3)
FALLBACK_GPUS = (4, 5, 6, 7)
ALLOWED_GPUS = PRIMARY_GPUS + FALLBACK_GPUS


def validate_devices(values):
    devices = tuple(int(value) for value in values)
    if not devices or len(devices) != len(set(devices)) or any(device not in ALLOWED_GPUS for device in devices):
        raise ValueError("devices must be a unique nonempty subset of physical GPUs 0-7")
    return devices


def resolve_devices(specification, min_free_mib, memory_reader=None):
    if specification != "auto":
        return validate_devices(int(value) for value in specification.split(","))
    reader = memory_reader or free_memory
    primary = tuple(device for device in PRIMARY_GPUS if reader(device) >= min_free_mib)
    if primary:
        return primary
    fallback = tuple(device for device in FALLBACK_GPUS if reader(device) >= min_free_mib)
    if not fallback:
        raise RuntimeError("no primary or authorized fallback GPU has enough free memory")
    return fallback


def load_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    required = {"id", "stage", "config", "seed", "gpu_requirement", "command", "output_dir", "dependencies", "status"}
    ids = set()
    for task in tasks:
        missing = required - set(task)
        if missing:
            raise ValueError("task is missing fields {}".format(sorted(missing)))
        if task["id"] in ids:
            raise ValueError("duplicate task id {}".format(task["id"]))
        ids.add(task["id"])
        if int(task["gpu_requirement"]) != 1:
            raise ValueError("this queue supports one physical GPU per task")
    unknown = {dep for task in tasks for dep in task["dependencies"] if dep not in ids}
    if unknown:
        raise ValueError("unknown dependencies {}".format(sorted(unknown)))
    return tasks


def free_memory(device):
    command = [
        "nvidia-smi", "--id={}".format(device),
        "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ]
    return int(subprocess.check_output(command, text=True).strip().splitlines()[0])


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def output_complete(task):
    required = task.get("completion_files", ["summary.json", "per_sample.jsonl"])
    output = Path(task["output_dir"])
    if not all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required):
        return False
    summary = output / "summary.json"
    if summary.is_file():
        try:
            return json.loads(summary.read_text()).get("status") == "COMPLETED"
        except (ValueError, OSError):
            return False
    return True


def archive_partial_output(output, attempt):
    """Move an OOM attempt aside so a retry starts clean without losing evidence."""
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return None
    candidate = output.with_name("{}.oom_attempt{}".format(output.name, attempt))
    suffix = 1
    while candidate.exists():
        candidate = output.with_name("{}.oom_attempt{}_{}".format(output.name, attempt, suffix))
        suffix += 1
    output.rename(candidate)
    output.mkdir(parents=True, exist_ok=True)
    return str(candidate)


class Queue:
    def __init__(self, tasks, output_root, min_free_mib, poll_seconds):
        self.tasks = {task["id"]: task for task in tasks}
        self.output_root = Path(output_root)
        self.state_dir = self.output_root / "logs" / "scheduler_state"
        self.min_free_mib = int(min_free_mib)
        self.poll_seconds = float(poll_seconds)
        self.lock = threading.Lock()
        self.running = set()
        self.failed = set()
        self.completed = {task_id for task_id, task in self.tasks.items() if output_complete(task)}

    def claim(self):
        with self.lock:
            for task_id, task in self.tasks.items():
                if task_id in self.completed or task_id in self.running or task_id in self.failed:
                    continue
                if all(dep in self.completed for dep in task["dependencies"]):
                    self.running.add(task_id)
                    return task
            return None

    def finish(self, task_id, success):
        with self.lock:
            self.running.discard(task_id)
            (self.completed if success else self.failed).add(task_id)

    def done(self):
        with self.lock:
            terminal = self.completed | self.failed
            blocked = {
                task_id for task_id, task in self.tasks.items()
                if any(dep in self.failed for dep in task["dependencies"])
            }
            return len(terminal | blocked) == len(self.tasks) and not self.running

    def run_task(self, task, device):
        task_id = task["id"]
        output = Path(task["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        log_dir = self.output_root / "logs" / task["stage"] / task_id
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "resolved_config.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n")
        batch_sizes = task.get("oom_batch_sizes", [task.get("batch_size", 8)])
        attempts = []
        success = False
        for attempt, batch_size in enumerate(batch_sizes, 1):
            command = str(task["command"]).format(batch_size=batch_size)
            env = os.environ.copy()
            env.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(device),
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "8",
            })
            started = time.time()
            stdout_path = log_dir / "attempt{}_stdout.log".format(attempt)
            stderr_path = log_dir / "attempt{}_stderr.log".format(attempt)
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = subprocess.run(command, shell=True, env=env, stdout=stdout, stderr=stderr)
            record = {
                "attempt": attempt,
                "batch_size": batch_size,
                "command": command,
                "checkpoint": task.get("checkpoint"),
                "seed": task["seed"],
                "cuda_visible_devices": str(device),
                "exit_code": process.returncode,
                "elapsed_seconds": time.time() - started,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            stderr_text = stderr_path.read_text(errors="replace")
            success = process.returncode == 0 and output_complete(task)
            is_oom = "out of memory" in stderr_text.lower()
            if is_oom and not success and attempt < len(batch_sizes):
                record["archived_partial_output"] = archive_partial_output(output, attempt)
            attempts.append(record)
            if success or not is_oom:
                break
        state = {
            "id": task_id,
            "stage": task["stage"],
            "status": "COMPLETED" if success else "FAILED",
            "attempts": attempts,
        }
        atomic_json(self.state_dir / "{}.json".format(task_id), state)
        self.finish(task_id, success)

    def worker(self, device):
        while True:
            if self.done():
                return
            if free_memory(device) < self.min_free_mib:
                time.sleep(self.poll_seconds)
                continue
            task = self.claim()
            if task is None:
                time.sleep(self.poll_seconds)
                continue
            self.run_task(task, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--min-free-mib", type=int, default=18000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    devices = resolve_devices(args.devices, args.min_free_mib)
    tasks = load_tasks(args.manifest)
    snapshot = {str(device): free_memory(device) for device in devices}
    print(json.dumps({"allowed_devices": devices, "free_memory_mib": snapshot, "tasks": len(tasks)}, sort_keys=True))
    if args.preflight_only:
        return
    queue = Queue(tasks, args.output_root, args.min_free_mib, args.poll_seconds)
    workers = [threading.Thread(target=queue.worker, args=(device,), daemon=False) for device in devices]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if queue.failed:
        raise SystemExit("failed tasks: {}".format(sorted(queue.failed)))


if __name__ == "__main__":
    main()
