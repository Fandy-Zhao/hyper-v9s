"""Generation/scoring and metric collation for controlled A/B runs."""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from compose.eval.formal_ucit_eval import BASE_MODEL, IMAGE_FOLDER, PROJECTOR_PATH, PYTHON, VISION_TOWER


TASK_NAMES = {0: "ImageNet-R", 1: "ArxivQA", 4: "CLEVR-Math"}
RUNS = [("A", task, rank, 2, f"A/task{task}_rank{rank}") for task in (0, 1, 4) for rank in (8, 16, 32)] + [("B", 0, rank, 1, f"B/task0_single_rank{rank}") for rank in (8, 16, 32)]
SCORE_RE = re.compile(r"^\s*Accuracy\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*$", re.I)


def snapshot(run_root: Path, task_id: int):
    directory = run_root / f"task{task_id}" / "snapshots" / f"task{task_id}"
    manifest = json.loads((directory / "manifest.json").read_text())
    return directory, manifest["pool_checkpoint_dir"]


def unique_questions(source: Path, target: Path):
    records = json.loads(source.read_text())
    for record in records:
        record["question_id"] = str(record.get("id", record.get("question_id")))
        if not record.get("answer"):
            responses = [
                turn.get("value")
                for turn in record.get("conversations", [])
                if turn.get("from") == "gpt" and turn.get("value")
            ]
            if not responses:
                raise ValueError(f"{source}: record {record['question_id']} has no ground-truth answer")
            record["answer"] = responses[-1]
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    return len(records)


def score(annotation: Path, answers: Path, score_dir: Path):
    score_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([PYTHON, "-m", "llava.eval.eval_deepseek_r1", "--annotation-file", str(annotation), "--result-file", str(answers), "--output-dir", str(score_dir)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    value = None
    for line in (score_dir / "Result.text").read_text().splitlines():
        match = SCORE_RE.match(line)
        if match: value = float(match.group(1)); break
    if value is None: raise RuntimeError("scorer produced no Accuracy")
    metric = {"metric": "Accuracy", "value": value, "scorer": "llava.eval.eval_deepseek_r1", "annotation_file": str(annotation), "prediction_file": str(answers)}
    (score_dir / "metric.json").write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n")
    return metric


def eval_split(run_root: Path, task_id: int, split: str, gpus):
    task_root = run_root / f"task{task_id}"
    out = run_root / "analysis" / split
    out.mkdir(parents=True, exist_ok=True)
    snapshot_dir, pool = snapshot(run_root, task_id)
    annotation = out / "questions.json"
    source = task_root / "data" / f"teacher_{split}.json"
    samples = unique_questions(source, annotation)
    answers = out / "answers.jsonl"
    if not answers.exists():
        chunks = min(4, len(gpus), samples)
        procs = []
        for index, gpu in enumerate(gpus[:chunks]):
            chunk = out / f"chunk_{chunks}_{index}.jsonl"
            summary = out / f"run_summary_{index}.json"
            command = [PYTHON, "-m", "compose.eval.eval_task", "--adapter-kind", "compose", "--model-path", BASE_MODEL, "--vision-tower", VISION_TOWER, "--projector-path", PROJECTOR_PATH, "--question-file", str(annotation), "--image-folder", IMAGE_FOLDER, "--checkpoint-dir", str(pool), "--router-checkpoint", str(snapshot_dir / "router_checkpoint.pt"), "--answers-file", str(chunk), "--run-summary-file", str(summary), "--device", "cuda:0", "--num-chunks", str(chunks), "--chunk-idx", str(index), "--max-new-tokens", "128"]
            procs.append((chunk, subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
        for chunk, proc in procs:
            _, stderr = proc.communicate()
            if proc.returncode: raise RuntimeError(stderr.decode("utf-8", "replace")[-4000:])
        with answers.open("w", encoding="utf-8") as handle:
            for chunk, _ in procs: handle.write(chunk.read_text())
    metric = score(annotation, answers, out / "score")
    summaries = [json.loads(path.read_text()) for path in sorted(out.glob("run_summary_*.json"))]
    aggregate = {"samples": samples, "metric": metric, "duration_seconds_parallel": max(item["duration_seconds"] for item in summaries), "samples_per_second": samples / max(item["duration_seconds"] for item in summaries), "peak_memory_bytes": max(item["peak_memory_bytes"] for item in summaries), "shards": len(summaries)}
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


def score_test(run_root: Path, task_id: int):
    task_root = run_root / f"task{task_id}"
    answers = task_root / "eval_output" / "answers.jsonl"
    config = yaml.safe_load((run_root / "config.yaml").read_text())
    annotation = Path(config["task_sequence"][task_id]["test_instructions"])
    metric = score(annotation, answers, run_root / "analysis" / "test" / "score")
    summary = json.loads((task_root / "eval_output" / "run_summary.json").read_text())
    aggregate = {"samples": summary["samples"], "metric": metric, "duration_seconds": summary["duration_seconds"], "samples_per_second": summary["samples_per_second"], "peak_memory_bytes": summary["peak_memory_bytes"]}
    target = run_root / "analysis" / "test"; target.mkdir(parents=True, exist_ok=True)
    (target / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


def training_stats(run_root: Path, task_id: int):
    trainer = json.loads((run_root / f"task{task_id}" / "lora" / "cluster_training" / "trainer_state.json").read_text())
    last = next(item for item in reversed(trainer["log_history"]) if "train_loss" in item)
    return float(last["train_loss"]), float(last["train_runtime"]), int(trainer["global_step"])


def rms_stats(run_root: Path, task_id: int):
    report = json.loads((run_root / f"task{task_id}" / "rms" / "rms_report.json").read_text())
    raw, calibrated, kappa = [], [], []
    for layers in report["per_expert"].values():
        for values in layers.values():
            raw.append(float(values["raw_rms"]))
            calibrated.append(float(values["calibrated_rms"]))
            kappa.append(float(values["kappa"]))
    mean = lambda values: sum(values) / len(values)
    return {
        "lora_raw_rms_mean": mean(raw),
        "lora_calibrated_rms_mean": mean(calibrated),
        "lora_kappa_mean": mean(kappa),
        "lora_clip_ratio_mean": mean([float(v) for v in report["clip_ratio"].values()]),
        "lora_dominance_ratio_mean": mean([float(v) for v in report["dominance_ratio"].values()]),
    }


def sampled_training_peak(study: Path, relative: str, run_root: Path, task_id: int):
    phase = relative.split("/", 1)[0]
    monitor = study / f"{phase}_gpu_samples.csv"
    contract = run_root / f"task{task_id}" / "distributed_training_contract.json"
    done = run_root / f"task{task_id}" / "stages" / "s6_cluster_training.done"
    if not (monitor.exists() and contract.exists() and done.exists()):
        return None
    group_4_7 = relative in {
        "A/task0_rank16", "A/task1_rank8", "A/task1_rank16", "A/task1_rank32",
        "A/task4_rank8", "A/task4_rank16", "A/task4_rank32",
        "B/task0_single_rank8", "B/task0_single_rank16", "B/task0_single_rank32",
    }
    gpu_ids = set(range(4, 8) if group_4_7 else range(4))
    start, end = contract.stat().st_mtime - 20, done.stat().st_mtime + 20
    peaks = []
    with monitor.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
            if start <= timestamp <= end and int(row["index"]) in gpu_ids:
                peaks.append(int(row["memory_used_mib"]) * 1024 * 1024)
    return max(peaks) if peaks else None


def collect(study: Path):
    rows = []
    for experiment, task_id, rank, expected_experts, relative in RUNS:
        run_root = study / relative
        formation = json.loads((run_root / f"task{task_id}" / "cluster" / "formation.json").read_text())
        experts = len(formation["formed_experts"])
        if experts != expected_experts: raise ValueError(f"{relative}: expected {expected_experts} experts, got {experts}")
        train_loss, training_time, steps = training_stats(run_root, task_id)
        train_nll = json.loads((run_root / "analysis" / "nll_train.json").read_text())
        val_nll = json.loads((run_root / "analysis" / "nll_val.json").read_text())
        train = json.loads((run_root / "analysis" / "train" / "aggregate.json").read_text())
        val = json.loads((run_root / "analysis" / "val" / "aggregate.json").read_text())
        test = json.loads((run_root / "analysis" / "test" / "aggregate.json").read_text())
        pool = snapshot(run_root, task_id)[1]
        pool_manifest = json.loads((Path(pool) / "compose_experts.json").read_text())
        config = yaml.safe_load((run_root / "config.yaml").read_text())
        trainable = 131_072_000 + experts * rank * 2_498_560
        evaluation_peak = max(train["peak_memory_bytes"], val["peak_memory_bytes"], test["peak_memory_bytes"])
        training_peak = sampled_training_peak(study, relative, run_root, task_id)
        row = {"experiment": experiment, "task_id": task_id, "task": TASK_NAMES[task_id], "num_experts": experts, "rank_per_expert": rank, "alpha": config["lora"]["alpha"], "alpha_over_rank": config["lora"]["alpha"] / rank, "total_rank": experts * rank, "adapter_params_in_snapshot": pool_manifest["metrics"]["adapter_parameter_count"], "trainable_params": trainable, "optimizer_steps": steps, "train_final_nll": train_loss, "train_routed_nll": train_nll["mean_nll"], "val_nll": val_nll["mean_nll"], "train_metric": train["metric"]["value"], "val_metric": val["metric"]["value"], "test_metric": test["metric"]["value"], "peak_vram_bytes": training_peak or evaluation_peak, "training_peak_vram_bytes": training_peak, "evaluation_peak_vram_bytes": evaluation_peak, "training_time_seconds": training_time, "training_samples_per_second": steps * 8 / training_time, "inference_samples_per_second": test["samples_per_second"]}
        row.update(rms_stats(run_root, task_id))
        rows.append(row)
    return rows


def summarize(study: Path):
    rows = collect(study)
    a = [row for row in rows if row["experiment"] == "A"]
    b_single = [row for row in rows if row["experiment"] == "B"]
    b_cluster = [row for row in a if row["task_id"] == 0 and row["rank_per_expert"] in (8, 16)]
    b = b_single + [{**row, "experiment": "B_clustered_reuse_A"} for row in b_cluster]
    for name, values in (("A_rank_ablation_summary.csv", a), ("B_task0_bootstrap_ablation.csv", b)):
        with (study / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    report_a = ["# Experiment A — Expert Rank Controlled Ablation", "", "| task | experts | rank | alpha/rank | train NLL | val NLL | train metric | val metric | test metric |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in a: report_a.append("| {task} | {num_experts} | {rank_per_expert} | {alpha_over_rank:.1f} | {train_final_nll:.4f} | {val_nll:.4f} | {train_metric:.2f} | {val_metric:.2f} | {test_metric:.2f} |".format(**row))
    (study / "A_rank_ablation_report.md").write_text("\n".join(report_a) + "\n")
    report_b = ["# Experiment B — Task0 Bootstrap Structure Ablation", "", "| structure | experts | rank | total rank | val metric | test metric | val NLL |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in b: report_b.append("| {} | {num_experts} | {rank_per_expert} | {total_rank} | {val_metric:.2f} | {test_metric:.2f} | {val_nll:.4f} |".format("single" if row["num_experts"] == 1 else "clustered", **row))
    (study / "B_task0_bootstrap_report.md").write_text("\n".join(report_b) + "\n")
    (study / "AB_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "A_rows": len(a), "B_rows": len(b)}))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("eval-split"); split.add_argument("--run-root", required=True); split.add_argument("--task-id", type=int, required=True); split.add_argument("--split", choices=("train", "val"), required=True); split.add_argument("--gpus", required=True)
    test = sub.add_parser("score-test"); test.add_argument("--run-root", required=True); test.add_argument("--task-id", type=int, required=True)
    summary = sub.add_parser("summarize"); summary.add_argument("--study-root", required=True)
    args = parser.parse_args()
    if args.command == "eval-split": eval_split(Path(args.run_root), args.task_id, args.split, [item for item in args.gpus.split(",") if item])
    elif args.command == "score-test": score_test(Path(args.run_root), args.task_id)
    else: summarize(Path(args.study_root))


if __name__ == "__main__":
    main()
