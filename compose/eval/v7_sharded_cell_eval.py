"""Deterministic multi-GPU sharded formal V7 evaluation for one matrix cell."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from compose.eval.v7_formal_ucit_eval import (
    _ensure_cell_selections,
    _mirror_to_hyper_layout,
    _repo_root,
    _score_answers,
    _update_matrix,
    generation_command,
)


def _sample_id(record):
    return str(record.get("question_id", record.get("id")))


def _backup_incomplete(answers, expected_count):
    if not answers.exists():
        return None
    count = sum(1 for line in answers.open(encoding="utf-8") if line.strip())
    if count == expected_count:
        raise RuntimeError("canonical answers already complete; refuse to replace {}".format(answers))
    backup = answers.with_name("answers.interrupted_{}of{}.jsonl".format(count, expected_count))
    suffix = 1
    while backup.exists():
        backup = answers.with_name("answers.interrupted_{}of{}.{}.jsonl".format(count, expected_count, suffix))
        suffix += 1
    answers.replace(backup)
    return backup


def _merge(parts, records, output):
    expected = [_sample_id(record) for record in records]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise RuntimeError("test JSON has non-unique sample IDs; unsafe to merge")
    by_id = {}
    for part in parts:
        if not part.is_file():
            raise RuntimeError("missing shard output {}".format(part))
        with part.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample = str(row.get("question_id"))
                if sample not in expected_set:
                    raise RuntimeError("{}:{} unexpected question_id {}".format(part, line_number, sample))
                if sample in by_id:
                    raise RuntimeError("duplicate question_id {} across shards".format(sample))
                by_id[sample] = row
    missing = [sample for sample in expected if sample not in by_id]
    if missing or len(by_id) != len(expected):
        raise RuntimeError("incomplete shard merge: got {}, expected {}, missing first {}".format(len(by_id), len(expected), missing[:5]))
    temporary = output.with_name(output.name + ".merging")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in expected:
            handle.write(json.dumps(by_id[sample], ensure_ascii=False) + "\n")
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--formal-config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--backbone-path", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    formal = yaml.safe_load(Path(args.formal_config).read_text(encoding="utf-8"))
    method = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < 2 or len(gpus) != len(set(gpus)):
        raise ValueError("need at least two distinct GPUs for sharded mode")
    if not Path(args.cache_manifest).is_file() or not Path(args.backbone_path).is_dir():
        raise FileNotFoundError("cache manifest or bound backbone missing")
    if not (root / "task{}".format(args.stage) / "task_complete.json").is_file():
        raise FileNotFoundError("stage is not complete")
    metric_path = root / "evaluation" / "scores" / "t{}".format(args.stage) / "task{}".format(args.task) / "metric.json"
    if metric_path.is_file():
        print("metric already exists: {}".format(metric_path))
        return

    manifest = _ensure_cell_selections(root, formal, method, args.stage, args.task, gpus[0], args.python, args.cache_manifest)
    answers, command = generation_command(root, formal, method, args.stage, args.task, args.python, selection_manifest=manifest)
    answers.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(formal["tasks"][args.task]["test_file"]).read_text(encoding="utf-8"))
    backup = _backup_incomplete(answers, len(records))
    if backup:
        print("preserved interrupted output {}".format(backup))

    processes, parts = [], []
    log_path = answers.with_name("generation.sharded.log")
    for index, gpu in enumerate(gpus):
        part = answers.with_name("answers.shard{}_of{}.jsonl".format(index, len(gpus)))
        summary = answers.with_name("run_summary.shard{}_of{}.json".format(index, len(gpus)))
        parts.append(part)
        shard_command = list(command)
        shard_command[shard_command.index("--answers-file") + 1] = str(part)
        shard_command[shard_command.index("--run-summary-file") + 1] = str(summary)
        shard_command.extend(["--num-chunks", str(len(gpus)), "--chunk-idx", str(index)])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", ""))
        processes.append((index, gpu, subprocess.Popen(shard_command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    errors = []
    with log_path.open("a", encoding="utf-8") as log:
        for index, gpu, proc in processes:
            captured, _ = proc.communicate()
            log.write("\n===== shard {}/{} GPU{} =====\n{}".format(index, len(gpus), gpu, captured))
            if proc.returncode:
                errors.append("shard {}/{} GPU{} exit {}".format(index, len(gpus), gpu, proc.returncode))
    if errors:
        raise RuntimeError("; ".join(errors) + "; see {}".format(log_path))
    _merge(parts, records, answers)
    annotation = Path(formal["tasks"][args.task]["test_file"])
    coco_annotation = annotation.with_name("val_coco_type_3000.json")
    if coco_annotation.is_file():
        annotation = coco_annotation
    metric = _score_answers(root, args.stage, args.task, answers, annotation_file=str(annotation))
    metrics = []
    for task in range(args.stage + 1):
        path = root / "evaluation" / "scores" / "t{}".format(args.stage) / "task{}".format(task) / "metric.json"
        if not path.is_file():
            raise RuntimeError("row metric missing after cell score: {}".format(path))
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    metrics.sort(key=lambda value: value["task_id"])
    _update_matrix(root, args.stage, metrics)
    _mirror_to_hyper_layout(root, args.stage, metrics)
    print("completed sharded A[{}][{}]: {}".format(args.stage, args.task, metric))


if __name__ == "__main__":
    main()
