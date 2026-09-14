"""Evaluate the completed V7 six-task run on the 21 UCIT lower-triangle cells."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from compose.eval.formal_ucit_eval import (
    _mirror_to_hyper_layout,
    _score_answers,
    _update_matrix,
)


def lower_triangle_cells(num_tasks=6):
    return [(stage, task) for stage in range(num_tasks) for task in range(stage + 1)]


def selection_path(root, stage, task):
    return (
        root / "evaluation" / "selections" / "t{}".format(stage)
        / "task{}".format(task) / "selections.json"
    )


def generation_command(root, formal, method, stage, task, python, selection_manifest=None):
    task_root = root / "task{}".format(stage)
    task_config = formal["tasks"][task]
    data = formal["data"]
    answers = (
        root / "evaluation" / "predictions" / "t{}".format(stage)
        / "task{}".format(task) / "answers.jsonl"
    )
    summary = answers.with_name("run_summary.json")
    command = [
        python, "-m", "compose.eval.eval_task",
        "--adapter-kind", "compose",
        "--model-path", data["model_path"],
        "--checkpoint-dir", str(task_root / "committed"),
        "--projector-path", data["projector_path"],
        "--vision-tower", data["vision_tower"],
        "--question-file", task_config["test_file"],
        "--image-folder", data["image_folder"],
        "--answers-file", str(answers),
        "--run-summary-file", str(summary),
        "--runtime-contract", str(task_root / "data" / "runtime_contract.json"),
        "--device", "cuda:0",
    ]
    if selection_manifest is not None:
        # Cache mode (spec §21): committed-pool Top-2 selections were
        # precomputed from the fixed-query cache; eval_task loads no CLIP
        # query encoder and makes zero encoder calls.
        command.extend(["--selection-manifest", selection_manifest])
    else:
        query_contract = json.loads(
            (task_root / "data" / "query_contract.json").read_text(encoding="utf-8")
        )
        command.extend(
            [
                "--v7-key-state", str(task_root / "committed" / "v7_keys.pt"),
                "--query-vision-model", method["query"]["path"],
                "--query-backbone-hash", query_contract["backbone_hash"],
            ]
        )
    return answers, command


def _repo_root():
    return str(Path(__file__).resolve().parents[2])


def _ensure_cell_selections(root, formal, method, stage, task, gpu, python, cache_manifest):
    """Precompute one cell's committed-pool selections from the cache.

    Runs on the worker's GPU (``CUDA_VISIBLE_DEVICES``), mirroring the
    legacy live eval's device so routing numerics are reproduced exactly.
    """
    target = selection_path(root, stage, task)
    if target.is_file():
        return str(target)
    command = [
        python, "-m", "compose.v7.cached_selections",
        "--cache-manifest", cache_manifest,
        "--key-state", str(root / "task{}".format(stage) / "committed" / "v7_keys.pt"),
        "--questions", formal["tasks"][task]["test_file"],
        "--question-task-index", str(task),
        "--model-task-index", str(stage),
        "--output", str(target),
        "--audit-output", str(target.with_name("selections_audit.json")),
        "--backbone-path", method["query"]["path"],
        "--device", "cuda:0",
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        PYTHONPATH=_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = target.with_name("selections.log")
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(
            "V7 selections A[{}][{}] failed on GPU{}; see {}".format(
                stage, task, gpu, log_path
            )
        )
    return str(target)


def _evaluate_gpu_queue(gpu, cells, root, formal, method, python,
                        cache_manifest=None):
    results = []
    for stage, task in cells:
        metric_path = (
            root / "evaluation" / "scores" / "t{}".format(stage)
            / "task{}".format(task) / "metric.json"
        )
        if metric_path.is_file():
            results.append((stage, json.loads(metric_path.read_text(encoding="utf-8"))))
            continue
        if cache_manifest is not None:
            manifest = _ensure_cell_selections(
                root, formal, method, stage, task, gpu, python, cache_manifest
            )
        else:
            manifest = None
        answers, command = generation_command(
            root, formal, method, stage, task, python, selection_manifest=manifest
        )
        answers.parent.mkdir(parents=True, exist_ok=True)
        log_path = answers.with_name("generation.log")
        expected_count = len(json.loads(
            Path(formal["tasks"][task]["test_file"]).read_text(encoding="utf-8")
        ))
        existing_count = (
            sum(1 for line in answers.open(encoding="utf-8") if line.strip())
            if answers.is_file() else 0
        )
        if existing_count != expected_count:
            if answers.is_file():
                backup = answers.with_name(
                    "answers.incomplete_{}of{}.jsonl".format(
                        existing_count, expected_count
                    )
                )
                if backup.exists():
                    raise RuntimeError(
                        "refusing to overwrite prior incomplete answers: {}".format(
                            backup
                        )
                    )
                answers.replace(backup)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            with log_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    "V7 generation A[{}][{}] failed on GPU{}; see {}".format(
                        stage, task, gpu, log_path
                    )
                )
        # The source evaluator's module-level test paths refer to the old
        # server. Always score against the formal config's target-local test
        # asset. Caption tasks require the colocated COCO annotation rather
        # than their instruction JSON.
        annotation = Path(formal["tasks"][task]["test_file"])
        coco_annotation = annotation.with_name("val_coco_type_3000.json")
        if coco_annotation.is_file():
            annotation = coco_annotation
        metric = _score_answers(
            root, stage, task, answers, annotation_file=str(annotation)
        )
        results.append((stage, metric))
    return results


def _write_markdown(root, matrix):
    names = ["ImgNetR", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr"]
    lines = ["# V7 UCIT Continual Matrix", "", "| Stage | " + " | ".join(names) + " |", "|---|" + "---|" * 6]
    for stage in range(6):
        row = matrix["rows"].get(str(stage), {})
        values = [
            str(row[str(task)]["value"]) if str(task) in row else "—"
            for task in range(6)
        ]
        lines.append("| {} | {} |".format(stage, " | ".join(values)))
    path = root / "evaluation" / "continual_matrix.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--formal-config", default="configs/v7_ucit_formal.yaml")
    parser.add_argument("--method-config", default="configs/v7_global_coevolution.yaml")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument(
        "--stage-task", type=int,
        help="evaluate only the completed lower-triangle row A[t][0..t]; "
             "default evaluates all six completed rows",
    )
    parser.add_argument(
        "--cache-manifest",
        help="fixed-query cache manifest; cells then route from cache rows "
        "(--selection-manifest) with zero encoder calls instead of live CLIP",
    )
    parser.add_argument(
        "--backbone-path",
        help="frozen CLIP directory whose content hash binds the cache's "
        "test splits (required together with --cache-manifest)",
    )
    args = parser.parse_args()
    if (args.backbone_path is None) != (args.cache_manifest is None):
        raise ValueError("--cache-manifest and --backbone-path must be used together")
    if args.cache_manifest and not Path(args.cache_manifest).is_file():
        raise FileNotFoundError("query cache manifest not found: {}".format(args.cache_manifest))

    root = Path(args.root)
    formal = yaml.safe_load(Path(args.formal_config).read_text(encoding="utf-8"))
    method = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != len(set(gpus)):
        raise ValueError("formal V7 evaluation requires distinct GPUs")
    if args.cache_manifest is not None and not gpus:
        raise ValueError("cached formal evaluation requires at least one GPU")
    if args.cache_manifest is None and len(gpus) not in (3, 4):
        raise ValueError("formal V7 evaluation requires three or four distinct GPUs")
    if args.stage_task is None:
        stages = list(range(6))
    else:
        if not 0 <= args.stage_task < 6:
            raise ValueError("--stage-task must be in [0, 5]")
        stages = [args.stage_task]
    for stage in stages:
        if not (root / "task{}".format(stage) / "task_complete.json").is_file():
            raise FileNotFoundError("Task{} is not complete".format(stage))

    cells = [(stage, task) for stage in stages for task in range(stage + 1)]
    queues = [[] for _ in gpus]
    for index, cell in enumerate(cells):
        queues[index % len(gpus)].append(cell)
    collected = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                _evaluate_gpu_queue, gpu, queue, root, formal, method, args.python,
                args.cache_manifest,
            )
            for gpu, queue in zip(gpus, queues)
        ]
        for future in as_completed(futures):
            collected.extend(future.result())

    for stage in stages:
        metrics = [metric for owner, metric in collected if owner == stage]
        metrics.sort(key=lambda value: value["task_id"])
        if len(metrics) != stage + 1:
            raise RuntimeError("incomplete lower-triangle row {}".format(stage))
        _update_matrix(root, stage, metrics)
        _mirror_to_hyper_layout(root, stage, metrics)
    matrix_path = root / "evaluation" / "continual_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    _write_markdown(root, matrix)
    if args.stage_task is None:
        print("completed exactly 21 V7 lower-triangle cells")
    else:
        print("completed V7 lower-triangle row A[{}][0..{}]".format(args.stage_task, args.stage_task))


if __name__ == "__main__":
    main()
