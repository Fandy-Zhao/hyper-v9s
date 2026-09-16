"""Formal UCIT continual evaluator (Hyper-LLaVA-compatible protocol).

After task ``t`` of the compose run completes, this harness scores the
task-``t`` snapshot on the test sets of every learned task ``j <= t``,
building the continual matrix row ``A[t][j]``.

Two hard compatibility rules (spec §4-§8):

1. Generation reuses ``compose.eval.eval_task`` with the SAME protocol as
   Hyper-LLaVA's ``llava.eval.model_answer`` (temperature 0 / do_sample=False,
   num_beams=1, max_new_tokens=128, conv_mode vicuna_v1, same instruction
   files, same image folder, same prompt construction).
2. Scoring executes the ORIGINAL Hyper-LLaVA evaluator modules verbatim
   (``llava.eval.eval_deepseek_r1`` for the four accuracy tasks,
   ``llava.eval.eval_caption`` for the two caption tasks) on the Compose
   prediction file. Compose never re-implements a scorer; it only parses the
   ``Result.text`` the original module writes.

Outputs (under ``<root>/evaluation``):

    predictions/t{t}/task{j}/answers.jsonl   router-based predictions (shards
                                             merged in record order)
    scores/t{t}/task{j}/Result.text          original-scorer result file
    scores/t{t}/task{j}/metric.json          parsed Accuracy / Average
    continual_matrix.json                    rows A[0..t][j] appended here
    hyper_result_root/{Dataset}/hyper-task{t+1}/Result.text
                                             mirrored layout consumed by the
                                             original summarize_continual_metrics.py
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from compose.data.records import question_text

PYTHON = os.environ.get("COMPOSE_PYTHON", sys.executable)
REPO_ROOT = str(Path(__file__).resolve().parents[2])
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR_PATH = os.path.join(BASE_MODEL, "mm_projector.bin")
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"

#: Original Hyper-LLaVA UCIT task registry (1-based task ids), copied from
#: scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py::TASKS so the
#: mirrored result-root layout matches the original script exactly.
HYPER_TASKS = [
    {"task_id": 1, "dataset": "ImageNet-R", "metric": "Accuracy"},
    {"task_id": 2, "dataset": "ArxivQA", "metric": "Accuracy"},
    {"task_id": 3, "dataset": "VizWiz", "metric": "Average"},
    {"task_id": 4, "dataset": "IconQA", "metric": "Accuracy"},
    {"task_id": 5, "dataset": "CLEVR-Math", "metric": "Accuracy"},
    {"task_id": 6, "dataset": "Flickr30k", "metric": "Average"},
]

#: Compose test instruction paths (0-based tasks), mirrors configs/compose_ucit.yaml.
TEST_FILES = [
    "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/ArxivQA/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/IconQA/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/test_3000.json",
]

#: Caption tasks use the COCO-style annotation file of the same test set.
VAL_COCO_FILES = [
    None,  # ImageNet-R
    None,  # ArxivQA
    "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/val_coco_type_3000.json",
    None,  # IconQA
    None,  # CLEVR
    "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/val_coco_type_3000.json",
]

_SCORE_RE = re.compile(
    r"^\s*(?:Accuracy|Average)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*$",
    re.IGNORECASE,
)


def _config(root: Path) -> Dict[str, Any]:
    """The resolved config persisted with the run (config copy), or the
    in-repo default when the run was created without one."""
    config_copy = root / "configs" / "config.yaml"
    if config_copy.is_file():
        return json.loads(config_copy.read_text(encoding="utf-8"))
    return {"eval": {"max_new_tokens": 128}, "data": {"seed": 42}}


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_manifest(root: Path, task_id: int) -> Dict[str, Any]:
    snapshot_dir = root / "task{}".format(task_id) / "snapshots" / "task{}".format(task_id)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "task {} snapshot missing (expected {})".format(task_id, manifest_path)
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _predictions_for(
    root: Path,
    task_id: int,
    eval_task_id: int,
    config: Dict[str, Any],
    gpus: List[str],
    reuse_s11: bool,
) -> Path:
    """Answers for ``eval_task_id`` with the task-``task_id`` snapshot.

    ``eval_task_id == task_id`` reuses the S11 ``eval_output/answers.jsonl``
    the runner already produced (same snapshot, same protocol). Historical
    tasks are generated here, sharded across ``gpus``; the chunks are merged
    in record order so the original caption scorer's sequential image_id
    mapping (row index -> image_id) stays valid.
    """
    out_dir = root / "evaluation" / "predictions" / "t{}".format(task_id) / "task{}".format(eval_task_id)
    answers = out_dir / "answers.jsonl"
    if answers.is_file():
        return answers

    s11_answers = root / "task{}".format(task_id) / "eval_output" / "answers.jsonl"
    if reuse_s11 and eval_task_id == task_id and s11_answers.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        data = s11_answers.read_bytes()
        answers.write_bytes(data)
        return answers

    snapshot_dir = root / "task{}".format(task_id) / "snapshots" / "task{}".format(task_id)
    router_checkpoint = snapshot_dir / "router_checkpoint.pt"
    manifest = _snapshot_manifest(root, task_id)
    pool_dir = manifest.get("pool_checkpoint_dir")
    if not pool_dir or not Path(pool_dir).is_dir():
        raise RuntimeError(
            "task {} snapshot has no resolvable pool_checkpoint_dir".format(task_id)
        )
    if not router_checkpoint.is_file():
        raise RuntimeError(
            "task {} snapshot has no router_checkpoint.pt".format(task_id)
        )

    test_file = TEST_FILES[eval_task_id]
    with open(test_file, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    num_chunks = max(1, min(len(gpus), 4))
    if len(records) < num_chunks:
        num_chunks = 1

    os.makedirs(out_dir, exist_ok=True)
    procs = []
    for index, gpu in enumerate(gpus[:num_chunks]):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        chunk_file = out_dir / "chunk_{}_{}.jsonl".format(num_chunks, index)
        command = [
            PYTHON, "-m", "compose.eval.eval_task",
            "--adapter-kind", "compose",
            "--model-path", BASE_MODEL,
            "--vision-tower", VISION_TOWER,
            "--projector-path", PROJECTOR_PATH,
            "--question-file", test_file,
            "--image-folder", IMAGE_FOLDER,
            "--checkpoint-dir", str(pool_dir),
            "--router-checkpoint", str(router_checkpoint),
            "--answers-file", str(chunk_file),
            "--run-summary-file", str(out_dir / "run_summary_{}.json".format(index)),
            "--device", "cuda:0",
            "--num-chunks", str(num_chunks),
            "--chunk-idx", str(index),
            "--max-new-tokens", str(int(config.get("eval", {}).get("max_new_tokens", 128))),
        ]
        procs.append(
            (index, gpu, chunk_file, subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        )
    for index, gpu, chunk_file, proc in procs:
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "eval_task chunk {}/{} on gpu {} failed:\n{}".format(
                    index, num_chunks, gpu, stderr.decode("utf-8", "replace")[-4000:]
                )
            )
    with open(answers, "w", encoding="utf-8") as output:
        for index, gpu, chunk_file, proc in procs:
            output.write(chunk_file.read_text(encoding="utf-8"))
    return answers


def _score_answers(
    root: Path,
    task_id: int,
    eval_task_id: int,
    answers: Path,
    annotation_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the ORIGINAL Hyper-LLaVA scorer on the prediction file and parse
    the metric from the Result.text it writes. ``annotation_file`` defaults
    to the task's official test/val_coco annotation (tests may pass a subset
    COCO annotation for caption tasks)."""
    score_dir = root / "evaluation" / "scores" / "t{}".format(task_id) / "task{}".format(eval_task_id)
    score_dir.mkdir(parents=True, exist_ok=True)
    test_file = TEST_FILES[eval_task_id]
    if VAL_COCO_FILES[eval_task_id] is not None:
        module = "llava.eval.eval_caption"
        annotation = annotation_file or VAL_COCO_FILES[eval_task_id]
        metric_name = "Average"
    else:
        module = "llava.eval.eval_deepseek_r1"
        annotation = annotation_file or test_file
        metric_name = "Accuracy"
    result = subprocess.run(
        [
            PYTHON, "-m", module,
            "--annotation-file", annotation,
            "--result-file", str(answers),
            "--output-dir", str(score_dir),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "original scorer {} failed:\n{}".format(module, result.stderr[-4000:])
        )
    result_text = score_dir / "Result.text"
    if not result_text.is_file():
        raise RuntimeError("{} did not write Result.text".format(module))
    value = None
    for line in result_text.read_text(encoding="utf-8").splitlines():
        match = _SCORE_RE.match(line)
        if match:
            value = round(float(match.group(1)), 6)
            break
    if value is None:
        raise RuntimeError(
            "no {} line in {}:\n{}".format(metric_name, result_text, result_text.read_text())
        )
    metric = {
        "task_id": eval_task_id,
        "dataset": HYPER_TASKS[eval_task_id]["dataset"],
        "metric": metric_name,
        "value": value,
        "score_unit": "percentage_points",
        "scorer": module,
        "annotation_file": annotation,
        "prediction_file": str(answers),
        "result_text_sha256": _sha256(str(result_text)),
    }
    with open(score_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump(metric, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metric


def _mirror_to_hyper_layout(root: Path, task_id: int, metrics: List[Dict[str, Any]]) -> None:
    """Mirror Result.text files into the layout the ORIGINAL
    summarize_continual_metrics.py expects:

        {result_root}/{dataset}/hyper-task{model_task}/Result.text

    Compose stage ``t`` (0-based) is Hyper ``model_task = t+1`` (1-based).
    """
    mirror_root = root / "evaluation" / "hyper_result_root"
    for metric in metrics:
        dataset = metric["dataset"]
        model_task = task_id + 1
        result_text = (
            root / "evaluation" / "scores" / "t{}".format(task_id)
            / "task{}".format(metric["task_id"]) / "Result.text"
        )
        target = mirror_root / dataset / "hyper-task{}".format(model_task) / "Result.text"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result_text.read_text(encoding="utf-8"), encoding="utf-8")


def _update_matrix(root: Path, task_id: int, metrics: List[Dict[str, Any]]) -> None:
    """Merge this invocation's cells into row ``task_id``.

    Merging rather than replacing is what lets a row be filled in more than one
    pass: the per-task pass writes the diagonal cell and the final sweep adds
    the cross-task cells beside it.  A measurement that is re-taken overwrites
    its own cell, so the row is still last-write-wins per cell, and a cell
    already scored is never silently dropped by a later, narrower invocation.

    The cell keys are strings on both sides of the merge.  JSON has no integer
    keys, so a row read back from the file arrives keyed by string; adding
    integer keys beside those would leave one dict holding both, and the
    ``sort_keys`` sort below would raise rather than write a row that has been
    through two passes.  Every reader of the file looks cells up by string --
    ``formal_ucit_summary._matrix_rows`` among them -- so that is the spelling
    the row is kept in.
    """
    matrix_path = root / "evaluation" / "continual_matrix.json"
    matrix = {"schema_version": 1, "rows": {}}
    if matrix_path.is_file():
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    row = {str(cell): metric for cell, metric in (matrix["rows"].get(str(task_id)) or {}).items()}
    row.update({str(int(metric["task_id"])): metric for metric in metrics})
    matrix["rows"][str(task_id)] = row
    matrix["final_row_task"] = task_id
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with open(matrix_path, "w", encoding="utf-8") as handle:
        json.dump(matrix, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="compose run root")
    parser.add_argument("--stage-task", type=int, required=True,
                        help="latest completed compose task (0..5)")
    parser.add_argument("--config", default="configs/compose_ucit.yaml")
    parser.add_argument("--gpus", default="4,5,6,7",
                        help="physical GPU ids for generation shards")
    parser.add_argument("--no-reuse-s11", action="store_true",
                        help="regenerate the current-task predictions instead "
                             "of reusing the runner's S11 answers")
    args = parser.parse_args()

    root = Path(args.root)
    config = _load_yaml(args.config)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]

    metrics = []
    for eval_task_id in range(0, args.stage_task + 1):
        answers = _predictions_for(
            root, args.stage_task, eval_task_id, config, gpus,
            reuse_s11=not args.no_reuse_s11,
        )
        metric = _score_answers(root, args.stage_task, eval_task_id, answers)
        metrics.append(metric)
        print(
            "A[{t}][{j}] {dataset} {metric} = {value} ({n} samples)".format(
                t=args.stage_task, j=eval_task_id, dataset=metric["dataset"],
                metric=metric["metric"], value=metric["value"],
                n=sum(1 for _ in open(answers, encoding="utf-8")),
            )
        )
    _update_matrix(root, args.stage_task, metrics)
    _mirror_to_hyper_layout(root, args.stage_task, metrics)
    print("continual matrix row {} written to {}".format(
        args.stage_task, root / "evaluation" / "continual_matrix.json"
    ))


if __name__ == "__main__":
    main()
