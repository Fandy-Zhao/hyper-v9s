"""Evaluator parity + metric self-consistency tests for the formal UCIT run.

Requirements before the formal run (spec §5, §28, §29):

* For every task: the SAME prediction file scored by the ORIGINAL
  Hyper-LLaVA evaluator (llava.eval.eval_deepseek_r1 / eval_caption) and by
  the Compose formal wrapper (compose.eval.formal_ucit_eval) yields the same
  score (|A - B| < 1e-6).
* Continual metrics: the ORIGINAL summarize_continual_metrics.py and the
  wrapper recomputation agree on the same matrix (diff < 1e-8).
* The final per-task columns are A[5][j] (final row), never the diagonal.

These tests need the hyper conda env (torch 2.3.1, pycocoevalcap) and the
real UCIT instruction files under /data/dataset/zhaozhuofan/UCIT.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from compose.eval.formal_ucit_eval import (  # noqa: E402
    TEST_FILES,
    VAL_COCO_FILES,
    _score_answers,
    _update_matrix,
)
from compose.eval.formal_ucit_summary import (  # noqa: E402
    wrapper_metrics,
)

PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"

ACCURACY_TASKS = [0, 1, 3, 4]  # ImageNet-R, ArxivQA, IconQA, CLEVR
CAPTION_TASKS = [2, 5]  # VizWiz, Flickr30k
FIXTURE_SIZE = 30

ROOT = pytest.mark.skipif(
    not os.path.isdir("/data/dataset/zhaozhuofan/UCIT/instructions"),
    reason="UCIT instruction files not mounted",
)


def _fixture_predictions(task_id: int, tmp_dir) -> str:
    """30 predictions for the first 30 records of the task's test set:
    first 15 answer fields copied verbatim (correct), next 15 deliberately
    wrong strings (exercises the mismatch path)."""
    with open(TEST_FILES[task_id], "r", encoding="utf-8") as handle:
        records = json.load(handle)[:FIXTURE_SIZE]
    lines = []
    for index, record in enumerate(records):
        if task_id in ACCURACY_TASKS:
            text = record["answer"] if index < 15 else "FIXTURE_WRONG_ANSWER"
        else:
            captions = json.load(open(VAL_COCO_FILES[task_id], encoding="utf-8"))["annotations"]
            first_gt = captions[0]["caption"]
            text = first_gt if index < 15 else "a fixture wrong caption"
        lines.append(json.dumps({
            "question_id": str(record["question_id"]),
            "prompt": record["text"],
            "text": text,
            "model_id": "fixture",
            "metadata": {},
        }, ensure_ascii=False))
    path = os.path.join(tmp_dir, "predictions_task{}.jsonl".format(task_id))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _subset_coco_annotation(task_id: int, tmp_dir: str) -> str:
    """COCO-style annotation subset with image_ids 1..FIXTURE_SIZE (the
    prediction rows map to image_id = row_index + 1; the scorer's GT keys
    must match the prediction image ids exactly)."""
    full = json.load(open(VAL_COCO_FILES[task_id], encoding="utf-8"))
    subset = dict(full)
    subset["images"] = [img for img in full["images"] if int(img["id"]) <= FIXTURE_SIZE]
    subset["annotations"] = [
        ann for ann in full["annotations"] if int(ann["image_id"]) <= FIXTURE_SIZE
    ]
    path = os.path.join(tmp_dir, "val_coco_subset_task{}.json".format(task_id))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(subset, handle, ensure_ascii=False)
    return path


def _original_scorer(task_id: int, predictions: str, output_dir: str,
                     annotation: str = None) -> float:
    module = "llava.eval.eval_caption" if task_id in CAPTION_TASKS else "llava.eval.eval_deepseek_r1"
    annotation = annotation or (VAL_COCO_FILES[task_id] if task_id in CAPTION_TASKS else TEST_FILES[task_id])
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        [PYTHON, "-m", module, "--annotation-file", annotation,
         "--result-file", predictions, "--output-dir", output_dir],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    with open(os.path.join(output_dir, "Result.text"), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.lower().startswith("accuracy") or line.lower().startswith("average"):
                return round(float(line.split(":")[1].strip().rstrip("%")), 6)
    raise AssertionError("no metric line in Result.text")


@ROOT
@pytest.mark.parametrize("task_id", ACCURACY_TASKS + CAPTION_TASKS,
                         ids=lambda v: ["ImageNet-R", "ArxivQA", "VizWiz",
                                        "IconQA", "CLEVR", "Flickr30k"][v])
def test_scorer_parity_original_vs_compose_wrapper(tmp_path, task_id):
    """Same predictions -> same score through the original scorer and the
    Compose formal wrapper (spec §5: abs(score_A - score_B) < 1e-6)."""
    predictions = _fixture_predictions(task_id, str(tmp_path))
    subset = None
    if task_id in CAPTION_TASKS:
        subset = _subset_coco_annotation(task_id, str(tmp_path))
    score_a = _original_scorer(task_id, predictions, str(tmp_path / "original"), subset)
    score_b = _score_answers(
        tmp_path, task_id=0, eval_task_id=task_id,
        answers=os.path.join(tmp_path, "predictions_task{}.jsonl".format(task_id)),
        annotation_file=subset,
    )["value"]
    assert abs(score_a - score_b) < 1e-6, (score_a, score_b)


def _synthetic_matrix():
    return [
        [10.0, None, None, None, None, None],
        [11.0, 20.0, None, None, None, None],
        [12.0, 21.0, 30.0, None, None, None],
        [13.0, 22.0, 31.0, 40.0, None, None],
        [14.0, 23.0, 32.0, 41.0, 50.0, None],
        [15.0, 24.0, 33.0, 42.0, 51.0, 60.0],
    ]


def _write_mirror(root, rows, datasets, metrics):
    """Materialize the original script's expected result layout."""
    for t, row in enumerate(rows):
        for j, value in enumerate(row):
            if value is None:
                continue
            d = root / datasets[j]
            d.mkdir(parents=True, exist_ok=True)
            (d / "hyper-task{}".format(t + 1)).mkdir(parents=True, exist_ok=True)
            text = "{}: {:.2f}%\n".format(metrics[j], value)
            (d / "hyper-task{}".format(t + 1) / "Result.text").write_text(text)


def _metric(stage, cell, value):
    return {"task_id": cell, "value": value, "metric": "Accuracy", "dataset": "D",
            "scorer": "s", "stage": stage}


def test_a_row_is_filled_in_more_than_one_pass(tmp_path):
    """``_update_matrix`` merges; a later, narrower pass must not drop cells.

    A row is written twice on the per-task schedule -- the diagonal beside its
    own training, the cross-task cells in the sweep at the end -- so a writer
    that replaced the row would erase the first pass's cells the moment the
    second one ran, and the matrix would come out with fewer cells than the run
    actually measured.
    """
    _update_matrix(tmp_path, 2, [_metric(2, 2, 70.0)])
    _update_matrix(tmp_path, 2, [_metric(2, 0, 50.0), _metric(2, 1, 60.0)])
    matrix = json.loads((tmp_path / "evaluation" / "continual_matrix.json").read_text())

    assert sorted(matrix["rows"]["2"]) == ["0", "1", "2"]
    assert matrix["rows"]["2"]["2"]["value"] == 70.0
    assert matrix["rows"]["2"]["0"]["value"] == 50.0


def test_a_re_measured_cell_replaces_only_itself(tmp_path):
    """Merging is per cell, so a measurement re-taken still wins for its cell."""
    _update_matrix(tmp_path, 1, [_metric(1, 0, 10.0), _metric(1, 1, 20.0)])
    _update_matrix(tmp_path, 1, [_metric(1, 0, 11.0)])
    matrix = json.loads((tmp_path / "evaluation" / "continual_matrix.json").read_text())

    assert matrix["rows"]["1"]["0"]["value"] == 11.0
    assert matrix["rows"]["1"]["1"]["value"] == 20.0
    assert len(matrix["rows"]["1"]) == 2


def test_two_rows_do_not_displace_each_other(tmp_path):
    _update_matrix(tmp_path, 0, [_metric(0, 0, 1.0)])
    _update_matrix(tmp_path, 1, [_metric(1, 0, 2.0), _metric(1, 1, 3.0)])
    matrix = json.loads((tmp_path / "evaluation" / "continual_matrix.json").read_text())

    assert sorted(matrix["rows"]) == ["0", "1"]
    assert matrix["final_row_task"] == 1


@ROOT
def test_continual_metrics_self_consistency(tmp_path):
    """The ORIGINAL summarize_continual_metrics.py and the wrapper agree on
    the same matrix within 1e-8 (spec §28)."""
    from compose.eval.formal_ucit_summary import ORIGINAL_METRICS_SCRIPT

    datasets = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k"]
    metrics = ["Accuracy", "Accuracy", "Average", "Accuracy", "Accuracy", "Average"]
    rows = _synthetic_matrix()
    mirror = tmp_path / "mirror"
    _write_mirror(mirror, rows, datasets, metrics)

    result = subprocess.run(
        [PYTHON, ORIGINAL_METRICS_SCRIPT, "--result-root", str(mirror),
         "--num-tasks", "6", "--output-file", str(tmp_path / "orig.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    original = json.loads((tmp_path / "orig.json").read_text())["metrics"]
    wrapper = wrapper_metrics(rows)
    for key in ("MFN", "MAA", "MFT", "BWT"):
        assert abs(original[key] - wrapper[key]) < 1e-8, (key, original, wrapper)


@ROOT
def test_final_table_uses_final_row_not_diagonal(tmp_path):
    """spec §29: per-task columns of the final table come from A[5][j], never
    the MFT diagonal."""
    from compose.eval.formal_ucit_summary import _write_final_table

    rows = _synthetic_matrix()
    final_row = rows[-1]
    diagonal = [rows[i][i] for i in range(6)]
    assert final_row != diagonal
    metrics = {
        "MFN": 36.5, "MAA": 29.25, "MFT": 35.0, "BWT": -5.0,
        "final_per_task": final_row,
    }
    _write_final_table(tmp_path, metrics)
    with open(tmp_path / "evaluation" / "final_ucit_table.csv", "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    header = lines[0].split(",")
    compose = dict(zip(header, lines[2].split(",")))
    for j, expected in enumerate(final_row):
        assert float(compose["ImgNetR" if j == 0 else ["ImgNetR", "ArxivQA", "VizWiz",
                                                        "IconQA", "CLEVR", "Flickr"][j]]) == expected
    assert float(compose["MFT"]) == 35.0  # MFT stays the diagonal mean
