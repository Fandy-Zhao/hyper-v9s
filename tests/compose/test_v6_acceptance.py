"""Regression tests for compose.eval.v6_acceptance.

The performance matrix has a metric duality: Accuracy tasks (ImageNet-R /
ArxivQA / IconQA / CLEVR) are scored with case-insensitive exact match,
while Average tasks (VizWiz / Flickr30k) use the accepted original COCO
caption evaluator (llava.eval.eval_caption -> mean of Bleu_1..4, METEOR,
ROUGE_L, CIDEr). Scoring a caption task with exact match yields a
meaningless ~0% and must not resurface.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from compose.eval import v6_acceptance as acc  # noqa: E402

FORMAL_CONFIG = REPO / "configs/v6_ucit_formal_locked.yaml"

# Real Result.text from the seed-42 VizWiz formal predictions
VIZWIZ_RESULT = """Samples: 3000
Bleu_1: 59.01
Bleu_2: 40.65
Bleu_3: 27.62
Bleu_4: 18.33
METEOR: 22.20
ROUGE_L: 44.14
CIDEr: 58.09
Average: 38.58
"""


def test_metric_types_duality():
    """VizWiz/Flickr30k are Average (COCO caption), the rest Accuracy."""
    assert acc.METRIC_TYPES == [
        "Accuracy", "Accuracy", "Average", "Accuracy", "Accuracy", "Average"
    ]
    with open(FORMAL_CONFIG, "r", encoding="utf-8") as handle:
        locked = yaml.safe_load(handle)
    for i, task in enumerate(locked["task_sequence"]):
        assert task["metric_type"] == acc.METRIC_TYPES[i], task


def test_parse_coco_result():
    average, components = acc.parse_coco_result(VIZWIZ_RESULT)
    assert average == pytest.approx(0.3858)
    assert components["Bleu_1"] == pytest.approx(59.01)
    assert components["CIDEr"] == pytest.approx(58.09)
    assert components["METEOR"] == pytest.approx(22.20)
    # Average == mean of the 7 component percentages, rounded to 2 dp
    mean = sum(components.values()) / len(components)
    assert average == pytest.approx(mean / 100.0, rel=1e-4)


def test_parse_coco_result_missing_average_raises():
    with pytest.raises(RuntimeError, match="Average not computed"):
        acc.parse_coco_result("Bleu_1: 1.0\n")


def test_coco_annotation_exists_next_to_test_files():
    """The COCO caption annotation (images id 1..N matching prediction
    order) must sit next to each Average task's test file."""
    with open(FORMAL_CONFIG, "r", encoding="utf-8") as handle:
        locked = yaml.safe_load(handle)
    for i in range(6):
        if acc.METRIC_TYPES[i] != "Average":
            continue
        test_p = Path(locked["task_sequence"][i]["test_instructions"])
        coco_ann = test_p.with_name("val_coco_type_3000.json")
        assert coco_ann.is_file(), coco_ann
        payload = __import__("json").loads(coco_ann.read_text())
        assert "images" in payload and payload["images"][0]["id"] == 1


def test_exact_match_still_used_for_accuracy_tasks():
    """Sanity: exact_match_accuracy is the Accuracy-task path and raises
    on count mismatch."""
    annotations = [{"question_id": "1", "answer": "A"}]
    predictions = [{"question_id": "1", "text": "a"}]
    assert acc.exact_match_accuracy(annotations, predictions) == (1, 1)
    with pytest.raises(ValueError, match="prediction count"):
        acc.exact_match_accuracy(annotations, predictions[:0])
