import ast
import json
import math
from pathlib import Path

import pytest

from compose.eval import no_router_oracle as oracle
from compose.oracle.candidate_sets import build_candidate_sets


def test_final_pool_enumerates_empty_ten_singles_and_45_pairs():
    candidates = build_candidate_sets(range(10))
    assert len(candidates) == 56
    assert sum(not value.expert_ids for value in candidates) == 1
    assert sum(len(value.expert_ids) == 1 for value in candidates) == 10
    pairs = [value for value in candidates if len(value.expert_ids) == 2]
    assert len(pairs) == 45
    assert len({value.expert_ids for value in pairs}) == 45
    assert all(left < right for left, right in (value.expert_ids for value in pairs))
    assert all(value.gates == pytest.approx((1 / math.sqrt(2),) * 2) for value in pairs)
    assert all(value.normalization == "l2" for value in pairs)


def test_chunk_bounds_are_complete_disjoint_and_ordered():
    bounds = [oracle._bounds(3000, 4, index) for index in range(4)]
    assert bounds == [(0, 750), (750, 1500), (1500, 2250), (2250, 3000)]
    uneven = [oracle._bounds(7, 4, index) for index in range(4)]
    assert uneven == [(0, 2), (2, 4), (4, 6), (6, 7)]


def test_gpu_guard_rejects_every_physical_gpu_outside_four_to_seven():
    assert oracle._validate_gpus("4,5,6,7") == ["4", "5", "6", "7"]
    with pytest.raises(ValueError):
        oracle._validate_gpus("0,4")
    with pytest.raises(ValueError):
        oracle._validate_gpus("4,4")


def test_oracle_execution_source_has_no_router_import_or_select_call():
    tree = ast.parse(Path(oracle.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "compose.router" not in (node.module or "")
        if isinstance(node, ast.Import):
            assert all("compose.router" not in alias.name for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "select"


def _formal_validation_records():
    return [
        {
            "id": "sample-{}".format(index),
            "image": "ImageNet-R/train/sample.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "answer-{}".format(index)},
            ],
        }
        for index in range(200)
    ]


def test_validation_materialization_is_frozen_and_leak_free(tmp_path):
    formal = tmp_path / "formal"
    output = tmp_path / "oracle"
    source = formal / "task0" / "data" / "teacher_val.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_formal_validation_records()), encoding="utf-8")
    result = oracle._prepare_validation(formal, output, 0)
    rows = json.loads(Path(result["questions"]).read_text(encoding="utf-8"))
    assert len(rows) == 200
    assert [row["question_id"] for row in rows] == [str(index) for index in range(200)]
    assert rows[0]["answer"] == "answer-0"
    assert Path(result["annotation"]) == Path(result["questions"])


def test_caption_validation_builds_deterministic_coco_annotation(tmp_path):
    formal = tmp_path / "formal"
    output = tmp_path / "oracle"
    source = formal / "task2" / "data" / "teacher_val.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_formal_validation_records()), encoding="utf-8")
    result = oracle._prepare_validation(formal, output, 2)
    annotation = json.loads(Path(result["annotation"]).read_text(encoding="utf-8"))
    assert len(annotation["images"]) == 200
    assert len(annotation["annotations"]) == 200
    assert annotation["annotations"][0] == {
        "caption": "answer-0",
        "category_id": 1,
        "id": 1,
        "image_id": 1,
    }
