import pytest

from compose.expansion.residual_buffer import ResidualRecord
from compose.expansion.validation import validate_buffer_record


def make(task=1, split="train"):
    return ResidualRecord("x", task, "ArxivQA", split, "data:1", "dmh", "image:1", "prompt:1", "answer:1", "qhash",
                          (0,), (0,), True, 0.7, 1.0, 0.9, 0.1, 0.8, "2026-08-03T00:00:00Z", "commit", "config")


def test_buffer_rejects_current_or_future_teacher_expert():
    with pytest.raises(ValueError, match="future"):
        validate_buffer_record(make(task=1), {0: 1})


def test_buffer_rejects_current_or_future_predicted_expert():
    record = make(task=1)
    record = record.__class__(**{**record.__dict__, "teacher_set": (), "predicted_set": (0,)})
    with pytest.raises(ValueError, match="predicted set"):
        validate_buffer_record(record, {0: 1})


def test_buffer_record_rejects_non_train_split():
    with pytest.raises(ValueError, match="train"):
        make(split="validation")
