import pytest
from dataclasses import replace

from compose.expansion.residual_buffer import ResidualBuffer, ResidualRecord


def make():
    return ResidualRecord("x", 1, "ArxivQA", "train", "data:1", "dmh", "image:1", "prompt:1", "answer:1", "qhash",
                          (0,), (0,), True, 0.7, 1.0, 0.9, 0.1, 0.8, "2026-08-03T00:00:00Z", "commit", "config")


def test_buffer_deduplicates_identical_and_rejects_conflicting_rows():
    buffer = ResidualBuffer("predicted_buffer")
    assert buffer.add(make())
    assert not buffer.add(make())
    with pytest.raises(ValueError, match="conflicting"):
        buffer.add(replace(make(), residual_gain=9.0))
