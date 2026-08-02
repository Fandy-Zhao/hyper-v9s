import pytest

from compose.router.anchor_memory import AnchorMemory, AnchorRecord


def record(sample, expert=0, task=0, split="train"):
    return AnchorRecord(sample, expert, task, split, "hash", tuple([0.0] * 128))


def test_anchor_capacity_dedup_and_round_trip():
    memory = AnchorMemory(2)
    assert memory.add(record("a"), 0)
    assert not memory.add(record("a"), 0)
    assert memory.add(record("b"), 0)
    assert not memory.add(record("c"), 0)
    restored = AnchorMemory()
    restored.load_state_dict(memory.state_dict())
    assert [item.sample_id for item in restored.records(0)] == ["a", "b"]


def test_anchor_rejects_non_train_and_future():
    with pytest.raises(ValueError, match="train"):
        record("x", split="validation")
    with pytest.raises(ValueError, match="future"):
        AnchorMemory().add(record("x", task=2), current_task=1)
