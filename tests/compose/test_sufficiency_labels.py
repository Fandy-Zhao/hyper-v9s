import pytest

from compose.expansion.sufficiency import teacher_sufficiency


def record(selected=(0,), gain=0.03, scope="historical_only", split="train"):
    return {"sample_id": "x", "temporal_scope": scope, "composition_mode": "direct_sum", "split": split,
            "selected_expert_ids": list(selected), "selected_mean_nll": 1.0 - gain, "selected_score": 1.0 - gain + 0.01 * len(selected),
            "empty": {"mean_nll": 1.0, "score": 1.0}}


def test_teacher_definition_requires_nonempty_better_and_raw_gain_threshold():
    assert teacher_sufficiency(record()).old_expert_sufficient
    assert not teacher_sufficiency(record(selected=())).old_expert_sufficient
    assert not teacher_sufficiency(record(gain=0.019)).old_expert_sufficient


def test_teacher_rejects_nonhistorical_and_test():
    with pytest.raises(ValueError): teacher_sufficiency(record(scope="post_task_diagnostic"))
    with pytest.raises(ValueError): teacher_sufficiency(record(split="test"))
