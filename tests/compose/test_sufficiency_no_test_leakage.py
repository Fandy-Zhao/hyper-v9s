import pytest

from compose.expansion.validation import validate_head_payload


def test_predicted_sufficiency_payload_rejects_answer_teacher_task_fields():
    for field in ("answer_nll", "target", "teacher_set", "task_id", "labels"):
        with pytest.raises(ValueError, match="leaks"):
            validate_head_payload({"query": 1, field: 2})
