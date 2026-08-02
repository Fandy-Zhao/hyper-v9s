import pytest
import torch

from compose.router.query_encoder import prompt_only_mask
from compose.router.validation import validate_query_payload


def test_answer_region_is_cut_at_token_level():
    mask = prompt_only_mask(torch.arange(12).view(2, 6), torch.ones(2, 6), torch.tensor([4, 2]))
    assert mask.tolist() == [[True, True, True, True, False, False], [True, True, False, False, False, False]]


@pytest.mark.parametrize("field", ["answer", "labels", "task_id", "oracle_set", "test_accuracy"])
def test_query_schema_rejects_answer_task_and_oracle_fields(field):
    payload = {"image_features": None, "text_features": None, "image_available": None, "text_available": None, field: 1}
    with pytest.raises(ValueError, match="forbidden"):
        validate_query_payload(payload)
