import inspect

import torch

from compose.expansion.sufficiency_head import SufficiencyHead


def test_sufficiency_head_shape_and_answer_free_signature():
    head = SufficiencyHead()
    result = head(torch.randn(2, 128), torch.randn(2, 3), torch.randn(2, 5), *[torch.randn(2) for _ in range(4)])
    assert result.shape == (2,)
    names = set(inspect.signature(head.forward).parameters)
    assert not names & {"answer", "answer_nll", "target", "teacher_set", "task_id"}
