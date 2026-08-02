import torch

from compose.router.retrieval import retrieve_experts


def test_retrieval_is_deterministic_and_respects_visible_mask():
    queries = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = retrieve_experts(queries, keys, [7, 8, 9], torch.tensor([True, True, False]), 4)
    assert result.expert_ids.tolist() == [[7, 8]]
    assert result.visible_expert_ids == (7, 8)
