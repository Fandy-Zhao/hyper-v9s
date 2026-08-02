import torch

from compose.router.losses import empty_margin_loss, multi_positive_retrieval_loss


def test_pair_is_one_multi_positive_target_not_competing_ce():
    similarities = torch.tensor([[4.0, 3.0, -2.0]], requires_grad=True)
    positives = torch.tensor([[True, True, False]])
    visible = torch.ones_like(positives)
    good = multi_positive_retrieval_loss(similarities, positives, visible)
    bad = multi_positive_retrieval_loss(torch.tensor([[-2.0, -3.0, 4.0]]), positives, visible)
    assert good < bad
    good.backward()
    assert torch.isfinite(similarities.grad).all()


def test_empty_margin_penalizes_only_above_margin():
    visible = torch.tensor([[True, True], [True, True]])
    value = empty_margin_loss(torch.tensor([[0.1, 0.2], [0.4, 0.1]]), torch.tensor([True, True]), visible, 0.2)
    assert torch.allclose(value, torch.tensor(0.1))
