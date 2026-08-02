import torch

from compose.router.cardinality_head import CardinalityHead


def test_cardinality_head_outputs_zero_one_two_logits():
    output = CardinalityHead()(torch.randn(4, 128), torch.randn(4, 5), torch.tensor([1, 2, 3, 4]))
    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()
