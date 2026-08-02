import torch

from compose.router.set_losses import set_router_loss
from compose.router.set_router import ExpertSetRouter


def test_empty_single_pair_losses_are_finite_and_pair_order_insensitive():
    router = ExpertSetRouter(top_m=3)
    output = router(torch.randn(3, 128), torch.randn(3, 128), [0, 1, 2], torch.ones(3, 3, dtype=torch.bool))
    first, _ = set_router_loss(output, [(), (0,), (1, 2)], [0, 1, 2])
    second, _ = set_router_loss(output, [(), (0,), (2, 1)], [0, 1, 2])
    assert torch.isfinite(first)
    assert torch.equal(first, second)
