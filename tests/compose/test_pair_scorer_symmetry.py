import torch

from compose.router.pair_scorer import SymmetricPairScorer


def test_pair_score_is_exactly_symmetric():
    scorer = SymmetricPairScorer()
    q, a, b = torch.randn(3, 128), torch.randn(3, 128), torch.randn(3, 128)
    sa, sb = torch.randn(3), torch.randn(3)
    assert torch.equal(scorer(q, a, b, sa, sb), scorer(q, b, a, sb, sa))
