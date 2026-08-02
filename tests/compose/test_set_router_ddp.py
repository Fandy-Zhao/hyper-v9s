import torch

from compose.router.checkpoint import state_fingerprint
from compose.router.set_router import ExpertSetRouter


def test_set_router_identical_seed_matches_across_ranks():
    torch.manual_seed(42)
    rank0 = ExpertSetRouter()
    torch.manual_seed(42)
    rank1 = ExpertSetRouter()
    assert state_fingerprint(rank0) == state_fingerprint(rank1)
