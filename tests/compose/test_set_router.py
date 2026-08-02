import torch

from compose.router.inference import predict_sets
from compose.router.set_router import ExpertSetRouter


def test_set_router_executes_zero_one_two_paths():
    torch.manual_seed(42)
    router = ExpertSetRouter(top_m=3)
    output = router(torch.randn(3, 128), torch.randn(3, 128), [0, 1, 2], torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool))
    assert output.candidate_expert_ids[0] == ()
    assert len(output.candidate_expert_ids[1]) == 1
    assert len(output.pair_ids[2]) == 3
    output.cardinality_logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    output.pair_scores[2] = torch.full_like(output.pair_scores[2], 9.0)
    assert list(map(len, predict_sets(output, 0.65))) == [0, 1, 2]
