import torch

from compose.cli.train_set_router import continual_anchor_indices
from compose.router.set_router import ExpertSetRouter


def test_future_experts_never_enter_candidates_or_pairs():
    router = ExpertSetRouter(top_m=8)
    output = router(torch.randn(1, 128), torch.randn(4, 128), [0, 1, 2, 3], torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert set(output.candidate_expert_ids[0]) == {0, 1}
    assert output.pair_ids[0] == ((output.candidate_expert_ids[0][0], output.candidate_expert_ids[0][1]),)


def test_continual_anchor_replay_is_historical_member_only_and_bounded():
    bundle = {
        "task_ids": [0] * 40 + [1] * 2 + [2] + [3],
        "sample_ids": [f"sample-{index}" for index in range(44)],
        "oracle_sets": [(0,)] * 35 + [()] * 5 + [(0,), (1,), (0, 1), ()],
        "expert_metadata": [
            {"expert_id": 0, "creation_task": 0, "archived": False},
            {"expert_id": 1, "creation_task": 1, "archived": False},
            {"expert_id": 2, "creation_task": 2, "archived": False},
            {"expert_id": 3, "creation_task": 3, "archived": False},
        ],
    }
    indices, by_expert = continual_anchor_indices(bundle, current_task=3, max_per_expert=32, seed=42)
    assert indices[0] == 43
    assert len(by_expert["0"]) == 32
    assert 41 in by_expert["1"]
    assert all(bundle["task_ids"][index] < 3 for rows in by_expert.values() for index in rows)
    assert all(int(expert) in bundle["oracle_sets"][index] for expert, rows in by_expert.items() for index in rows)
    assert all(len(rows) <= 32 for rows in by_expert.values())
    assert by_expert["2"] == []
    assert "3" not in by_expert
    assert (indices, by_expert) == continual_anchor_indices(bundle, current_task=3, max_per_expert=32, seed=42)
