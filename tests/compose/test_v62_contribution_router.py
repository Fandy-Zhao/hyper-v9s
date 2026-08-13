import hashlib
import inspect

import torch
from torch.nn import functional as F

from compose.router.contribution import (
    contribution_record,
    refine_new_keys_from_contribution,
    train_contribution_set_router,
    update_route_anchors,
)
from compose.router.router import ComposeRouter
from compose.teacher.oracle_set import OracleConfig
from compose.teacher.teacher import ComposeTeacherSearcher


def _search(nll, delta=0.2):
    return ComposeTeacherSearcher(
        OracleConfig(
            oracle_name="v62_test", composition_mode="rms_calibrated",
            lambda_expert=0.0, delta_pair_raw=delta,
            top_k_for_pair=4, max_pairs=6,
        ),
        "v62", 1, top_m=8,
    ).search_from_nll("sample", 1, [0, 1], nll, 2, {})


def _fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def test_pair_without_strict_positive_conditional_gain_downgrades_to_single():
    result = _search({(): 3.0, (0,): 2.0, (1,): 2.4, (0, 1): 1.8}, delta=0.2)
    assert result.teacher_set == (0,)


def test_pair_with_positive_conditional_gain_is_allowed():
    result = _search({(): 3.0, (0,): 2.0, (1,): 2.4, (0, 1): 1.7}, delta=0.2)
    assert result.teacher_set == (0, 1)


def test_contribution_record_uses_actual_teacher_membership():
    record = contribution_record(4, (3,), [3, 4], 3.0, {3: 2.0, 4: 2.5}, {})
    assert record["cluster_label"] == 4
    assert record["contribution_label"] == [3]
    assert record["agreement"] is False


def test_refinement_changes_only_new_keys_and_preserves_historical_key():
    router = ComposeRouter()
    router.add_expert(0, 0, "old", torch.randn(128))
    router.add_expert(1, 1, "new", torch.randn(128))
    old_hash = _fingerprint(router.key_store.keys["0"])
    queries = F.normalize(torch.randn(3, 128), dim=1)
    records = [
        {"contribution_label": [1]},
        {"contribution_label": []},
        {"contribution_label": [1]},
    ]
    result = refine_new_keys_from_contribution(queries, records, router.key_store.keys, [1])
    assert _fingerprint(router.key_store.keys["0"]) == old_hash
    assert result["historical_keys_touched"] is False
    assert result["positive_count"] == 2


def test_learned_router_checkpoint_executes_empty_single_pair_without_answer_or_task_id(tmp_path):
    torch.manual_seed(2)
    router = ComposeRouter(top_m=3)
    for expert_id in range(3):
        router.add_expert(expert_id, 0, str(expert_id), torch.randn(128))
    queries = F.normalize(torch.randn(12, 128), dim=1)
    targets = [()] * 4 + [(0,)] * 4 + [(1, 2)] * 4
    metrics = train_contribution_set_router(
        router.set_router, queries, router.key_store.normalized(router.expert_ids),
        router.expert_ids, targets, epochs=2,
    )
    router.set_router_enabled = True
    assert metrics["router_trainable_parameters"] < 1_000_000
    names = set(inspect.signature(router.select).parameters)
    assert not names & {"answer", "labels", "targets", "task_id", "oracle_set"}
    state = router.state_dict_extra(4, "cfg")
    restored = ComposeRouter()
    restored.load_state_dict_extra(state)
    assert restored.set_router_enabled is True
    assert all(len(value) <= 2 for value in restored.select(queries[:3], restored.expert_ids).sets)


def test_train_only_route_anchors_are_bounded_and_checkpointed():
    queries = F.normalize(torch.randn(12, 128), dim=1)
    targets = [()] * 4 + [(0,)] * 4 + [(0, 1)] * 4
    anchors = update_route_anchors([], queries, targets, ["s%02d" % i for i in range(12)], 2)
    assert {size: sum(len(value["target"]) == size for value in anchors) for size in range(3)} == {0: 2, 1: 2, 2: 2}
    router = ComposeRouter()
    router.route_anchors = list(anchors)
    restored = ComposeRouter()
    restored.load_state_dict_extra(router.state_dict_extra(1, "cfg"))
    assert restored.route_anchors == router.route_anchors
