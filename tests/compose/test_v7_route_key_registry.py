"""Route-key registry: one expert owns several keys, frozen canonical + learnable reuse.

Covers the four required CPU tests (routing dedup, optimizer audit, checkpoint
round-trip, non-reusable negative) plus the task-5 / commit-retention semantics
and the key-loss gradient target.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from compose.adapters.lora import ComposeLinear
from compose.train.profiler import TrainingProfiler
from compose.v7.config import V7Config
from compose.v7.hf_trainer import V7ComposeTrainer
from compose.v7.pool import (
    KEY_LIFECYCLE_HISTORICAL,
    V7ExpertKeyPool,
    initialize_candidate_keys,
    initialize_current_task_key,
)
from compose.v7.routing import GlobalTop2Router
from compose.v7.training import selected_current_key_loss, trainable_selection
from compose.v7.workflow import prepare_candidate_pool

DIM = 1536


def basis(index, dim=DIM):
    value = torch.zeros(dim)
    value[index % dim] = 1.0
    return value


def key_with_cosine(axis, cosine, seed_index):
    """A unit key whose cosine against ``basis(axis)`` is exactly ``cosine``."""
    value = cosine * basis(axis) + (1.0 - cosine ** 2) ** 0.5 * basis(seed_index)
    return F.normalize(value, dim=0)


def make_linear(shape=(4, 8)):
    base = nn.Linear(shape[1], shape[0], bias=False)
    nn.init.zeros_(base.weight)
    base.requires_grad_(False)
    return ComposeLinear(base, rank=4, alpha=8.0)


def add_expert(layer, expert_id, trainable):
    expert = layer.add_expert(expert_id)
    expert.requires_grad_(bool(trainable))
    return expert


class _StubModel(nn.Module):
    """Registers the key pool and the LoRA layers as named parameters."""

    def __init__(self, pool, layers, extra_trainable=False):
        super().__init__()
        self.v7_key_pool = pool
        self.layers = nn.ModuleDict({name: layer for name, layer in layers.items()})
        if extra_trainable:
            self.unknown = nn.Parameter(torch.zeros(3))


def stub_trainer(pool, layers, extra_trainable=False):
    """A ``V7ComposeTrainer`` with only the attributes ``create_optimizer`` reads."""
    trainer = V7ComposeTrainer.__new__(V7ComposeTrainer)
    trainer.optimizer = None
    trainer.v7_key_pool = pool
    trainer.expert_pool = SimpleNamespace(manager=SimpleNamespace(layers=layers))
    trainer.v7_config = V7Config()
    trainer.args = SimpleNamespace(weight_decay=0.0)
    trainer.profiler = TrainingProfiler(None)
    trainer.model = _StubModel(pool, layers, extra_trainable=extra_trainable)
    return trainer


def small_pool(candidates=2):
    """1 reusable historical + 1 non-reusable historical + N candidates."""
    pool = V7ExpertKeyPool()
    pool.add(0, basis(0), origin_task=0, lifecycle="historical", trainable=False)
    pool.add(1, basis(1), origin_task=0, lifecycle="historical", trainable=False)
    pool.add_reuse_key(
        0, 1, initialize_current_task_key(basis(0), 0.01, 12345), trainable=True
    )
    for offset in range(candidates):
        pool.add(
            2 + offset, basis(2 + offset), origin_task=1, lifecycle="current",
            trainable=True,
        )
    return pool


def candidate_optimizer_fixture(candidates=(2, 3), historical=(0, 1)):
    layers = {"proj": make_linear()}
    for expert_id in list(candidates) + list(historical):
        add_expert(layers["proj"], expert_id, trainable=expert_id in candidates)
    return layers


# ----------------------------------------------------------------------
# 1. routing must aggregate by expert before Top-K
# ----------------------------------------------------------------------
def test_route_key_dedup_one_expert_cannot_take_both_slots():
    pool = V7ExpertKeyPool()
    # Expert 3: canonical A0 (0.95) + reuse A1 (0.93); Expert 4: B0 (0.90);
    # Expert 5: C0 (0.50).  A key-level Top-2 would return {A0, A1}.
    pool.add(3, key_with_cosine(0, 0.95, 1), origin_task=0, lifecycle="historical",
             trainable=False)
    pool.add_reuse_key(3, 1, key_with_cosine(0, 0.93, 2), trainable=True)
    pool.add(4, key_with_cosine(0, 0.90, 3), origin_task=0, lifecycle="historical",
             trainable=False)
    pool.add(5, key_with_cosine(0, 0.50, 4), origin_task=0, lifecycle="historical",
             trainable=False)
    query = basis(0).unsqueeze(0)

    # A key-level Top-2 over the four keys would have been {A0(0.95), A1(0.93)}.
    key_level = torch.topk(torch.tensor([0.95, 0.93, 0.90, 0.50]), k=2).indices.tolist()
    assert key_level == [0, 1]

    result = GlobalTop2Router(pool)(query)
    selected = result.expert_ids.tolist()

    assert selected == [[3, 4]], selected
    assert selected[0][0] != selected[0][1], "expert 3 must not fill both slots"
    assert result.key_ids[0][0] == "3", "canonical key A0 is the winner for expert 3"
    assert result.key_ids[0][1] == "4"
    assert result.key_types[0] == ("canonical", "canonical")
    # The reported similarities are the per-expert maxima, not key-level cosines.
    scores = result.similarities.detach().tolist()[0]
    assert scores[0] == pytest.approx(0.95, abs=1e-5)
    assert scores[1] == pytest.approx(0.90, abs=1e-5)
    assert pool.keys["e3_t1_reuse"].requires_grad
    # The expert is still historical: dedup must not relabel anybody.
    assert pool.historical_ids == (3, 4, 5) and pool.current_ids == ()


def test_winning_key_id_switches_to_the_reuse_key_when_it_wins():
    pool = V7ExpertKeyPool()
    pool.add(3, key_with_cosine(0, 0.80, 1), origin_task=0, lifecycle="historical",
             trainable=False)
    pool.add_reuse_key(3, 1, key_with_cosine(0, 0.95, 2), trainable=True)
    pool.add(4, key_with_cosine(0, 0.50, 3), origin_task=0, lifecycle="historical",
             trainable=False)
    pool.add(5, key_with_cosine(0, 0.40, 4), origin_task=0, lifecycle="historical",
             trainable=False)

    result = GlobalTop2Router(pool)(basis(0).unsqueeze(0))
    assert result.expert_ids.tolist()[0][0] == 3
    assert result.key_ids[0][0] == "e3_t1_reuse"
    assert result.key_types[0][0] == "reuse"


def test_route_types_stay_expert_level_when_a_reuse_key_wins():
    pool = V7ExpertKeyPool()
    pool.add(0, key_with_cosine(0, 0.80, 1), origin_task=0, lifecycle="historical",
             trainable=False)
    pool.add(1, key_with_cosine(0, 0.10, 2), origin_task=0, lifecycle="historical",
             trainable=False)
    pool.add_reuse_key(0, 1, key_with_cosine(0, 0.95, 3), trainable=True)
    pool.add(2, key_with_cosine(0, 0.60, 4), origin_task=1, lifecycle="current",
             trainable=True)

    result = GlobalTop2Router(pool)(basis(0).unsqueeze(0))
    assert result.key_ids[0][0] == "e0_t1_reuse"
    # ... and the route is still Old(historical 0) + New(candidate 2).
    assert result.route_types == ("OldNew",)
    assert result.expert_ids.tolist() == [[0, 2]]
    assert 0 in pool.historical_ids and 0 not in pool.current_ids


# ----------------------------------------------------------------------
# 2. optimizer audit
# ----------------------------------------------------------------------
def test_optimizer_contains_reuse_key_candidates_and_never_history():
    pool = small_pool()
    layers = candidate_optimizer_fixture()
    trainer = stub_trainer(pool, layers)

    optimizer = trainer.create_optimizer()
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}

    for key_id in ("2", "3", "e0_t1_reuse"):
        assert id(pool.keys[key_id]) in optimized, key_id
    assert id(pool.keys["0"]) not in optimized, "canonical historical key must be absent"
    assert id(pool.keys["1"]) not in optimized
    for expert_id in (0, 1):
        for parameter in layers["proj"].experts[str(expert_id)].parameters():
            assert id(parameter) not in optimized, "historical LoRA must be absent"
    for expert_id in (2, 3):
        for parameter in layers["proj"].experts[str(expert_id)].parameters():
            assert id(parameter) in optimized, "candidate LoRA must be present"
    assert pool.keys["e0_t1_reuse"].requires_grad
    assert pool.keys["0"].requires_grad is False


def test_optimizer_refuses_unknown_trainable_parameters():
    pool = small_pool()
    layers = candidate_optimizer_fixture()
    trainer = stub_trainer(pool, layers, extra_trainable=True)
    with pytest.raises(AssertionError, match="refuses non-current-Key/LoRA"):
        trainer.create_optimizer()


def test_key_loss_updates_reuse_key_but_never_the_canonical_key():
    pool = small_pool()
    query = basis(0).unsqueeze(0)
    result = GlobalTop2Router(pool)(query)
    # The frozen canonical key of expert 0 wins the slot; expert 1 is second.
    assert result.key_ids[0][0] == "0"
    assert result.expert_ids.tolist() == [[0, 1]]
    loss, per_sample = selected_current_key_loss(query, result.expert_ids, pool)
    assert float(loss) > 0.0
    loss.backward()

    # Expert 0 was reached through its FROZEN canonical key, and yet its
    # learnable reuse key is the one that learns -- this is the whole point.
    assert pool.keys["0"].grad is None, "frozen canonical key must get no gradient"
    assert pool.keys["1"].grad is None
    assert pool.keys["e0_t1_reuse"].grad is not None
    assert float(pool.keys["e0_t1_reuse"].grad.norm()) > 0.0
    assert trainable_selection(result, pool) == ("e0_t1_reuse",)
    assert per_sample.shape == (1,)


# ----------------------------------------------------------------------
# 3. checkpoint round-trip
# ----------------------------------------------------------------------
def test_checkpoint_round_trip_preserves_the_key_registry(tmp_path):
    pool = small_pool()
    pool.commit(
        retained_ids=(2, 3),
        metrics={2: {"keep": True}, 3: {"keep": True}},
        reuse_key_retention={0: True},
    )
    state = pool.export_state()
    assert state["schema_version"] == 2 and set(state["route_keys"]) == set(pool.key_ids)
    target = tmp_path / "v7_keys.pt"
    torch.save(state, target)

    restored = V7ExpertKeyPool.from_state(torch.load(target, weights_only=False))

    assert restored.key_ids == pool.key_ids
    assert restored.expert_ids == pool.expert_ids
    assert restored.trainable_key_ids == pool.trainable_key_ids == set()
    for key_id in pool.key_ids:
        left, right = pool.route_keys[key_id], restored.route_keys[key_id]
        assert (left.key_type, left.expert_id, left.origin_task, left.lifecycle,
                left.trainable) == (right.key_type, right.expert_id, right.origin_task,
                                    right.lifecycle, right.trainable)
        assert torch.allclose(pool.keys[key_id], restored.keys[key_id], atol=1e-6)
    assert restored.route_keys["e0_t1_reuse"].key_type == "reuse"
    assert restored.route_keys["e0_t1_reuse"].expert_id == 0
    assert restored.route_keys["e0_t1_reuse"].lifecycle == KEY_LIFECYCLE_HISTORICAL
    assert restored.metadata[0]["reuse_keys"] == {
        "1": {"key_id": "e0_t1_reuse", "origin_task": 1}
    }


def test_round_trip_keeps_a_live_reuse_key_trainable():
    pool = small_pool()
    restored = V7ExpertKeyPool.from_state(pool.export_state())
    assert restored.trainable_key_ids == {"2", "3", "e0_t1_reuse"}
    assert restored.keys["e0_t1_reuse"].requires_grad
    assert restored.keys["0"].requires_grad is False
    assert restored.trainable_keys_of(0) == ("e0_t1_reuse",)


def test_legacy_schema_one_state_still_loads():
    """A pre-registry v7_keys.pt round-trips as canonical keys."""
    legacy = {
        "schema_version": 1,
        "query_dim": DIM,
        "pool_version": 2,
        "keys": {"0": basis(0), "1": basis(1), "7": basis(7)},
        "metadata": {
            "0": {"expert_id": 0, "origin_task": 0, "lifecycle": "historical",
                  "rms_state": {}},
            "1": {"expert_id": 1, "origin_task": 0, "lifecycle": "historical",
                  "rms_state": {}},
            "7": {"expert_id": 7, "origin_task": 1, "lifecycle": "current",
                  "rms_state": {}},
        },
    }
    pool = V7ExpertKeyPool.from_state(legacy)
    assert pool.key_ids == ("0", "1", "7")
    assert pool.route_keys["0"].key_type == "canonical"
    assert pool.route_keys["7"].key_type == "candidate"
    assert pool.trainable_key_ids == {"7"}
    assert pool.historical_ids == (0, 1) and pool.current_ids == (7,)


def test_task_five_adds_a_reuse_key_without_overwriting_task_four():
    pool = V7ExpertKeyPool()
    pool.add(0, basis(0), origin_task=0, lifecycle="historical", trainable=False)
    pool.add_reuse_key(0, 4, initialize_current_task_key(basis(0), 0.01, 900), True)
    pool.commit((), {}, reuse_key_retention={0: True})
    task4 = pool.keys["e0_t4_reuse"].detach().clone()

    # Task 5 reuses the same expert again: a NEW key, same expert.
    new_id = pool.add_reuse_key(
        0, 5, initialize_current_task_key(basis(0), 0.01, 901), True
    )
    assert new_id == "e0_t5_reuse"
    assert pool.keys_of_expert(0, key_type="reuse") == ("e0_t4_reuse", "e0_t5_reuse")
    assert torch.equal(pool.keys["e0_t4_reuse"], task4)
    assert pool.trainable_key_ids == {"e0_t5_reuse"}
    assert pool.active_key_ids() == ("0", "e0_t4_reuse", "e0_t5_reuse")
    assert pool.route_keys["e0_t4_reuse"].origin_task == 4
    assert pool.route_keys["e0_t5_reuse"].origin_task == 5


# ----------------------------------------------------------------------
# 4. negative: a non-reusable historical expert gets no reuse key
# ----------------------------------------------------------------------
def test_add_reuse_key_refuses_unknown_and_non_historical_experts():
    pool = V7ExpertKeyPool()
    pool.add(0, basis(0), origin_task=0, lifecycle="historical", trainable=False)
    pool.add(7, basis(7), origin_task=0, lifecycle="current", trainable=True)

    with pytest.raises(ValueError, match="unknown expert"):
        pool.add_reuse_key(9, 1, basis(0), True)
    with pytest.raises(ValueError, match="frozen historical experts"):
        pool.add_reuse_key(7, 1, basis(7), True)
    with pytest.raises(ValueError, match="must be later than"):
        pool.add_reuse_key(0, 0, basis(0), True)
    assert pool.has_reuse_key(0, 1) is False


def test_prepare_candidate_pool_only_creates_keys_for_the_reusable_set(tmp_path):
    cache = tmp_path / "train.json"
    records = {}
    generator = torch.Generator().manual_seed(7)
    for index in range(8):
        query = F.normalize(torch.randn(DIM, generator=generator), dim=0)
        records["v7_t1_train_{}".format(index)] = {"query": query.tolist()}
    cache.write_text(json.dumps({"query_mode": "v7_fixed", "records": records}))

    previous = V7ExpertKeyPool()
    previous.add(0, basis(0), origin_task=0, lifecycle="historical", trainable=False)
    previous.add(1, basis(1), origin_task=0, lifecycle="historical", trainable=False)
    state_path = tmp_path / "previous.pt"
    torch.save(previous.export_state(), state_path)

    pool, center, audit = prepare_candidate_pool(
        str(cache), 8, 1, 42, 0.01, str(state_path), reusable_historical_ids=[0],
    )
    assert pool.keys_of_expert(0, key_type="reuse") == ("e0_t1_reuse",)
    assert pool.keys_of_expert(1, key_type="reuse") == ()  # NOT reusable -> no key
    assert pool.keys_of_expert(1, key_type="canonical") == ("1",)
    assert pool.trainable_key_ids == {"2", "3", "4", "5", "e0_t1_reuse"}
    assert pool.current_ids == (2, 3, 4, 5) and pool.historical_ids == (0, 1)
    assert audit["reusable_historical_ids"] == [0]
    assert audit["reuse_key_ids"] == {"0": "e0_t1_reuse"}
    assert audit["historical_lora_trainable"] is False
    assert audit["historical_canonical_keys_trainable"] is False

    # Same kernel, same task center, same perturbation as the candidates ...
    assert float(F.cosine_similarity(pool.keys["e0_t1_reuse"], center, dim=0)) > 0.99
    # ... but NOT a copy of the historical key.
    assert not torch.allclose(pool.keys["e0_t1_reuse"], pool.keys["0"])
    expected_keys, *_ = initialize_candidate_keys(
        torch.tensor([records[key]["query"] for key in sorted(records)]), 8,
        count=4, perturbation=0.01, seed=42 + 1,
    )
    assert torch.allclose(pool.keys["2"], expected_keys[0], atol=1e-6)

    with pytest.raises(ValueError, match="frozen pool"):
        prepare_candidate_pool(
            str(cache), 8, 1, 42, 0.01, str(state_path), reusable_historical_ids=[9],
        )
    with pytest.raises(ValueError, match="Task0"):
        prepare_candidate_pool(
            str(cache), 8, 0, 42, 0.01, str(state_path), reusable_historical_ids=[0],
        )


# ----------------------------------------------------------------------
# 5. task-end commit
# ----------------------------------------------------------------------
def test_commit_freezes_used_reuse_keys_and_discards_unused_ones():
    pool = V7ExpertKeyPool()
    pool.add(0, basis(0), origin_task=0, lifecycle="historical", trainable=False)
    pool.add(1, basis(1), origin_task=0, lifecycle="historical", trainable=False)
    pool.add_reuse_key(0, 1, initialize_current_task_key(basis(0), 0.01, 11), True)
    pool.add_reuse_key(1, 1, initialize_current_task_key(basis(0), 0.01, 12), True)
    pool.add(2, basis(2), origin_task=1, lifecycle="current", trainable=True)

    pool.commit((2,), {2: {}}, reuse_key_retention={0: True, 1: False})

    assert pool.route_keys["e0_t1_reuse"].lifecycle == KEY_LIFECYCLE_HISTORICAL
    assert pool.route_keys["e0_t1_reuse"].trainable is False
    assert pool.route_keys["e0_t1_reuse"].key_type == "reuse"
    assert pool.keys["e0_t1_reuse"].requires_grad is False
    # Discard, never pollute: the unused reuse key is gone entirely.
    assert "e1_t1_reuse" not in pool.keys and "e1_t1_reuse" not in pool.route_keys
    assert pool.has_reuse_key(0, 1) and not pool.has_reuse_key(1, 1)
    assert pool.route_keys["2"].key_type == "canonical"
    assert pool.metadata[2]["lifecycle"] == "historical"
    assert set(pool.historical_checksums()) == {"0", "1", "2", "e0_t1_reuse"}
    assert pool.trainable_key_ids == set()


def test_historical_checksums_detect_a_changed_reuse_key():
    pool = V7ExpertKeyPool()
    pool.add(0, basis(0), origin_task=0, lifecycle="historical", trainable=False)
    pool.add_reuse_key(0, 1, initialize_current_task_key(basis(0), 0.01, 13), True)
    pool.commit((), {}, reuse_key_retention={0: True})
    before = pool.historical_checksums()
    with torch.no_grad():
        pool.keys["e0_t1_reuse"].add_(basis(500))
    assert pool.historical_checksums() != before
