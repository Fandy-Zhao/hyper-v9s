import ast
import json

import pytest
import torch

from compose.v7.pool import V7ExpertKeyPool
from compose.v7.routing import GlobalTop2Router
from compose.v7.hf_trainer import resolve_v8_selectable_experts
from compose.v8.screening import (
    aggregate_reusable_experts,
    load_reusable_screening,
    sample_teacher_records,
    teacher_router_recall,
    initialize_reuse_keys,
)
from compose.v7.training import selected_current_key_loss


def _record(index, answer):
    return {
        "id": "sample-{:03d}".format(index),
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": answer},
        ],
    }


def test_teacher_sampling_is_seeded_stratified_and_not_prefix():
    records = [_record(index, "yes" if index < 80 else "no") for index in range(100)]
    first, audit = sample_teacher_records(records, num_samples=20, seed=42)
    second, second_audit = sample_teacher_records(records, num_samples=20, seed=42)
    ids = [row["id"] for row in first]
    assert ids == [row["id"] for row in second]
    assert audit == second_audit
    assert set(ids) != {row["id"] for row in records[:20]}
    answers = [row["conversations"][-1]["value"] for row in first]
    assert answers.count("yes") == 16
    assert answers.count("no") == 4


def test_teacher_sampler_rejects_conflicting_controls():
    with pytest.raises(ValueError):
        sample_teacher_records([_record(0, "x")], num_samples=1, sample_ratio=1.0)


def test_feature_kcenter_sampling_is_deterministic_and_diverse():
    records = [_record(index, "a" if index % 2 else "b") for index in range(8)]
    queries = torch.zeros(8, 1536)
    for cluster in range(4):
        queries[cluster * 2:(cluster + 1) * 2, cluster] = 1.0
    kwargs = dict(num_samples=4, seed=9, strategy="answer_stratified_feature_kcenter",
                  queries=queries, query_sample_ids=["sample-{:03d}".format(index) for index in range(8)])
    first, audit = sample_teacher_records(records, **kwargs)
    second, _ = sample_teacher_records(records, **kwargs)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4
    assert audit["duplicate_cosine_threshold"] == 0.99


def test_feature_kcenter_large_cpu_shape_has_no_quadratic_state():
    count, dim, budget = 2048, 64, 64
    records = [_record(index, str(index % 4)) for index in range(count)]
    queries = torch.nn.functional.normalize(torch.randn(count, dim), dim=-1)
    selected, audit = sample_teacher_records(
        records, num_samples=budget, seed=3, strategy="answer_stratified_feature_kcenter",
        queries=queries, query_sample_ids=["sample-{:03d}".format(index) for index in range(count)],
    )
    assert len(selected) == budget
    assert audit["feature_coverage_radius"] >= 0.0


def test_reuse_key_is_strict_selected_query_centroid_without_perturbation():
    queries = torch.zeros(2, 1536); queries[0, 0] = 1; queries[1, 1] = 1
    teacher = {"task_id": 1, "records": [
        {"sample_id": "a", "selected_experts": [7]},
        {"sample_id": "b", "selected_experts": [7]},
    ]}
    keys, audit = initialize_reuse_keys(
        teacher, [7], {"sample_ids": ["a", "b"], "queries": queries},
        center=None, perturbation=0.0, min_support=1, seed=1, task_index=1,
    )
    expected = torch.nn.functional.normalize(queries.mean(dim=0), dim=0)
    assert torch.allclose(keys[7], expected)
    assert audit["experts"]["7"]["perturbation"] == 0.0
    with pytest.raises(ValueError):
        initialize_reuse_keys(teacher, [8], {"sample_ids": ["a", "b"], "queries": queries},
                              center=None, perturbation=0.0, min_support=1, seed=1, task_index=1)


def test_reuse_quality_scales_only_reuse_key_gradient():
    pool = V7ExpertKeyPool(query_dim=1536)
    query = torch.zeros(1536); query[0] = 1
    pool.add(0, query, origin_task=0, lifecycle="historical", trainable=False)
    offset = torch.zeros(1536); offset[1] = 1
    pool.add_reuse_key(0, 1, torch.nn.functional.normalize(query + offset, dim=0), True)
    pool.add(1, torch.nn.functional.normalize(query + torch.eye(2, 1536)[1], dim=0), origin_task=1, lifecycle="current", trainable=True)
    ids = torch.tensor([[0, 1]])
    high, _ = selected_current_key_loss(query.unsqueeze(0), ids, pool, reuse_quality_weights=torch.tensor([0.9]))
    high.backward(); high_grad = pool.keys[pool.reuse_key_id(0, 1)].grad.norm().item()
    pool.zero_grad()
    low, _ = selected_current_key_loss(query.unsqueeze(0), ids, pool, reuse_quality_weights=torch.tensor([0.1]))
    low.backward(); low_grad = pool.keys[pool.reuse_key_id(0, 1)].grad.norm().item()
    assert high_grad > low_grad


def test_screening_aggregates_pairs_and_requires_stable_support(tmp_path):
    records = []
    for index in range(10):
        selected = [1, 3] if index < 3 else ([1] if index < 5 else [])
        records.append({
            "sample_id": str(index), "selected_experts": selected,
            "delta_nll": 0.2 if selected else None,
            "teacher_gain": 1.0 if selected else 0.0,
        })
    teacher = {"task_id": 4, "teacher_search_mode": "full_history_single_oracle",
               "historical_experts_visible": [1, 2, 3], "records": records}
    result = aggregate_reusable_experts(
        teacher, min_teacher_support=3, min_teacher_usage_rate=0.2
    )
    assert result["reusable_historical_expert_ids"] == [1, 3]
    assert result["expert_statistics"]["1"]["single_usage_count"] == 2
    assert result["expert_statistics"]["1"]["pair_usage_count"] == 3
    assert result["expert_statistics"]["2"]["reusable"] is False
    assert result["full_training_oracle_eval_sample_count"] == 0

    path = tmp_path / "screening.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    loaded = load_reusable_screening(path, expected_task=4, historical_ids=[1, 2, 3])
    assert loaded["reusable_historical_expert_ids"] == [1, 3]
    with pytest.raises(ValueError):
        load_reusable_screening(path, expected_task=3, historical_ids=[1, 2, 3])


def test_selectable_pool_is_reusable_old_plus_all_current():
    pool = V7ExpertKeyPool(query_dim=1536)
    def key(position):
        value = torch.zeros(1536)
        value[position] = 1.0
        return value
    pool.add(0, key(0), origin_task=0, lifecycle="historical", trainable=False)
    pool.add(1, key(1), origin_task=1, lifecycle="historical", trainable=False)
    pool.add(2, key(2), origin_task=4, lifecycle="current", trainable=True)
    pool.add(3, key(3), origin_task=4, lifecycle="current", trainable=True)
    query = (key(0) * 2 + key(1) * 3 + key(2)).unsqueeze(0)
    routed = GlobalTop2Router(pool)(query, excluded=[0])
    assert set(routed.expert_ids[0].tolist()) == {1, 2}
    assert 0 not in routed.expert_ids[0].tolist()


def test_all_three_full_training_route_types_are_legal():
    pool = V7ExpertKeyPool(query_dim=1536)
    keys = []
    for position in range(4):
        value = torch.zeros(1536)
        value[position] = 1.0
        keys.append(value)
    pool.add(0, keys[0], origin_task=0, lifecycle="historical", trainable=False)
    pool.add(1, keys[1], origin_task=1, lifecycle="historical", trainable=False)
    pool.add(2, keys[2], origin_task=4, lifecycle="current", trainable=True)
    pool.add(3, keys[3], origin_task=4, lifecycle="current", trainable=True)
    queries = torch.stack([
        keys[0] + keys[1],
        keys[0] + keys[2],
        keys[2] + keys[3],
    ])
    result = GlobalTop2Router(pool)(queries)
    assert result.route_types == ("OldOld", "OldNew", "NewNew")


def test_task0_skips_history_and_routes_over_candidates_only():
    reusable, excluded, selectable = resolve_v8_selectable_experts(
        [], [0, 1, 2, 3], [], 0
    )
    assert reusable == ()
    assert excluded == ()
    assert selectable == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        resolve_v8_selectable_experts([9], [0, 1, 2, 3], [9], 0)


def test_teacher_router_recall_is_measured_inside_reusable_pool():
    keys = {str(i): torch.eye(3, 1536)[i] for i in range(3)}
    teacher = {
        "records": [
            {"sample_id": "a", "selected_experts": [1]},
            {"sample_id": "b", "selected_experts": [2]},
        ]
    }
    screening = {"reusable_historical_expert_ids": [0, 1, 2]}
    queries = torch.stack([keys["1"], keys["0"] * 0.8 + keys["2"] * 0.7])
    metric = teacher_router_recall(
        teacher, screening,
        {"sample_ids": ["a", "b"], "queries": queries},
        {"keys": keys},
    )
    assert metric["TeacherRouterRecall@1"] == 0.5
    assert metric["TeacherRouterRecall@2"] == 1.0


def test_aggregation_output_feeds_the_reuse_key_provenance_check():
    """``expert_statistics`` is the provenance source the reuse keys validate against.

    ``initialize_reuse_keys`` fails closed unless each reusable expert's support
    ids, count and hash match the screening evidence it is handed, so the mapping
    passed as ``expected_support_by_expert`` must be the aggregation's own
    ``expert_statistics``.  Naming a different key removes the check by raising
    ``KeyError`` inside the teacher stage, after the shards have already run.
    """
    records = [
        {"sample_id": str(index), "selected_experts": [1] if index < 5 else []}
        for index in range(10)
    ]
    teacher = {"task_id": 4, "teacher_search_mode": "full_history_single_oracle",
               "historical_experts_visible": [1, 2], "records": records}
    result = aggregate_reusable_experts(
        teacher, min_teacher_support=3, min_teacher_usage_rate=0.2
    )
    assert result["reusable_historical_expert_ids"] == [1]
    assert "expert_statistics" in result and "experts" not in result

    payload = {
        "sample_ids": [str(index) for index in range(10)],
        "queries": torch.eye(10, 1536),
    }
    keys, audit = initialize_reuse_keys(
        teacher, result["reusable_historical_expert_ids"], payload,
        center=None, perturbation=0.0, min_support=3, seed=46, task_index=4,
        expected_support_by_expert=result["expert_statistics"],
    )
    assert sorted(keys) == [1]
    assert audit["experts"]["1"]["support"] == 5

    tampered = dict(result["expert_statistics"])
    tampered["1"] = dict(tampered["1"], reuse_support_count=4)
    with pytest.raises(ValueError):
        initialize_reuse_keys(
            teacher, [1], payload, center=None, perturbation=0.0, min_support=3,
            seed=46, task_index=4, expected_support_by_expert=tampered,
        )


def test_full_data_route_boundary_has_no_answer_or_oracle_inputs():
    source = open("compose/v7/hf_trainer.py", encoding="utf-8").read()
    tree = ast.parse(source)
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "route_full_data_queries"
    )
    names = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
    assert not ({"answer", "ground_truth", "nll", "teacher"} & names)
    assert len(method.args.args) == 2  # self, queries
