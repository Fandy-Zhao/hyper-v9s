import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection, pad_selection
from compose.v7.checkpoint import load_v7_checkpoint, save_v7_checkpoint
from compose.v7.commit import commit_retained_candidates
from compose.v7.config import V7Config, V7PruningConfig
from compose.v7.inference import V7InferenceRouter
from compose.v7.hf_trainer import padded_compose_selections
from compose.v7.pool import (
    V7ExpertKeyPool,
    initialize_candidate_keys,
    tensor_checksum,
)
from compose.v7.pruning import CandidatePruner
from compose.v7.query import FixedMultimodalQuery, full_train_task_center
from compose.v7.routing import GlobalTop2Router, route_signature_groups
from compose.v7.training import adapter_checksums, selected_current_key_loss
from compose.v7.training import (
    full_data_coverage_audit,
    per_sample_teacher_forcing_token_nll,
    supervised_token_mask,
    teacher_forcing_token_nll,
)
from compose.v7.provenance import (
    audit_split_isolation,
    bind_pipeline_data_usage,
    build_runtime_contract,
    validate_runtime_contract,
)
from compose.v7.workflow import validate_query_cache_contract
from compose.eval.query_features import query_backbone_provenance
from compose.lora.rms import (
    apply_kappa_calibration,
    merge_commit_frozen_calibration,
    runtime_kappa_calibration,
)


def basis(index):
    value = torch.zeros(1536)
    value[index] = 1.0
    return value


def pool_with(hist=(), current=()):
    pool = V7ExpertKeyPool()
    for expert_id, axis in hist:
        pool.add(expert_id, basis(axis), axis, "historical", False)
    for expert_id, axis in current:
        pool.add(expert_id, basis(axis), 9, "current", True)
    return pool


def selection(rows):
    padded = [pad_selection(tuple(row), (1.0,) * len(row)) for row in rows]
    return ComposeSelection(
        torch.tensor([value[0] for value in padded]),
        torch.tensor([value[1] for value in padded]),
    )


def make_linear(expert_ids):
    base = nn.Linear(3, 2, bias=False)
    nn.init.zeros_(base.weight)
    layer = ComposeLinear(base, rank=1, alpha=1.0)
    for expert_id in expert_ids:
        expert = layer.add_expert(expert_id)
        nn.init.constant_(expert.lora_A.weight, 0.2 + expert_id * 0.01)
        nn.init.constant_(expert.lora_B.weight, 0.3 + expert_id * 0.01)
    return layer


def write_commit_source(root, pool):
    root.mkdir()
    state = {
        "model.layers.0.self_attn.q_proj.experts.{}.{}.weight".format(i, part): torch.ones(1, 1)
        for i in pool.expert_ids for part in ("lora_A", "lora_B")
    }
    torch.save(state, root / "compose_experts.bin")
    manifest = {
        "format_version": 1,
        "adapter": {"rank": 8, "alpha": 16.0, "dropout": 0.0, "layers": ["model.layers.0.self_attn.q_proj"]},
        "experts": [
            {"expert_id": i, "adapter_name": "e{}".format(i), "rank": 8, "alpha": 16.0, "status": "frozen"}
            for i in pool.expert_ids
        ],
    }
    (root / "compose_experts.json").write_text(json.dumps(manifest))


def test_01_fixed_query_is_1536_normalized_parameter_free_and_stage_stable():
    module = FixedMultimodalQuery()
    visual, text = torch.randn(3, 768, requires_grad=True), torch.randn(3, 768)
    first = module(visual, text)
    second = module(visual, text)
    assert first.shape == (3, 1536)
    assert torch.allclose(first.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert sum(value.numel() for value in module.parameters()) == 0
    assert not first.requires_grad
    assert torch.equal(first, second)


def test_02_task_center_requires_the_complete_train_split_without_2000_cutoff():
    queries = torch.nn.functional.normalize(torch.randn(2001, 1536), dim=-1)
    _, audit = full_train_task_center(queries, num_train_samples=2001)
    assert audit == {"num_train_samples": 2001, "num_queries_used_for_center": 2001}
    with pytest.raises(ValueError, match="full-data center mismatch"):
        full_train_task_center(queries[:2000], num_train_samples=2001)


def test_03_four_candidate_keys_share_center_but_are_distinct_and_reproducible():
    queries = torch.nn.functional.normalize(torch.randn(17, 1536), dim=-1)
    keys_a, center, audit = initialize_candidate_keys(queries, 17, seed=7)
    keys_b, _, _ = initialize_candidate_keys(queries, 17, seed=7)
    assert keys_a.shape == (4, 1536)
    assert torch.equal(keys_a, keys_b)
    assert torch.all((keys_a @ center) > 0.99)
    off = torch.tensor(audit["pairwise_cosine"])
    assert torch.all(off[~torch.eye(4, dtype=torch.bool)] < 1.0 - 1e-8)


def test_04_global_top2_can_mix_historical_and_current():
    pool = pool_with(hist=((1, 0),), current=((7, 1), (8, 2), (9, 3)))
    query = torch.nn.functional.normalize((basis(0) + basis(1)).unsqueeze(0), dim=-1)
    result = GlobalTop2Router(pool)(query)
    assert set(result.expert_ids[0].tolist()) == {1, 7}
    assert result.route_types == ("OldNew",)


def test_05_task0_routes_directly_over_four_current_candidates():
    pool = pool_with(current=((0, 0), (1, 1), (2, 2), (3, 3)))
    result = GlobalTop2Router(pool)(basis(2).unsqueeze(0))
    assert result.expert_ids.shape == (1, 2)
    assert result.route_types == ("NewNew",)


def _backward_linear(layer, rows, inputs=None):
    inputs = torch.randn(len(rows), 3) if inputs is None else inputs
    with use_selection(selection(rows)):
        loss = layer(inputs).sum()
    if loss.requires_grad:
        loss.backward()
    return loss


def test_06_old_old_forward_is_valid_and_updates_no_expert():
    layer = make_linear((1, 2, 7))
    layer.experts["1"].requires_grad_(False)
    layer.experts["2"].requires_grad_(False)
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    _backward_linear(layer, [(1, 2)])
    assert all(value.grad is None for value in layer.experts["7"].parameters())
    assert all(torch.equal(before[name], value) for name, value in layer.state_dict().items())


def test_07_old_new_updates_only_new():
    layer = make_linear((1, 7, 8))
    layer.experts["1"].requires_grad_(False)
    _backward_linear(layer, [(1, 7)])
    assert all(value.grad is None for value in layer.experts["1"].parameters())
    assert any(value.grad is not None for value in layer.experts["7"].parameters())
    assert all(value.grad is None for value in layer.experts["8"].parameters())


def test_08_new_new_updates_exactly_both_selected_new_experts():
    layer = make_linear((7, 8, 9))
    _backward_linear(layer, [(7, 8)])
    assert all(any(value.grad is not None for value in layer.experts[str(i)].parameters()) for i in (7, 8))
    assert all(value.grad is None for value in layer.experts["9"].parameters())


def test_09_per_sample_routes_do_not_leak_gradient_across_rows():
    layer = make_linear((1, 2, 7, 8))
    groups = route_signature_groups(torch.tensor([[1, 7], [2, 8], [1, 2]]))
    assert {key: value.tolist() for key, value in groups.items()} == {
        (1, 7): [0], (2, 8): [1], (1, 2): [2]
    }
    first = torch.tensor([[1.0, 0.0, 0.0]])
    _backward_linear(layer, [(1, 7)], first)
    grad7 = layer.experts["7"].lora_B.weight.grad.clone()
    layer.zero_grad(set_to_none=True)
    _backward_linear(layer, [(1, 7), (2, 8)], torch.cat([first, torch.zeros_like(first)]))
    assert torch.allclose(layer.experts["7"].lora_B.weight.grad, grad7)


def test_10_historical_lora_checksum_unchanged_after_current_update():
    layer = make_linear((1, 7))
    layer.experts["1"].requires_grad_(False)
    before = tensor_checksum(layer.experts["1"].lora_A.weight)
    optimizer = torch.optim.SGD(layer.experts["7"].parameters(), lr=0.1)
    _backward_linear(layer, [(1, 7)])
    optimizer.step()
    assert tensor_checksum(layer.experts["1"].lora_A.weight) == before


def test_11_historical_key_checksum_unchanged_after_key_update():
    pool = pool_with(hist=((1, 0),), current=((7, 1), (8, 2), (9, 3)))
    before = pool.historical_checksums()
    result = GlobalTop2Router(pool)(basis(1).unsqueeze(0))
    loss, _ = selected_current_key_loss(basis(4).unsqueeze(0), result.expert_ids, pool)
    loss.backward()
    torch.optim.SGD([pool.keys["7"]], lr=0.1).step()
    assert pool.historical_checksums() == before


def test_12_key_loss_targets_only_selected_current_keys():
    pool = pool_with(hist=((1, 0),), current=((7, 1), (8, 2), (9, 3)))
    selected = torch.tensor([[1, 7], [7, 8]])
    loss, per_sample = selected_current_key_loss(torch.stack([basis(4), basis(5)]), selected, pool)
    loss.backward()
    assert per_sample.shape == (2,)
    assert pool.keys["1"].grad is None
    assert pool.keys["7"].grad is not None
    assert pool.keys["8"].grad is not None
    assert pool.keys["9"].grad is None


def test_13_remove_and_reroute_reexecutes_global_top2():
    pool = pool_with(hist=((1, 0), (2, 1)), current=((7, 2), (8, 3), (9, 4), (10, 5)))
    queries = torch.stack([basis(2), basis(3), basis(4), basis(5)])
    seen = []

    def scorer(routes):
        seen.append(routes.clone())
        return {"metric": float(routes.eq(7).any(dim=1).float().mean()), "loss": float(routes.ne(7).all(dim=1).float().mean())}

    pruner = CandidatePruner(pool, V7PruningConfig(candidate_prune_enabled=False))
    pruner.evaluate(queries, queries, basis(2), scorer)
    assert len(seen) == 5
    full, minus_7 = seen[0], seen[1]
    assert torch.any(full.eq(7))
    assert not torch.any(minus_7.eq(7))
    assert torch.all(minus_7.ge(0))


def test_14_pruned_candidate_is_absent_from_committed_selectable_pool(tmp_path):
    pool = pool_with(hist=((1, 0), (2, 1)), current=((7, 2), (8, 3), (9, 4), (10, 5)))
    source = tmp_path / "source"
    write_commit_source(source, pool)
    metrics = {i: {"keep": i in (7, 8)} for i in pool.current_ids}
    target = tmp_path / "committed"
    commit_retained_candidates(str(source), str(target), pool, (7, 8), metrics)
    restored = V7ExpertKeyPool.from_state(torch.load(target / "v7_keys.pt", weights_only=False))
    assert 9 not in restored.selectable_ids()
    assert 10 not in restored.selectable_ids()


def test_15_retained_key_and_lora_reload_exactly(tmp_path):
    # The filter is byte-exact for selected tensors and normalized-exact for keys.
    test_14_pruned_candidate_is_absent_from_committed_selectable_pool(tmp_path)
    state = torch.load(tmp_path / "committed" / "compose_experts.bin", weights_only=False)
    assert state
    assert all(".experts.9." not in key and ".experts.10." not in key for key in state)
    pool = V7ExpertKeyPool.from_state(torch.load(tmp_path / "committed" / "v7_keys.pt", weights_only=False))
    assert torch.equal(pool.keys["7"], basis(2))


def test_16_inference_has_no_task_id_and_records_cross_task_pairs():
    pool = pool_with(hist=((1, 0), (2, 1)))
    router = V7InferenceRouter(pool)
    assert "task" not in inspect.signature(router.forward).parameters
    result = router(torch.randn(1, 768), torch.randn(1, 768))
    assert result.expert_ids.shape == (1, 2)
    assert router.cross_task_pairs(result, pool)


def test_17_checkpoint_resume_refreezes_historical_experts(tmp_path):
    pool = pool_with(hist=((1, 0), (2, 1)), current=((7, 2), (8, 3), (9, 4), (10, 5)))
    optimizer = torch.optim.AdamW([pool.keys["7"]], lr=1e-3)
    path = tmp_path / "v7.pt"
    save_v7_checkpoint(
        str(path), task_index=1, training_step=3, key_pool=pool,
        candidate_lora_state={"x": torch.ones(1)}, optimizer=optimizer,
        scheduler=None, usage_counters={"7": 4}, config=V7Config(), rms_state={"a": 1},
    )
    payload, restored, config = load_v7_checkpoint(str(path), restore_rng=False)
    assert payload["training_step"] == 3 and config.method == "v7_global_coevolution"
    assert all(not restored.keys[str(value)].requires_grad for value in restored.historical_ids)
    assert all(restored.keys[str(value)].requires_grad for value in restored.current_ids)


def test_18_v7_top2_rows_are_padded_to_unified_four_slot_contract():
    pool = pool_with(hist=((1, 0),), current=((7, 1), (8, 2), (9, 3)))
    queries = torch.stack([
        torch.nn.functional.normalize(basis(0) + basis(1), dim=0),
        torch.nn.functional.normalize(basis(2) + basis(3), dim=0),
    ])
    routed = GlobalTop2Router(pool)(queries)

    rows = padded_compose_selections(routed.selection)
    assert len(rows) == 2
    assert all(len(expert_ids) == 4 and len(gates) == 4 for expert_ids, gates in rows)
    assert all(expert_ids[2:] == (-1, -1) for expert_ids, _ in rows)
    assert all(gates[2:] == (0.0, 0.0) for _, gates in rows)

    restored = ComposeSelection(
        torch.tensor([expert_ids for expert_ids, _ in rows], dtype=torch.long),
        torch.tensor([gates for _, gates in rows], dtype=torch.float32),
    )
    assert [record["expert_ids"] for record in restored.per_sample_sets()] == routed.expert_ids.tolist()


def test_19_historical_adapter_checksum_supports_bfloat16_exactly():
    layer = make_linear((2,)).to(dtype=torch.bfloat16)
    manager = type("Manager", (), {"layers": {"decoder.q_proj": layer}})()

    before = adapter_checksums(manager, (2,))
    assert before == adapter_checksums(manager, (2,))
    with torch.no_grad():
        layer.experts["2"].lora_A.weight.view(-1)[0] += torch.tensor(
            1.0, dtype=torch.bfloat16
        )
    assert adapter_checksums(manager, (2,)) != before


def test_20_formal_recipe_has_no_implicit_30_step_cap_and_smoke_is_explicit():
    import yaml
    import compose.experiments.v7_task_run as runner

    config = yaml.safe_load(
        (Path(__file__).parents[2] / "configs" / "v7_global_coevolution.yaml").read_text()
    )
    assert config["training"]["num_train_epochs"] == 1
    assert (
        config["training"]["per_device_train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
    ) == 64
    source = inspect.getsource(runner)
    assert '"--smoke-max-steps"' in source
    assert '"--smoke-gradient-accumulation-steps"' in source
    assert "if formal_run else args.smoke_gradient_accumulation_steps" in source
    assert '"--max-steps", type=int, default=30' not in source


def test_20b_official_classification_metric_uses_rewritten_validation_ids():
    import compose.experiments.v7_task_run as runner

    source = inspect.getsource(runner)
    assert "Path(annotation_file).resolve() == Path(args.val_file).resolve()" in source
    assert 'annotation_file = str(val_json)' in source


def test_21_formal_full_data_coverage_fails_but_explicit_smoke_can_be_partial():
    complete = full_data_coverage_audit(2001, map(str, range(2001)), 2001, 32, 2001, True)
    assert complete["unique_train_sample_ids_seen"] == 2001
    assert complete["train_sample_coverage"] == 1.0
    with pytest.raises(RuntimeError, match="did not cover"):
        full_data_coverage_audit(2001, map(str, range(2000)), 30, 30, 30, True)
    smoke = full_data_coverage_audit(2001, map(str, range(30)), 30, 30, 30, False)
    assert smoke["train_sample_coverage"] < 1.0


def test_22_answer_nll_uses_only_shifted_supervised_positions():
    labels = torch.tensor([[-100, -100, 2, 3]])
    logits = torch.zeros(1, 4, 5)
    baseline = teacher_forcing_token_nll(logits, labels)
    prompt_changed = logits.clone()
    prompt_changed[:, 0, :] = torch.tensor([100.0, -100.0, -100.0, -100.0, -100.0])
    assert torch.equal(supervised_token_mask(labels), labels[:, 1:].ne(-100))
    assert torch.allclose(teacher_forcing_token_nll(prompt_changed, labels), baseline)
    answer_changed = logits.clone()
    answer_changed[:, 1, 2] = 20.0
    assert teacher_forcing_token_nll(answer_changed, labels) < baseline
    with pytest.raises(ValueError, match="supervised token"):
        teacher_forcing_token_nll(logits, torch.full_like(labels, -100))


def test_22b_packed_answer_nll_preserves_sum_of_batch_one_losses():
    torch.manual_seed(7)
    logits = torch.randn(3, 6, 11)
    labels = torch.tensor([
        [-100, -100, 2, 3, -100, -100],
        [-100, 4, 5, 6, 7, -100],
        [-100, -100, -100, 8, 9, 10],
    ])
    packed = per_sample_teacher_forcing_token_nll(logits, labels)
    separate = torch.stack([
        teacher_forcing_token_nll(logits[index:index + 1], labels[index:index + 1])
        for index in range(3)
    ])
    assert torch.equal(packed, separate)
    assert torch.equal(packed.sum(), separate.sum())
    with pytest.raises(ValueError, match="every sample"):
        per_sample_teacher_forcing_token_nll(
            logits, torch.full_like(labels, -100)
        )


def test_23_historical_rms_is_one_persisted_runtime_contract():
    model = nn.Module()
    model.layer = make_linear((1, 7))
    historical = {"layer": {"1": 1.75}}
    apply_kappa_calibration(model, historical)
    before = runtime_kappa_calibration(model, (1,))
    assert before == historical
    merged = merge_commit_frozen_calibration(
        historical, {"layer": {"1": 0.5, "7": 1.25}}, (7,)
    )
    apply_kappa_calibration(model, merged)
    assert runtime_kappa_calibration(model, (1,)) == before
    assert runtime_kappa_calibration(model, (7,)) == {"layer": {"7": 1.25}}


def test_24_iterative_pruning_recomputes_substitute_contribution():
    pool = pool_with(
        hist=((1, 4), (2, 5)), current=((7, 0), (8, 1), (9, 2), (10, 3))
    )
    queries = torch.nn.functional.normalize((basis(0) + basis(1)).unsqueeze(0), dim=-1)

    def scorer(routes):
        useful = routes.eq(7).any(dim=1) | routes.eq(8).any(dim=1)
        metric = float(useful.float().mean())
        return {"metric": metric, "loss": 1.0 - metric}

    retained, metrics, audit = CandidatePruner(pool, V7PruningConfig()).evaluate(
        queries, queries, basis(0), scorer
    )
    assert len(set(retained) & {7, 8}) == 1
    assert not ({7, 8} <= set(metrics) - set(retained))
    assert sum(row["decision"] == "remove" for row in audit["pruning_trajectory"]) == 3


def test_25_global_top2_minimum_pool_and_task1_zero_current_retention():
    task0 = pool_with(current=((0, 0), (1, 1), (2, 2), (3, 3)))
    queries = torch.stack([basis(0), basis(1), basis(2), basis(3)])
    constant = lambda routes: {"metric": 1.0, "loss": 1.0}
    retained0, metrics0, _ = CandidatePruner(task0, V7PruningConfig()).evaluate(
        queries, queries, basis(0), constant
    )
    assert len(retained0) == 2
    assert all(
        "retained_for_global_top2_minimum_pool" in metrics0[value]["reason"]
        for value in retained0
    )

    task1 = pool_with(
        hist=((0, 4), (1, 5)), current=((4, 0), (5, 1), (6, 2), (7, 3))
    )
    retained1, _, audit1 = CandidatePruner(task1, V7PruningConfig()).evaluate(
        queries, queries, basis(0), constant
    )
    assert retained1 == ()
    assert audit1["final_selectable_pool"] == [0, 1]


def test_26_split_leakage_and_runtime_preprocessing_parity(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text(json.dumps(rows), encoding="utf-8")
        return str(path)

    train = write("train.json", [{"id": "a", "image": "a.jpg", "text": "qa", "answer": "a"}])
    val = write("val.json", [{"id": "b", "image": "b.jpg", "text": "qb", "answer": "b"}])
    test = write("test.json", [{"id": "c", "image": "c.jpg", "text": "qc", "answer": "c"}])
    audit = bind_pipeline_data_usage(
        audit_split_isolation(train, val, test),
        training_sources=(train,), key_learning_sources=(train,),
        rms_sources=(val,), pruning_sources=(val,),
    )
    assert audit["test_data_used_for_pruning"] is False
    with pytest.raises(ValueError, match="same normalized path"):
        audit_split_isolation(train, val, val)
    duplicate = write("duplicate.json", json.loads(Path(val).read_text()))
    with pytest.raises(ValueError, match="identical file hashes"):
        audit_split_isolation(train, val, duplicate)
    overlap = write("overlap.json", [{"id": "z", "image": "a.jpg", "text": "qa", "answer": "x"}])
    with pytest.raises(ValueError, match="record overlap"):
        audit_split_isolation(train, val, overlap)

    # UCIT files can reuse local numeric IDs across independent source rows.
    # This must be reported without rejecting otherwise disjoint splits.
    id_collision = write(
        "id_collision.json",
        [{"id": "a", "image": "other.jpg", "text": "other", "answer": "x"}],
    )
    collision_audit = audit_split_isolation(train, val, id_collision)
    collision = collision_audit["overlap_checks"]["train_vs_test"]
    assert collision["source_id_overlap"] == 1
    assert collision["image_question_overlap"] == 0
    assert collision["normalized_record_overlap"] == 0

    projector = tmp_path / "projector.bin"
    projector.write_bytes(b"projector")
    contract = build_runtime_contract(
        image_aspect_ratio="pad", vision_tower=str(tmp_path / "clip"),
        mm_vision_select_layer=-2, mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu", projector_path=str(projector),
    )
    validate_runtime_contract(contract, dict(contract), "test")
    mismatched = dict(contract, image_aspect_ratio="square")
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        validate_runtime_contract(contract, mismatched, "pruning")


def test_27_gradient_accumulation_audit_tracks_new_microbatch_contributions_only():
    from compose.v7.hf_trainer import V7ComposeTrainer

    trainer = object.__new__(V7ComposeTrainer)
    trainer._v7_key_gradient_ids = set()
    trainer._v7_lora_gradient_ids = set()
    trainer._v7_key_gradient_sq = 0.0
    trainer._v7_lora_gradient_sq = 0.0
    r1, r2 = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))
    r1.register_hook(lambda grad: trainer._record_v7_gradient("key", 1, grad))
    r2.register_hook(lambda grad: trainer._record_v7_gradient("key", 2, grad))
    (r1 * 2).backward()
    assert trainer._v7_key_gradient_ids == {1}
    trainer._v7_key_gradient_ids.clear()
    trainer._v7_key_gradient_sq = 0.0
    (r2 * 3).backward()
    assert r1.grad is not None and r2.grad is not None
    assert trainer._v7_key_gradient_ids == {2}


def test_28_nll_eval_output_writer_persists_json_and_honors_sharding(tmp_path):
    """Regression: nll_eval used os.makedirs without importing os, crashing
    after computing per-sample NLLs. The output persistence is now a unit-
    testable helper covering plain and sharded targets."""
    from compose.eval.nll_eval import write_nll_output

    results = {"v7_t0_val_0": {"global_top2": {"mean_answer_nll": 0.5}}}
    plain = str(tmp_path / "nested" / "nll.json")
    assert write_nll_output(plain, results, num_shards=1, shard_index=0) == plain
    assert json.loads(Path(plain).read_text()) == results
    sharded = str(tmp_path / "sharded.json")
    target = write_nll_output(sharded, results, num_shards=2, shard_index=1)
    assert target == sharded + ".rank1"
    assert json.loads(Path(target).read_text()) == results
    assert not Path(sharded).exists()


def test_29_query_backbone_and_pair_scale_are_formal_fixed_contracts(tmp_path):
    model = tmp_path / "clip"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"clip"}')
    (model / "preprocessor_config.json").write_text('{"size":336}')
    first = query_backbone_provenance(model)
    second = query_backbone_provenance(model)
    assert first == second and len(first["backbone_hash"]) == 64
    cache = tmp_path / "features.json"
    cache.write_text(json.dumps({
        "query_mode": "v7_fixed", "feature_source": "frozen_clip_l14_336",
        "query_backbone_provenance": first,
    }))
    assert validate_query_cache_contract(
        (cache,), "clip-vit-large-patch14-336", str(model)
    ) == first
    with pytest.raises(ValueError, match="pair_scale"):
        V7Config.from_dict({"routing": {"pair_scale": 0.8}})


def test_29b_cached_query_text_is_placeholder_free_question_text():
    """Regression: the query-cache text extraction fed CLIP the raw LLaVA
    human value, which embeds the literal <image> placeholder that live
    test-time routing (records.question_text) strips. The embedding shift is
    material (measured cosine ~0.94), so train/val cached queries must now be
    byte-identical to the live test text."""
    from compose.data.records import question_text
    from compose.eval.query_features import _sample_text

    conversation = {
        "id": "x",
        "image": "dir/x.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "A dog"},
        ],
    }
    extracted = _sample_text(conversation)
    assert extracted == question_text(conversation) == "What is shown?"
    assert "<image>" not in extracted and not extracted.startswith("\n")
    flat = {"question_id": "7", "image": "dir/y.jpg", "text": "Describe it.",
            "answer": "snow"}
    assert _sample_text(flat) == question_text(flat) == "Describe it."


def test_30_atomic_commit_failure_never_exposes_final_directory(tmp_path, monkeypatch):
    import compose.v7.commit as commit_module

    pool = pool_with(current=((7, 0), (8, 1), (9, 2), (10, 3)))
    source = tmp_path / "source"
    target = tmp_path / "committed"
    write_commit_source(source, pool)
    real_save = commit_module.torch.save
    calls = {"count": 0}

    def fail_second_save(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated commit interruption")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(commit_module.torch, "save", fail_second_save)
    with pytest.raises(OSError, match="simulated"):
        commit_retained_candidates(
            str(source), str(target), pool, (7, 8), {7: {}, 8: {}, 9: {}, 10: {}}
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".committed-*"))
    assert set(pool.current_ids) == {7, 8, 9, 10}


def test_31_checkpoint_continuous_and_resume_states_are_equivalent(tmp_path):
    pool = pool_with(current=((7, 0), (8, 1), (9, 2), (10, 3)))
    optimizer = torch.optim.AdamW([pool.keys["7"]], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def step(key, opt, sched, axis):
        opt.zero_grad(set_to_none=True)
        loss = (key[axis] - 0.25).square()
        loss.backward()
        opt.step()
        sched.step()

    step(pool.keys["7"], optimizer, scheduler, 4)
    path = tmp_path / "resume.pt"
    lora_state = {"layer.experts.7.lora_A.weight": torch.tensor([1.25])}
    counters = {"candidate_usage": {"7": 3}, "micro_steps": 1}
    save_v7_checkpoint(
        str(path), task_index=0, training_step=1, key_pool=pool,
        candidate_lora_state=lora_state, optimizer=optimizer, scheduler=scheduler,
        usage_counters=counters, config=V7Config(),
    )
    step(pool.keys["7"], optimizer, scheduler, 5)
    continuous_key = pool.keys["7"].detach().clone()
    continuous_optimizer = optimizer.state_dict()

    payload, resumed_pool, _ = load_v7_checkpoint(str(path), restore_rng=False)
    resumed_optimizer = torch.optim.AdamW([resumed_pool.keys["7"]], lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1, gamma=0.9
    )
    resumed_optimizer.load_state_dict(payload["optimizer"])
    resumed_scheduler.load_state_dict(payload["scheduler"])
    step(resumed_pool.keys["7"], resumed_optimizer, resumed_scheduler, 5)
    assert torch.allclose(resumed_pool.keys["7"], continuous_key, atol=1e-8)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    resumed_state = next(iter(resumed_optimizer.state_dict()["state"].values()))
    continuous_state = next(iter(continuous_optimizer["state"].values()))
    assert torch.allclose(resumed_state["exp_avg"], continuous_state["exp_avg"])
    assert torch.equal(payload["candidate_lora_state"][next(iter(lora_state))], torch.tensor([1.25]))
    assert payload["candidate_usage_counters"] == counters


def test_33_v7_stage_markers_are_bound_to_the_run_contract(tmp_path):
    from compose.experiments.v7_task_run import (
        bind_run_contract,
        mark,
        stage_done,
    )

    contract = {"schema_version": 1, "contract_hash": "contract-a"}
    assert bind_run_contract(tmp_path, contract, resume=False, had_entries=False) == "contract-a"
    mark(tmp_path, "s0_full_data", "contract-a")
    assert stage_done(tmp_path, "s0_full_data", "contract-a") is True
    with pytest.raises(ValueError, match="stale V7 stage marker"):
        stage_done(tmp_path, "s0_full_data", "contract-b")
    with pytest.raises(ValueError, match="resume contract mismatch"):
        bind_run_contract(
            tmp_path,
            {"schema_version": 1, "contract_hash": "contract-b"},
            resume=True,
            had_entries=True,
        )


def test_34_formal_recipe_records_single_process_effective_global_batch(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from compose.experiments.v7_task_run import build_run_contract

    files = {}
    for name in ("config", "train", "validation", "test", "annotation"):
        path = tmp_path / (name + ".json")
        path.write_text("{}", encoding="utf-8")
        files[name] = str(path)
    args = SimpleNamespace(
        config=files["config"], train_file=files["train"],
        val_file=files["validation"], test_file=files["test"],
        validation_annotation_file=files["annotation"], task_index=0,
        task_name="ImageNet-R", validation_metric="official_ucit",
        previous_checkpoint=None, model_path=str(tmp_path / "model"),
        vision_tower=str(tmp_path / "vision"),
        projector_path=str(tmp_path / "projector.bin"),
        image_folder=str(tmp_path / "images"), smoke_max_steps=None,
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    contract = build_run_contract(args, V7Config(), True, 64)
    assert contract["recipe"]["effective_global_batch_size"] == 64
    assert contract["recipe"]["world_size"] == 1
    assert contract["recipe"]["max_steps"] == -1
    assert contract["recipe"]["max_samples"] is None

    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="single-process"):
        build_run_contract(args, V7Config(), True, 64)


def test_35_three_rank_recipe_records_batch_63_without_lr_scaling(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from compose.experiments.v7_task_run import build_run_contract

    paths = []
    for name in ("config", "train", "validation", "test", "annotation"):
        path = tmp_path / (name + ".json")
        path.write_text("{}", encoding="utf-8")
        paths.append(str(path))
    args = SimpleNamespace(
        config=paths[0], train_file=paths[1], val_file=paths[2],
        test_file=paths[3], validation_annotation_file=paths[4], task_index=0,
        task_name="ImageNet-R", validation_metric="official_ucit",
        previous_checkpoint=None, model_path=str(tmp_path / "model"),
        vision_tower=str(tmp_path / "vision"),
        projector_path=str(tmp_path / "projector.bin"),
        image_folder=str(tmp_path / "images"), smoke_max_steps=None,
        training_world_size=3, training_gpus="0,1,2",
        distributed_backend="nccl",
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    contract = build_run_contract(args, V7Config(), True, 21)
    recipe = contract["recipe"]
    assert recipe["world_size"] == 3
    assert recipe["effective_global_batch_size"] == 63
    assert recipe["target_global_batch_size"] == 63
    assert recipe["global_batch_relative_difference"] == 0.0
    assert recipe["learning_rate"] == V7Config().training.learning_rate

    args.training_per_device_batch_size = 3
    args.training_dataloader_num_workers = 8
    optimized = build_run_contract(args, V7Config(), True, 7)["recipe"]
    assert optimized["per_device_train_batch_size"] == 3
    assert optimized["gradient_accumulation_steps"] == 7
    assert optimized["world_size"] == 3
    assert optimized["effective_global_batch_size"] == 63
    assert optimized["dataloader_num_workers"] == 8
    for locked in (
        "num_train_epochs", "learning_rate", "weight_decay", "warmup_ratio",
        "lr_scheduler_type", "bf16", "gradient_checkpointing", "seed",
        "dataloader_drop_last",
    ):
        assert optimized[locked] == recipe[locked]


def test_36_v7_ddp_anchor_exposes_all_current_keys(monkeypatch):
    import compose.v7.hf_trainer as module
    from compose.v7.pool import V7ExpertKeyPool

    monkeypatch.setattr(module, "_distributed", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)
    pool = V7ExpertKeyPool()
    pool.add(0, torch.ones(1536), 0, "current", True)
    pool.add(1, torch.arange(1536, dtype=torch.float32), 0, "current", True)

    class Model(nn.Module):
        def forward(self, value):
            return {"loss": value.square().mean()}

    model = Model()
    module.attach_v7_ddp_key_anchor(model, pool)
    output = model(torch.tensor([2.0], requires_grad=True))
    output["loss"].backward()
    assert all(parameter.grad is not None for parameter in pool.keys.values())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in pool.keys.values())


def test_37_v7_formal_evaluator_has_exact_lower_triangle_and_task_free_command(tmp_path):
    from compose.eval.v7_formal_ucit_eval import generation_command, lower_triangle_cells

    assert len(lower_triangle_cells()) == 21
    assert lower_triangle_cells()[0] == (0, 0)
    assert lower_triangle_cells()[-1] == (5, 5)
    task_root = tmp_path / "task2" / "data"
    task_root.mkdir(parents=True)
    (task_root / "query_contract.json").write_text(
        json.dumps({"backbone_hash": "abc"}), encoding="utf-8"
    )
    formal = {
        "data": {
            "model_path": "model", "vision_tower": "vision",
            "projector_path": "projector", "image_folder": "images",
        },
        "tasks": [{"test_file": "test0"}, {"test_file": "test1"}],
    }
    method = {"query": {"path": "query-model"}}
    _answers, command = generation_command(
        tmp_path, formal, method, 2, 1, "python"
    )
    assert "--v7-key-state" in command
    assert "--query-backbone-hash" in command
    assert "--expert-ids" not in command
    assert "--task-id" not in command


def test_32_nll_eval_reuses_training_preprocessing_mask():
    import compose.eval.nll_eval as nll_eval

    source = inspect.getsource(nll_eval)
    assert "LazySupervisedDataset" in source
    assert "DataCollatorForSupervisedDataset" in source
    assert "raw_labels = input_ids.clone()" not in source
