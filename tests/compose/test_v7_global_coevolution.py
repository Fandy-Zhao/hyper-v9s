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
    supervised_token_mask,
    teacher_forcing_token_nll,
)
from compose.v7.provenance import (
    audit_split_isolation,
    build_runtime_contract,
    validate_runtime_contract,
)
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
    source.mkdir()
    state = {
        "model.layers.0.self_attn.q_proj.experts.{}.{}.weight".format(i, part): torch.ones(1, 1)
        for i in pool.expert_ids for part in ("lora_A", "lora_B")
    }
    torch.save(state, source / "compose_experts.bin")
    manifest = {
        "format_version": 1,
        "adapter": {"rank": 8, "alpha": 16.0, "dropout": 0.0, "layers": ["model.layers.0.self_attn.q_proj"]},
        "experts": [
            {"expert_id": i, "adapter_name": "e{}".format(i), "rank": 8, "alpha": 16.0, "status": "frozen"}
            for i in pool.expert_ids
        ],
    }
    (source / "compose_experts.json").write_text(json.dumps(manifest))
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
    audit = audit_split_isolation(train, val, test)
    assert audit["test_data_used_for_pruning"] is False
    with pytest.raises(ValueError, match="same normalized path"):
        audit_split_isolation(train, val, val)
    duplicate = write("duplicate.json", json.loads(Path(val).read_text()))
    with pytest.raises(ValueError, match="identical file hashes"):
        audit_split_isolation(train, val, duplicate)
    overlap = write("overlap.json", [{"id": "z", "image": "a.jpg", "text": "qa", "answer": "x"}])
    with pytest.raises(ValueError, match="record overlap"):
        audit_split_isolation(train, val, overlap)

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
