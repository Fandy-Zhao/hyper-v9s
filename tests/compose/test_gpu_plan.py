"""Pure-CPU unit tests for the V7 adaptive GPU planner (spec 0903 §20-A)."""

import json
import os

import pytest

from compose.v7.gpu_plan import (
    DEFAULT_PER_DEVICE_BATCH,
    DEFAULT_TARGET_GLOBAL_BATCH,
    GPULeasePool,
    GpuPlanError,
    V7GPUPlan,
    allocate_stage_gpus,
    detect_idle_gpu_ids,
    parse_gpu_ids,
    plan_training_recipe,
    resolve_available_gpu_ids,
)


# ---------------------------------------------------------------------------
# Id parsing and detection
# ---------------------------------------------------------------------------


def test_parse_gpu_ids_accepts_whitespace_and_deduplicates():
    assert parse_gpu_ids(" 0 , 1,0, 2 ") == [0, 1, 2]
    assert parse_gpu_ids("4,5,6,7") == [4, 5, 6, 7]


def test_parse_gpu_ids_rejects_garbage():
    for bad in ("", "  ", "abc", "0,-1", "0,1,x"):
        with pytest.raises(GpuPlanError):
            parse_gpu_ids(bad)
    # Interior empty tokens are tolerated like the shell would ignore them.
    assert parse_gpu_ids("0,1,,2") == [0, 1, 2]


def test_resolution_precedence_cli_beats_env(monkeypatch):
    monkeypatch.setenv("V7_GPUS", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert resolve_available_gpu_ids(cli_ids="0,1") == [0, 1]
    assert resolve_available_gpu_ids(cli_ids=None) == [2, 3]
    monkeypatch.delenv("V7_GPUS")
    assert resolve_available_gpu_ids(cli_ids=None) == [4, 5]


def test_resolution_uses_provided_env_without_touching_process(monkeypatch):
    # A caller-supplied env dict must fully shadow the process env.
    monkeypatch.setenv("V7_GPUS", "0,1,2,3")
    env = {"V7_GPUS": "6,7"}
    assert resolve_available_gpu_ids(cli_ids=None, env=env) == [6, 7]


def test_resolution_fails_closed_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("V7_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "compose.v7.gpu_plan.probe_gpu_state",
        lambda: {0: (0, 0), 1: (0, 0)},
    )
    assert resolve_available_gpu_ids() == [0, 1]
    with pytest.raises(GpuPlanError, match="automatic detection is disabled"):
        resolve_available_gpu_ids(allow_auto_detect=False)


def test_detect_idle_only_picks_truly_free_devices(monkeypatch):
    monkeypatch.setattr(
        "compose.v7.gpu_plan.probe_gpu_state",
        lambda: {
            0: (4, 0),          # free
            1: (20480, 95),     # busy: memory + utilization
            2: (2048, 0),       # busy: above the 1024 MiB threshold
            3: (4, 30),         # busy: utilization
        },
    )
    assert detect_idle_gpu_ids() == [0]


def test_detect_idle_fails_when_all_busy(monkeypatch):
    monkeypatch.setattr(
        "compose.v7.gpu_plan.probe_gpu_state",
        lambda: {0: (20000, 100)},
    )
    with pytest.raises(GpuPlanError, match="no idle GPU"):
        detect_idle_gpu_ids()


# ---------------------------------------------------------------------------
# Strict recipe matrix (spec §5: 1..8 GPUs)
# ---------------------------------------------------------------------------


def test_strict_recipe_matrix_one_to_eight_gpus():
    """The strict formal rule: largest divisor world size, GA = 64 / ws.

    1 GPU -> 1x64, 2 -> 2x32, 3 -> 2x32 (GPU stays idle), 4 -> 4x16,
    5/6/7 -> 4x16, 8 -> 8x8.  Never 63/60/70.
    """
    expected = {
        1: (1, 64),
        2: (2, 32),
        3: (2, 32),
        4: (4, 16),
        5: (4, 16),
        6: (4, 16),
        7: (4, 16),
        8: (8, 8),
    }
    for count, (world_size, accumulation) in expected.items():
        plan = plan_training_recipe(count)
        assert plan.recipe_mode == "strict"
        assert plan.training_world_size == world_size, "n={}".format(count)
        assert plan.gradient_accumulation_steps == accumulation, "n={}".format(count)
        assert plan.effective_global_batch == DEFAULT_TARGET_GLOBAL_BATCH
        assert plan.recipe_exact is True
        assert plan.global_batch_delta == 0
        assert plan.relative_difference == 0.0
        assert (
            plan.per_device_batch * plan.training_world_size
            * plan.gradient_accumulation_steps
        ) == DEFAULT_TARGET_GLOBAL_BATCH


def test_strict_recipe_keeps_world_size_under_available_count():
    plan = plan_training_recipe(3)
    assert plan.training_world_size == 2
    plan = plan_training_recipe(6)
    assert plan.training_world_size == 4


def test_throughput_mode_uses_all_gpus_and_reports_recipe_inexact():
    cases = {1: 64, 2: 64, 3: 63, 4: 64, 5: 65, 6: 66, 7: 63, 8: 64}
    for count, effective in cases.items():
        plan = plan_training_recipe(count, recipe_mode="throughput")
        assert plan.training_world_size == count
        assert plan.effective_global_batch == effective
        assert plan.recipe_exact is False
        assert plan.global_batch_delta == effective - 64
        assert plan.relative_difference == (effective - 64) / 64.0


def test_recipe_rejects_invalid_mode_and_counts():
    with pytest.raises(GpuPlanError, match="strict.*throughput"):
        plan_training_recipe(2, recipe_mode="approximate")
    with pytest.raises(GpuPlanError, match="at least one GPU"):
        plan_training_recipe(0)
    with pytest.raises(GpuPlanError, match="does not divide"):
        plan_training_recipe(2, target_global_batch=63, per_device_batch=2)


# ---------------------------------------------------------------------------
# Stage allocation (spec §15)
# ---------------------------------------------------------------------------


def test_allocate_stages_query_rms_prune_eval_use_all_gpus():
    stages = allocate_stage_gpus([0, 1, 2], training_world_size=2)
    assert stages["query"] == [0, 1, 2]
    assert stages["rms"] == [0, 1, 2]
    assert stages["pruning"] == [0, 1, 2]
    assert stages["evaluation"] == [0, 1, 2]
    assert stages["training"] == [0, 1]  # GPU 2 idles during S3


def test_allocate_stages_is_deterministic_and_sorted():
    first = allocate_stage_gpus([3, 1, 2], training_world_size=2)
    second = allocate_stage_gpus([3, 1, 2], training_world_size=2)
    assert first == second
    assert first["training"] == [1, 2]
    assert first["query"] == [1, 2, 3]


def test_allocate_rejects_world_size_larger_than_available():
    with pytest.raises(GpuPlanError, match="exceeds available"):
        allocate_stage_gpus([0, 1], training_world_size=4)


# ---------------------------------------------------------------------------
# Full plan construction and serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ids", [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [4, 5, 6, 7]])
def test_full_plan_matrix_consistency(ids):
    plan = V7GPUPlan.build(ids)
    assert plan.available_gpu_ids == tuple(sorted(ids))
    assert plan.training_world_size == plan.recipe.training_world_size
    assert plan.training_gpu_ids == tuple(sorted(ids)[: plan.training_world_size])
    for stage in ("query", "rms", "pruning", "evaluation"):
        ids_attr = "{}_gpu_ids".format(stage)
        size_attr = "{}_world_size".format(stage)
        assert len(getattr(plan, ids_attr)) == getattr(plan, size_attr)
        assert set(getattr(plan, ids_attr)) == set(ids)
    assert plan.recipe_exact is True


def test_plan_serializes_to_gpu_plan_json_contract():
    plan = V7GPUPlan.build([0, 1, 2, 3])
    payload = json.loads(json.dumps(plan.to_dict()))
    assert payload["schema_version"] == 1
    assert payload["available_gpu_ids"] == [0, 1, 2, 3]
    assert payload["query_world_size"] == 4
    assert payload["training_world_size"] == 4
    assert payload["rms_world_size"] == 4
    assert payload["pruning_world_size"] == 4
    assert payload["evaluation_world_size"] == 4
    assert payload["training_per_device_batch"] == 1
    assert payload["training_gradient_accumulation_steps"] == 16
    assert payload["effective_global_batch"] == 64
    assert payload["target_global_batch"] == 64
    assert payload["recipe_exact"] is True

    three = V7GPUPlan.build([0, 1, 2])
    assert three.to_dict()["training_world_size"] == 2
    assert three.to_dict()["training_gradient_accumulation_steps"] == 32
    assert three.to_dict()["query_world_size"] == 3


def test_plan_render_prints_the_full_banner():
    plan = V7GPUPlan.build([0, 1, 2, 3])
    text = plan.render_plan_block()
    assert "V7 Adaptive GPU Execution Plan" in text
    assert "Query workers:" in text and "4" in text
    assert "4-rank DDP" in text
    assert "gradient accumulation: 16" in text
    assert "global batch: 64" in text
    assert "recipe exact: YES" in text


def test_recipe_equivalence_formula_b2_micro_batch_is_documented_not_default():
    # Spec §6: the first adaptive version keeps per_device_batch=1; the
    # 2 x 4 x 8 = 64 arrangement is a later candidate.  Planner must honor
    # an explicit per-device batch if it is ever unlocked.
    plan = plan_training_recipe(4, per_device_batch=2)
    assert plan.training_world_size == 4
    assert plan.gradient_accumulation_steps == 8
    assert plan.effective_global_batch == 64
    assert V7GPUPlan.build([0, 1, 2, 3]).recipe.per_device_batch == 1


# ---------------------------------------------------------------------------
# Lease bookkeeping
# ---------------------------------------------------------------------------


def test_lease_pool_assigns_each_gpu_once_and_rotates(tmp_path):
    log = str(tmp_path / "stage_gpu_usage.jsonl")
    pool = GPULeasePool([0, 1, 2], usage_log_path=log)
    acquired = [pool.acquire() for _ in range(5)]
    # Round-robin over three devices.
    assert acquired == [0, 1, 2, 0, 1]
    assert os.path.exists(log)
    actions = [json.loads(line) for line in open(log, encoding="utf-8")]
    assert len(actions) == 5
    assert {row["gpu_id"] for row in actions} == {0, 1, 2}
    assert all(row["action"] == "acquire" for row in actions)


def test_lease_pool_requires_a_gpu():
    with pytest.raises(GpuPlanError):
        GPULeasePool([])
