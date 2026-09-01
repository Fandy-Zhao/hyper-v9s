"""Low-cost CPU Task0/Task1 V7 smoke with real sparse ComposeLinear gradients."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.v7.checkpoint import load_v7_checkpoint, save_v7_checkpoint
from compose.v7.config import V7Config
from compose.v7.pool import V7ExpertKeyPool, initialize_candidate_keys, tensor_checksum
from compose.v7.routing import GlobalTop2Router
from compose.v7.training import V7JsonlLogger, V7StepEngine


def normalize(value):
    return torch.nn.functional.normalize(value.float(), dim=-1)


def make_layer():
    base = nn.Linear(8, 4, bias=False)
    nn.init.zeros_(base.weight)
    base.requires_grad_(False)
    return ComposeLinear(base, rank=8, alpha=16.0)


def add_lora(layer, expert_ids, trainable=True):
    for expert_id in expert_ids:
        expert = layer.add_expert(expert_id)
        expert.requires_grad_(trainable)


def train_task(root, task_index, layer, pool, queries, steps):
    current = pool.current_ids
    parameters = [pool.keys[str(value)] for value in current]
    parameters += [
        parameter
        for value in current
        for parameter in layer.experts[str(value)].parameters()
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-2, weight_decay=0.0)
    engine = V7StepEngine(
        GlobalTop2Router(pool), 0.1,
        logger=V7JsonlLogger(str(root / "task{}".format(task_index) / "steps.jsonl")),
    )
    lora_grad_seen = False
    key_grad_seen = False
    finite = True
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch = queries[(step * 3) % len(queries): (step * 3) % len(queries) + 3]
        if batch.shape[0] < 3:
            batch = torch.cat([batch, queries[: 3 - batch.shape[0]]])
        inputs = torch.randn(batch.shape[0], 8)
        target = torch.ones(batch.shape[0], 4)

        def answer_loss(_selection):
            return (layer(inputs) - target).square().mean()

        total, metrics, _ = engine.compute(batch, answer_loss)
        did_backward = engine.backward(total, metrics)
        if did_backward:
            lora_grad_seen |= any(
                parameter.grad is not None and bool(parameter.grad.ne(0).any())
                for value in current
                for parameter in layer.experts[str(value)].parameters()
            )
            key_grad_seen |= any(
                pool.keys[str(value)].grad is not None
                and bool(pool.keys[str(value)].grad.ne(0).any())
                for value in current
            )
            finite &= bool(torch.isfinite(total))
            optimizer.step()
        engine.finish_step(metrics)
    summary = engine.summary()
    summary.update({
        "task_index": task_index,
        "steps": steps,
        "lora_gradient_seen": lora_grad_seen,
        "key_gradient_seen": key_grad_seen,
        "finite": finite,
        "candidate_usage_not_all_identical": len(set(engine.selection_counts.values())) > 1,
    })
    return summary, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(42)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = V7Config()
    layer = make_layer()

    # Task0: all samples and all selectable experts are current candidates.
    task0_queries = normalize(torch.randn(24, 1536) * 0.15 + torch.eye(1536)[0])
    keys0, center0, init0 = initialize_candidate_keys(task0_queries, 24, seed=42)
    pool = V7ExpertKeyPool()
    for expert_id, key in enumerate(keys0):
        pool.add(expert_id, key, 0, "current", True)
    add_lora(layer, pool.current_ids)
    task0, optimizer0 = train_task(root, 0, layer, pool, task0_queries, args.steps)
    pool.commit(pool.current_ids, {value: {"smoke_keep": True} for value in pool.current_ids})
    for value in pool.historical_ids:
        layer.experts[str(value)].requires_grad_(False)
    task0["candidate_initialization"] = init0
    task0["retained_candidate_count"] = 4
    task0["pool_size_after_task"] = 4

    checkpoint = root / "task0" / "resume.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_v7_checkpoint(
        str(checkpoint), task_index=0, training_step=args.steps, key_pool=pool,
        candidate_lora_state={}, optimizer=optimizer0, scheduler=None,
        usage_counters=task0["selection_count"], config=config, rms_state={},
    )
    _, resumed, _ = load_v7_checkpoint(str(checkpoint), restore_rng=False)
    task0["resume_historical_frozen"] = all(
        not resumed.keys[str(value)].requires_grad for value in resumed.historical_ids
    )

    # Task1: construct a full split with explicit OldOld, OldNew and NewNew
    # geometry, then initialize all four candidates around its one task center.
    old_keys = pool.normalized(pool.historical_ids)
    new_center = torch.eye(1536)[10]
    oldold = normalize(old_keys[0] + old_keys[1]).repeat(8, 1)
    oldnew = normalize(old_keys[0] + new_center).repeat(8, 1)
    newnew = normalize(new_center + torch.randn(8, 1536) * 0.01)
    task1_queries = torch.cat([oldold, oldnew, newnew])
    keys1, center1, init1 = initialize_candidate_keys(task1_queries, 24, seed=43)
    for offset, key in enumerate(keys1, start=4):
        pool.add(offset, key, 1, "current", True)
    add_lora(layer, pool.current_ids)
    historical_key_before = pool.historical_checksums()
    historical_lora_before = {
        value: tensor_checksum(layer.experts[str(value)].lora_B.weight)
        for value in pool.historical_ids
    }
    task1, _ = train_task(root, 1, layer, pool, task1_queries, args.steps)
    task1["candidate_initialization"] = init1
    task1["historical_key_unchanged"] = pool.historical_checksums() == historical_key_before
    task1["historical_lora_unchanged"] = historical_lora_before == {
        value: tensor_checksum(layer.experts[str(value)].lora_B.weight)
        for value in pool.historical_ids
    }
    task1["all_route_types_observed"] = all(
        task1[name] > 0 for name in ("OldOldRate", "OldNewRate", "NewNewRate")
    )

    result = {"task0": task0, "task1": task1}
    checks = {
        "task0_gradients": task0["lora_gradient_seen"] and task0["key_gradient_seen"],
        "task0_finite": task0["finite"],
        "task0_resume": task0["resume_historical_frozen"],
        "task1_gradients": task1["lora_gradient_seen"] and task1["key_gradient_seen"],
        "task1_routes": task1["all_route_types_observed"],
        "historical_frozen": task1["historical_key_unchanged"] and task1["historical_lora_unchanged"],
    }
    result["checks"] = checks
    (root / "smoke_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all(checks.values()):
        raise AssertionError("V7 smoke checks failed: {}".format(checks))
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
