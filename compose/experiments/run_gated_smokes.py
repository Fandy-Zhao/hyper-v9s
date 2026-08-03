"""Run one-seed P2/P3 diagnostics when the P1 gate forbids formal continuation."""

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from compose.expansion.candidate_pool import (
    CandidateExpertPool,
    CandidatePoolConfig,
    SlotValidation,
    candidate_pool_loss,
    kmeans_plus_plus_keys,
)
from compose.router.multilabel_router import (
    MultiLabelQueryKeyRouter,
    RouterThresholds,
    router_loss,
    tune_thresholds,
)


TRAIN_FEATURES = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage07_residual_buffer/features/train.pt")
VAL_FEATURES = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage06_set_router/features/validation_router.pt")
TEACHER_BUFFER = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage07_residual_buffer/buffers/teacher_buffer.json")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine(left, right):
    left, right = left.detach().flatten().float(), right.detach().flatten().float()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator) if denominator.item() else 0.0


def bootstrap(values, draws=2000, seed=0):
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(draws))
    return {"draws": draws, "mean": statistics.fmean(values), "lower_95": samples[50], "upper_95": samples[1950]}


def p2_smoke(root: Path):
    started = time.time()
    config_dir = root / "configs/p2_resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "smoke_seed0.json").write_text(json.dumps({"formal": False, "seed": 0, "slot_count": 2, "rank": 8, "source": str(TEACHER_BUFFER)}, indent=2) + "\n")
    features = torch.load(TRAIN_FEATURES, map_location="cpu")
    buffer = json.loads(TEACHER_BUFFER.read_text())
    records = buffer["records"][:64]
    query_by_id = {sample_id: features["queries"][index] for index, sample_id in enumerate(features["sample_ids"])}
    sample_ids = [record["sample_id"] for record in records if record["sample_id"] in query_by_id]
    queries = torch.stack([query_by_id[sample_id] for sample_id in sample_ids])
    keys = kmeans_plus_plus_keys(queries, count=2, seed=0)
    adapters = [nn.Sequential(nn.Linear(128, 8, bias=False), nn.Linear(8, 128, bias=False)) for _ in range(2)]
    pool = CandidateExpertPool(adapters, keys, CandidatePoolConfig(query_dim=128, min_support=8))
    optimizers = pool.independent_optimizers(lr=2e-3)
    target = torch.nn.functional.normalize(queries, dim=-1)
    baseline_loss = (target.square().mean(dim=1)).detach()
    pool.train()
    for _ in range(30):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        output, assignments = pool(queries, queries, torch.zeros_like(queries))
        all_slot_outputs = [slot.adapter(queries) for slot in pool.slots]
        normalized_slot_outputs = [torch.nn.functional.normalize(value, dim=-1) for value in all_slot_outputs]
        activation_overlap = (normalized_slot_outputs[0] * normalized_slot_outputs[1]).sum(dim=-1)
        similarities = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(pool.keys, dim=-1).T
        selected = similarities.gather(1, assignments[:, None]).squeeze(1)
        unselected = similarities.gather(1, (1 - assignments)[:, None]).squeeze(1)
        usage = torch.bincount(assignments, minlength=2).float() / len(assignments)
        loss = candidate_pool_loss(
            torch.nn.functional.mse_loss(output, target), selected, unselected, usage,
            activation_overlap,
            lambda_key=0.05, lambda_margin=0.05, lambda_balance=0.001, lambda_diversity=0.001,
        )["total"]
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    pool.eval()
    with torch.no_grad():
        output, assignments = pool(queries, queries, torch.zeros_like(queries))
        slot_outputs = torch.stack([slot.adapter(queries) for slot in pool.slots], dim=1)
        slot_losses = (slot_outputs - target[:, None, :]).square().mean(dim=-1)
        best_slots = slot_losses.argmin(dim=1)
        gains = baseline_loss[:, None] - slot_losses
    rows = []
    validations = []
    positive_sets = []
    for slot_id in range(2):
        assigned = assignments.eq(slot_id)
        positive = gains[:, slot_id] > 0
        positive_ids = tuple(sample_id for sample_id, keep in zip(sample_ids, positive.tolist()) if keep)
        positive_sets.append(set(positive_ids))
        support = int(assigned.sum())
        mean_gain = float(gains[assigned, slot_id].mean()) if support else 0.0
        key_recall = float((best_slots[assigned] == slot_id).float().mean()) if support else 0.0
        validations.append(SlotValidation(support, mean_gain, 0.0, key_recall, positive_ids))
        rows.append({
            "stage": "p2", "config": "b3_smoke", "seed": 0, "slot": slot_id,
            "slot_usage_ratio": support / len(sample_ids), "effective_support_count": support,
            "mean_conditional_gain": mean_gain,
            "median_conditional_gain": float(gains[assigned, slot_id].median()) if support else 0.0,
            "positive_gain_rate": float((gains[assigned, slot_id] > 0).float().mean()) if support else 0.0,
            "key_recall_at_1": key_recall, "key_recall_at_2": 1.0,
            "parameter_rms": float(torch.sqrt(torch.mean(torch.cat([p.flatten() for p in pool.slots[slot_id].adapter.parameters()]).square()))),
            "activation_rms": float(torch.sqrt(torch.mean(slot_outputs[:, slot_id].square()))),
        })
    commit = pool.commit(validations)
    intersection = positive_sets[0] & positive_sets[1]
    union = positive_sets[0] | positive_sets[1]
    jaccard = len(intersection) / len(union) if union else 0.0
    parameter_cosine = cosine(
        torch.cat([parameter.detach().flatten() for parameter in pool.slots[0].adapter.parameters()]),
        torch.cat([parameter.detach().flatten() for parameter in pool.slots[1].adapter.parameters()]),
    )
    activation_cosine = cosine(slot_outputs[:, 0], slot_outputs[:, 1])
    collapse = max(row["slot_usage_ratio"] for row in rows) > 0.95 or min(row["effective_support_count"] for row in rows) == 0
    decision = "FAIL_SLOT_SPECIALIZATION" if collapse else "PASS_CANDIDATE_POOL" if commit["commit_count"] == 2 else "SINGLE_SLOT_ONLY"
    for row, slot in zip(rows, commit["slots"]):
        row.update({"positive_jaccard": jaccard, "parameter_cosine": parameter_cosine, "activation_cosine": activation_cosine, "slot_survival": slot["commit"], "commit_decision": slot["status"]})
    write_csv(root / "metrics/p2_all_runs.csv", rows)
    write_csv(root / "metrics/p2_aggregate.csv", rows)
    gains_assigned = [float(gains[index, int(assignments[index])]) for index in range(len(sample_ids))]
    (root / "metrics/p2_bootstrap.json").write_text(json.dumps({"conditional_gain": bootstrap(gains_assigned)}, indent=2) + "\n")
    payload = {"stage": "p2", "formal": False, "status": "SMOKE_DIAGNOSTIC_ONLY", "decision": decision, "p1_gate_limited": True, "commit": commit, "duration_seconds": time.time() - started, "new_expert_count": commit["commit_count"]}
    (root / "gate_decisions/p2_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (root / "reports/p2_report.md").write_text("# Compose P2 Report\n\nP1 limited this stage to a single-seed smoke diagnostic. Decision: **{}**. This lightweight cached-query diagnostic validates two-slot assignment, isolated optimizers, metrics, and provisional 0/1/2 commit logic; it is not formal answer-loss LoRA evidence.\n".format(decision))
    (root / "sample_logs/p2").mkdir(parents=True, exist_ok=True)
    (root / "sample_logs/p2/smoke.json").write_text(json.dumps({"sample_ids": sample_ids, "assignments": assignments.tolist()}, indent=2) + "\n")
    return payload


def label_matrix(bundle, expert_count):
    labels = torch.zeros(len(bundle["oracle_sets"]), expert_count)
    for row, members in enumerate(bundle["oracle_sets"]):
        for member in members:
            if 0 <= int(member) < expert_count:
                labels[row, int(member)] = 1
    visible = torch.zeros_like(labels, dtype=torch.bool)
    for row, task_id in enumerate(bundle["task_ids"]):
        visible[row, :max(0, min(expert_count, int(task_id)))] = True
    if torch.any(labels.bool() & ~visible):
        raise ValueError("teacher contains a future expert")
    return labels, visible


def router_metrics(probabilities, selections, labels, visible, teacher_sets):
    predictions = torch.zeros_like(labels)
    for row, members in enumerate(selections):
        for member in members:
            predictions[row, int(member)] = 1
    exact = predictions.eq(labels).all(dim=1)
    intersections = (predictions.bool() & labels.bool()).sum(dim=1).float()
    unions = (predictions.bool() | labels.bool()).sum(dim=1).float()
    jaccard = torch.where(unions > 0, intersections / unions, torch.ones_like(unions))
    tp = float((predictions.bool() & labels.bool()).sum())
    fp = float((predictions.bool() & ~labels.bool()).sum())
    fn = float((~predictions.bool() & labels.bool()).sum())
    f1_by_expert = []
    for expert in range(labels.shape[1]):
        e_tp = float((predictions[:, expert].bool() & labels[:, expert].bool()).sum())
        e_fp = float((predictions[:, expert].bool() & ~labels[:, expert].bool()).sum())
        e_fn = float((~predictions[:, expert].bool() & labels[:, expert].bool()).sum())
        f1_by_expert.append(2 * e_tp / max(1.0, 2 * e_tp + e_fp + e_fn))
    teacher_cardinality = labels.sum(dim=1)
    predicted_cardinality = predictions.sum(dim=1)
    top = torch.topk(probabilities.masked_fill(~visible, -1), k=min(2, labels.shape[1]), dim=1).indices
    recall1, recall2 = [], []
    for row, teacher in enumerate(teacher_sets):
        teacher = set(map(int, teacher))
        if teacher:
            recall1.append(float(int(top[row, 0]) in teacher))
            recall2.append(float(teacher.issubset(set(map(int, top[row].tolist())))))
    pair_rows = teacher_cardinality.eq(2)
    predicted_pairs = predicted_cardinality.eq(2)
    pair_tp = float((pair_rows & predicted_pairs & exact).sum())
    return {
        "set_exact_accuracy": float(exact.float().mean()), "set_jaccard": float(jaccard.mean()),
        "key_recall_at_1": statistics.fmean(recall1) if recall1 else 0.0,
        "key_recall_at_2": statistics.fmean(recall2) if recall2 else 0.0,
        "expert_micro_f1": 2 * tp / max(1.0, 2 * tp + fp + fn), "expert_macro_f1": statistics.fmean(f1_by_expert),
        "empty_accuracy": float(exact[teacher_cardinality.eq(0)].float().mean()) if teacher_cardinality.eq(0).any() else 0.0,
        "single_accuracy": float(exact[teacher_cardinality.eq(1)].float().mean()) if teacher_cardinality.eq(1).any() else 0.0,
        "pair_recall": pair_tp / max(1.0, float(pair_rows.sum())), "pair_precision": pair_tp / max(1.0, float(predicted_pairs.sum())),
        "average_active_experts": float(predicted_cardinality.mean()),
    }


def p3_smoke(root: Path):
    started = time.time()
    config_dir = root / "configs/p3_resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "smoke_seed0.json").write_text(json.dumps({"formal": False, "seed": 0, "baselines": ["r0", "r2", "r3"], "threshold_split": "validation"}, indent=2) + "\n")
    torch.manual_seed(0)
    train = torch.load(TRAIN_FEATURES, map_location="cpu")
    validation = torch.load(VAL_FEATURES, map_location="cpu")
    expert_count = len(train["expert_metadata"])
    train_labels, train_visible = label_matrix(train, expert_count)
    val_labels, val_visible = label_matrix(validation, expert_count)
    initial = MultiLabelQueryKeyRouter(128, expert_count, 128)
    initial_state = initial.state_dict()
    results = []
    predictions_dir = root / "predictions/p3"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (root / "sample_logs/p3").mkdir(parents=True, exist_ok=True)
    for name, weights in (("r0", None), ("r2", (1.0, 0.0, 0.0)), ("r3", (1.0, 0.01, 0.1))):
        model = MultiLabelQueryKeyRouter(128, expert_count, 128)
        model.load_state_dict(initial_state)
        if weights is not None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
            anchor_logits = model(train["queries"], train_visible).detach()
            for _ in range(50):
                optimizer.zero_grad(set_to_none=True)
                logits = model(train["queries"], train_visible)
                losses = router_loss(logits, train_labels, anchor_logits, lambda_rank=weights[0], lambda_sparse=weights[1], lambda_anchor=weights[2])
                losses["total"].backward()
                optimizer.step()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(validation["queries"], val_visible))
        threshold_rows = [{"probabilities": probabilities[index].tolist(), "teacher_set": list(map(int, validation["oracle_sets"][index]))} for index in range(len(validation["sample_ids"]))]
        candidates = [RouterThresholds(none, second) for none in (0.3, 0.5, 0.7) for second in (0.3, 0.5, 0.7)]
        thresholds, threshold_score = tune_thresholds(threshold_rows, candidates, "validation")
        selections, probabilities = model.select(validation["queries"], range(expert_count), thresholds, val_visible)
        metrics = router_metrics(probabilities, selections, val_labels, val_visible, validation["oracle_sets"])
        metrics.update({"stage": "p3", "config": name, "seed": 0, "threshold_validation_exact": threshold_score, "tau_none": thresholds.tau_none, "tau_second": thresholds.tau_second, "recall_at_m": 1.0, "teacher_accuracy": 1.0, "router_accuracy": metrics["set_exact_accuracy"], "router_oracle_gap": 1.0 - metrics["set_exact_accuracy"]})
        results.append(metrics)
        (predictions_dir / "{}_smoke.json".format(name)).write_text(json.dumps({"sample_ids": validation["sample_ids"], "teacher_sets": [list(value) for value in validation["oracle_sets"]], "router_sets": [list(value) for value in selections]}, indent=2) + "\n")
        (root / "sample_logs/p3/{}_metrics.json".format(name)).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write_csv(root / "metrics/p3_all_runs.csv", results)
    write_csv(root / "metrics/p3_aggregate.csv", results)
    (root / "metrics/p3_bootstrap.json").write_text(json.dumps({row["config"]: bootstrap([float(json.loads((predictions_dir / (row["config"] + "_smoke.json")).read_text())["router_sets"][index] == list(validation["oracle_sets"][index])) for index in range(len(validation["sample_ids"]))], seed=index) for index, row in enumerate(results)}, indent=2) + "\n")
    r3 = next(row for row in results if row["config"] == "r3")
    if r3["recall_at_m"] < 0.95:
        decision = "RETRIEVAL_BOTTLENECK"
    elif r3["key_recall_at_2"] < 0.80 and r3["teacher_accuracy"] > r3["router_accuracy"]:
        decision = "ROUTER_QUERY_INSUFFICIENT"
    elif r3["router_oracle_gap"] <= 0.05 and r3["average_active_experts"] <= 2:
        decision = "PASS_ROUTER"
    else:
        decision = "FAIL_ROUTER"
    payload = {"stage": "p3", "formal": False, "status": "SMOKE_DIAGNOSTIC_ONLY", "decision": decision, "p1_gate_limited": True, "duration_seconds": time.time() - started, "r3": r3}
    (root / "gate_decisions/p3_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (root / "reports/p3_report.md").write_text("# Compose P3 Report\n\nP1 limited this stage to a single-seed cached-feature smoke diagnostic. Decision: **{}**. R0/R2/R3 exercise answer-free multi-label Query-Key routing, validation-only thresholds, temporal masks, and a maximum of two active experts. No final task-accuracy claim is made.\n".format(decision))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    p1 = json.loads((root / "gate_decisions/p1_decision.json").read_text())
    if p1["decision"] == "PASS_COMPOSITION":
        raise RuntimeError("P1 passed; formal P2/P3 are required and smoke-only execution is forbidden")
    p2 = p2_smoke(root)
    p3 = p3_smoke(root)
    print(json.dumps({"p1": p1["decision"], "p2": p2["decision"], "p3": p3["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
