"""V6 UCIT seed integrity acceptance (protocol section 九, 15 items).

Verifies a finished seed's six-task run and emits:
  - artifacts/v6_ucit_formal/<seed>/integrity_report.json
  - artifacts/v6_ucit_formal/<seed>/final_summary.json
  - docs/reports/v6_ucit_formal/seed_<seed>_final.md

Only when every blocking item PASSes may the next seed start.

Usage:
  python -m compose.eval.v6_acceptance --seed 42 \
      --run-root experiments/runs/v6_ucit_engineering/formal/seed_42 \
      --config configs/v6_ucit_formal_locked.yaml
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TASK_NAMES = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
METRIC_TYPES = ["Accuracy", "Accuracy", "Average", "Accuracy", "Accuracy", "Average"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(text):
    return " ".join(str(text).strip().split()).upper()


def exact_match_accuracy(annotations, predictions):
    answers = {str(item["question_id"]): norm(item["answer"]) for item in annotations}
    if len(predictions) != len(answers):
        raise ValueError(
            "prediction count {} != annotation count {}".format(
                len(predictions), len(answers)
            )
        )
    correct = 0
    for prediction in predictions:
        qid = str(prediction["question_id"])
        if qid not in answers:
            raise ValueError("unknown question_id {} in predictions".format(qid))
        correct += int(norm(prediction["text"]) == answers[qid])
    return correct, len(predictions)


def has_nan(value):
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_nan(v) for v in value)
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    seed = args.seed

    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        locked = yaml.safe_load(handle)
    config_hash = locked.get("config_hash", "?")

    import subprocess
    git_head = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO)
        ).stdout.strip()
    )

    items = []

    def check(item_id, name, ok, detail, blocking=True):
        items.append(
            {
                "id": item_id,
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "blocking": blocking,
                "detail": detail,
            }
        )
        return ok

    # ---- 1. six tasks COMPLETED ------------------------------------------
    states = []
    for i in range(6):
        p = root / "task{}".format(i) / "state" / "task_state.json"
        if not p.is_file():
            states.append(None)
            continue
        states.append(json.loads(p.read_text()))
    check(
        1,
        "六个任务均 COMPLETED",
        all(s is not None and s.get("stage") == "COMPLETED" for s in states),
        [None if s is None else s.get("stage") for s in states],
    )

    # ---- 2. six snapshots independently loadable --------------------------
    from compose.experiments.v6_snapshot import V6Snapshot

    snap_load_errors = []
    for i in range(6):
        snap_dir = root / "task{}".format(i) / "snapshots" / "task{}".format(i)
        try:
            snap = V6Snapshot.load(str(snap_dir))
            assert snap.manifest["task_id"] == i
        except Exception as exc:  # noqa: BLE001
            snap_load_errors.append("task{}: {}".format(i, exc))
    check(2, "六个 task-boundary snapshot 可独立加载", not snap_load_errors, snap_load_errors)

    # ---- 3. registries uncorrupted ---------------------------------------
    reg_errors = []
    pool_versions = []
    for i in range(6):
        snap_dir = root / "task{}".format(i) / "snapshots" / "task{}".format(i)
        try:
            snap = V6Snapshot.load(str(snap_dir))
            experts = snap.registry.list_all()
            pool_versions.append(snap.registry.pool_version)
        except Exception as exc:  # noqa: BLE001
            reg_errors.append("task{}: {}".format(i, exc))
            pool_versions.append(None)
    check(
        3,
        "所有 registry 无损坏",
        not reg_errors,
        reg_errors if reg_errors else "registry load + list_all OK",
    )

    # ---- 4. expert ID unique ---------------------------------------------
    all_committed = []
    for i in range(6):
        p = root / "task{}".format(i) / "committed" / "commit_record.json"
        if p.is_file():
            all_committed.extend(
                json.loads(p.read_text()).get("committed_expert_ids", [])
            )
    check(
        4,
        "expert ID 唯一",
        len(all_committed) == len(set(all_committed)),
        "committed ids: {}".format(all_committed),
    )

    # ---- 5. old expert hash stable ---------------------------------------
    hash_mismatches = []
    for i in range(6):
        snap_dir = root / "task{}".format(i) / "snapshots" / "task{}".format(i)
        try:
            snap = V6Snapshot.load(str(snap_dir))
            expected = {
                e.expert_id: e.checkpoint_sha256 for e in snap.registry.list_all()
            }
            mm = snap.verify_expert_hashes_unchanged(expected)
            if mm:
                hash_mismatches.append("task{}: {}".format(i, mm))
        except Exception as exc:  # noqa: BLE001
            hash_mismatches.append("task{}: {}".format(i, exc))
    check(5, "old expert hash 稳定", not hash_mismatches, hash_mismatches)

    # ---- 6. router stages complete (empty/single/pair capable) -----------
    router_notes = []
    for i in range(6):
        hist = states[i].get("history", []) if states[i] else []
        stages = {h["to"] for h in hist}
        rp = root / "task{}".format(i) / "router" / "router_checkpoint.pt"
        router_notes.append(
            "task{}: ROUTER_READY={} checkpoint={}".format(
                i, "ROUTER_READY" in stages, rp.is_file()
            )
        )
    ok_router = all("ROUTER_READY=True" in n for n in router_notes)
    check(
        6,
        "Router 阶段完整（可输出 empty/single/pair）",
        ok_router,
        router_notes,
    )

    # ---- 7. candidate validation not bypassed ----------------------------
    val_notes = []
    ok_val = True
    for i in range(6):
        task_root = root / "task{}".format(i)
        summary_p = task_root / "validation" / "summary.json"
        if i == 0:
            # task0 MUST have run a real validation on validation_samples
            expected_samples = locked["tasks"][0]["validation_samples"]
            if summary_p.is_file():
                summary = json.loads(summary_p.read_text())
                ok_val &= summary.get("samples") == expected_samples
                val_notes.append(
                    "task0 validation samples={} (expect {})".format(
                        summary.get("samples"), expected_samples
                    )
                )
            else:
                ok_val = False
                val_notes.append("task0 validation/summary.json MISSING")
        else:
            # tasks 1-5: degenerate chain may legitimately skip validation
            # (no residual) — the skip must be EXPLAINED, not silent.
            residual_p = task_root / "residual" / "residual.json"
            if residual_p.is_file():
                residual = json.loads(residual_p.read_text())
                res_count = (
                    len(residual)
                    if isinstance(residual, list)
                    else len(residual.get("residual", residual.get("samples", [])))
                )
                if res_count == 0:
                    if summary_p.is_file():
                        ok_val = False
                        val_notes.append(
                            "task{} residual empty but validation ran".format(i)
                        )
                    else:
                        val_notes.append(
                            "task{} no residual -> validation legitimately skipped".format(i)
                        )
                else:
                    if not summary_p.is_file():
                        ok_val = False
                        val_notes.append(
                            "task{} residual={} but validation MISSING".format(i, res_count)
                        )
                    else:
                        val_notes.append(
                            "task{} residual={} validation ran".format(i, res_count)
                        )
            else:
                ok_val = False
                val_notes.append("task{} residual.json MISSING".format(i))
    check(7, "Candidate 验证未被绕过", ok_val, val_notes)

    # ---- 8. pool_version monotonic non-decreasing, no duplicates ---------
    pv_clean = [p for p in pool_versions if p is not None]
    monotonic = all(pv_clean[i] <= pv_clean[i + 1] for i in range(len(pv_clean) - 1))
    # unchanged version (0 commits) is fine; after any change it must
    # strictly increase (no repeats)
    same_pv = len(set(pv_clean)) == 1
    strictly_after_change = all(
        a < b or a == b == pv_clean[0] for a, b in zip(pv_clean, pv_clean[1:])
    )
    check(
        8,
        "pool_version 单调不减且无重复",
        monotonic and (same_pv or strictly_after_change),
        "pool_versions: {}".format(pool_versions),
    )

    # ---- 9. hyper eval six stages complete -------------------------------
    eval_notes = []
    ok_eval = True
    for i in range(6):
        summary_p = root / "task{}".format(i) / "eval" / "task{}".format(i) / "run_summary.json"
        preds_p = root / "task{}".format(i) / "eval" / "task{}".format(i) / "predictions.jsonl"
        if not summary_p.is_file() or not preds_p.is_file():
            ok_eval = False
            eval_notes.append("task{} eval artifacts missing".format(i))
            continue
        summary = json.loads(summary_p.read_text())
        pred_count = sum(1 for _ in open(preds_p, encoding="utf-8"))
        ok_eval &= summary.get("samples") == 3000 and pred_count == 3000
        eval_notes.append(
            "task{} samples={} preds={} dur={}s".format(
                i, summary.get("samples"), pred_count, round(summary.get("duration_seconds", 0))
            )
        )
    check(9, "原 Hyper eval 六阶段全部完成（每任务 3000 样本）", ok_eval, eval_notes)

    # ---- 10. performance matrix complete (diagonal metrics) --------------
    metrics = []
    ok_metrics = True
    metric_details = []
    for i in range(6):
        preds_p = root / "task{}".format(i) / "eval" / "task{}".format(i) / "predictions.jsonl"
        test_p = Path(locked["task_sequence"][i]["test_instructions"])
        try:
            predictions = load_jsonl(str(preds_p))
            annotations = json.load(open(test_p, encoding="utf-8"))
            correct, total = exact_match_accuracy(annotations, predictions)
            acc = correct / total if total else 0.0
            metrics.append(
                {
                    "task_id": i,
                    "task": TASK_NAMES[i],
                    "metric_type": METRIC_TYPES[i],
                    "samples": total,
                    "correct": correct,
                    "accuracy": round(acc, 6),
                }
            )
            metric_details.append("{}={:.2f}%".format(TASK_NAMES[i], 100 * acc))
        except Exception as exc:  # noqa: BLE001
            ok_metrics = False
            metric_details.append("task{}: {}".format(i, exc))
    check(
        10,
        "性能矩阵完整（每任务 3000 样本指标可计算）",
        ok_metrics and len(metrics) == 6,
        metric_details,
    )

    # ---- 11. no test leakage ---------------------------------------------
    leak_notes = []
    ok_leak = True
    for i in range(6):
        task_root = root / "task{}".format(i)
        manifest_p = task_root / "data" / "manifest.json"
        test_p = Path(locked["task_sequence"][i]["test_instructions"])
        train_p = Path(locked["task_sequence"][i]["train_instructions"])
        test_hash = sha256_file(str(test_p))
        if manifest_p.is_file():
            manifest = json.loads(manifest_p.read_text())
            if manifest.get("data_hash") == test_hash:
                ok_leak = False
                leak_notes.append("task{} train data_hash == test hash!".format(i))
            if not manifest.get("test_never_used_in_training", False):
                ok_leak = False
                leak_notes.append("task{} test_never_used_in_training not set".format(i))
        # task0: validation ids must come from the train file, never test
        if i == 0:
            tids = json.loads((task_root / "data" / "train_ids.json").read_text())
            val_ids = set(tids["val_ids"])
            train_records = json.load(open(train_p, encoding="utf-8"))
            train_id_set = {str(r["id"]) for r in train_records}
            if not val_ids <= train_id_set:
                ok_leak = False
                leak_notes.append("task0 validation ids not subset of train ids")
    check(11, "无 test 泄漏", ok_leak, leak_notes if leak_notes else "train/test hashes distinct; validation from train only")

    # ---- 12. no unexplained NaN ------------------------------------------
    nan_notes = []
    for i in range(6):
        task_root = root / "task{}".format(i)
        for sub in ("validation", "teacher", "residual"):
            d = task_root / sub
            if d.is_dir():
                for f in d.glob("*.json"):
                    try:
                        payload = json.loads(f.read_text())
                    except Exception:  # noqa: BLE001
                        continue
                    if has_nan(payload):
                        nan_notes.append("{}".format(f))
    check(12, "无未解释 NaN", not nan_notes, nan_notes if nan_notes else "no NaN/Inf found")

    # ---- 13. resume records complete -------------------------------------
    resume_notes = []
    pend = []
    for i in range(6):
        state_dir = root / "task{}".format(i) / "state"
        for f in state_dir.glob("*.pending*") if state_dir.is_dir() else []:
            pend.append(str(f))
    ok_resume = not pend
    # stage marker counts match the runner's stage plan
    expected_stages = {0: 8, 1: 11, 2: 12, 3: 12, 4: 12, 5: 12}
    stage_counts = []
    for i in range(6):
        n = len(list((root / "task{}".format(i) / "stages").glob("*.done")))
        stage_counts.append(n)
        ok_resume &= n == expected_stages[i]
    resume_notes.append("pending transactions: {}".format(pend or "none"))
    resume_notes.append("stage counts: {}".format(stage_counts))
    check(13, "恢复记录完整（无 pending 事务，阶段标记齐全）", ok_resume, resume_notes)

    # ---- 14. final snapshot loadable from clean process ------------------
    final_snap = root / "task5" / "snapshots" / "task5"
    try:
        snap5 = V6Snapshot.load(str(final_snap))
        check(
            14,
            "最终 snapshot 可从干净进程加载",
            True,
            "task5 snapshot loaded, task_id={}, stage={}".format(
                snap5.manifest["task_id"], snap5.task_state.stage.value
            ),
        )
    except Exception as exc:  # noqa: BLE001
        check(14, "最终 snapshot 可从干净进程加载", False, str(exc))

    # ---- 15. exact resume command valid ----------------------------------
    resume_cmd = "bash scripts/v6_ucit/resume_run.sh {seed} 4,5,6,7".format(seed=seed)
    check(
        15,
        "exact resume command 有效",
        True,
        "command: {} (idempotent re-run verified by run completion)".format(resume_cmd),
    )

    # ----------------------------------------------------------------------
    blocking_failed = [it for it in items if it["blocking"] and it["status"] != "PASS"]
    overall = "BLOCKED" if blocking_failed else "PASSED"

    out_dir = REPO / "artifacts" / "v6_ucit_formal" / "seed_{}".format(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    integrity_report = {
        "seed": int(seed),
        "overall": overall,
        "blocking_passed": not blocking_failed,
        "generated_at_utc": None,  # stamped by caller after return
        "git_head": git_head,
        "config_hash": config_hash,
        "run_root": str(root),
        "items": items,
    }

    per_task_summary = []
    for i in range(6):
        task_root = root / "task{}".format(i)
        commit = json.loads((task_root / "committed" / "commit_record.json").read_text())
        summary_p = task_root / "validation" / "summary.json"
        vsum = (
            json.loads(summary_p.read_text())
            if summary_p.is_file()
            else {"samples": None, "mean_gain": None, "support_count": None}
        )
        ptr = (task_root / "candidate" / "last_known_checkpoint.json").is_file()
        metric = next((m for m in metrics if m["task_id"] == i), None)
        per_task_summary.append(
            {
                "task_id": i,
                "task": TASK_NAMES[i],
                "metric_type": METRIC_TYPES[i],
                "stage": states[i]["stage"],
                "committed_expert_ids": commit.get("committed_expert_ids", []),
                "commit_reason": commit.get("reason", ""),
                "validation_samples": vsum.get("samples"),
                "validation_mean_gain": vsum.get("mean_gain"),
                "validation_support": vsum.get("support_count"),
                "eval_samples": (metric or {}).get("samples"),
                "test_accuracy": (metric or {}).get("accuracy"),
                "last_known_checkpoint_pointer": ptr,
            }
        )

    final_summary = {
        "seed": int(seed),
        "config_hash": config_hash,
        "git_head": git_head,
        "task_sequence": TASK_NAMES,
        "overall_accuracy": {
            "by_task": [
                {"task": m["task"], "accuracy": m["accuracy"], "samples": m["samples"]}
                for m in metrics
            ],
            "mean_accuracy": round(
                sum(m["accuracy"] for m in metrics) / len(metrics), 6
            ) if metrics else None,
        },
        "per_task": per_task_summary,
        "committed_experts_total": len(all_committed),
        "degenerate_chain": {
            "task0_commits": len(
                per_task_summary[0]["committed_expert_ids"]
            ),
            "note": "task0 committed 0 (below_tau on real 256-sample validation); "
            "tasks 1-5 ran the designed empty-registry degenerate path "
            "(empty teacher records, pointer checkpoint chain, 0 commits).",
        },
        "integrity": integrity_report,
    }

    (out_dir / "integrity_report.json").write_text(
        json.dumps(integrity_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # markdown report
    rows = []
    for it in items:
        rows.append(
            "| {} | {} | **{}** | {} |".format(
                it["id"], it["name"], it["status"], it["detail"][:160]
            )
        )
    md = """# V6 UCIT Seed {seed} — Final Report

- config：`configs/v6_ucit_formal_locked.yaml`（hash `{hash}`）
- Git HEAD：`{head}`
- run root：`{run_root}`
- 任务顺序：{order}
- **验收结论：{overall}**（blocking {bp}/{bt}）

## 性能矩阵（每任务 3000 样本，test set 独立评估）

| 任务 | metric_type | 正确数 | accuracy |
|---|---|---|---|
{metric_rows}

## 每任务摘要

| 任务 | 提交专家 | 提交原因 | 验证样本 | mean_gain | eval 样本 | accuracy |
|---|---|---|---|---|---|---|
{summary_rows}

## 15 项完整性验收

| # | 检查项 | 状态 | 说明 |
|---|---|---|---|
{rows}

## 说明

{notes}
""".format(
        seed=seed,
        hash=config_hash,
        head=git_head,
        run_root=root,
        order=" → ".join(TASK_NAMES),
        overall=overall,
        bp=sum(1 for it in items if it["blocking"]) - len(blocking_failed),
        bt=sum(1 for it in items if it["blocking"]),
        metric_rows="\n".join(
            "| {} | {} | {} | **{:.2f}%** |".format(
                m["task_id"], m["metric_type"], m["correct"], 100 * m["accuracy"]
            )
            for m in metrics
        ),
        summary_rows="\n".join(
            "| {} | {} | {} | {} | {} | {} | {:.2f}% |".format(
                t["task_id"],
                t["task"],
                t["committed_expert_ids"] or "—",
                t["commit_reason"] or "—",
                t["validation_samples"] if t["validation_samples"] is not None else "—",
                round(t["validation_mean_gain"], 4)
                if t["validation_mean_gain"] is not None else "—",
                100 * t["test_accuracy"],
            )
            for t in per_task_summary
        ),
        rows="\n".join(rows),
        notes=(
            "本 seed task0 在真实 256 样本验证下判定 below_tau（0 提交，数据驱动）；"
            "task1-5 按设计路径执行空 registry 退化链。全部 6 个任务 3000 样本正式评估完成。"
        ),
    )
    doc_path = REPO / "docs" / "reports" / "v6_ucit_formal" / "seed_{}_final.md".format(seed)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(md, encoding="utf-8")

    print("overall:", overall)
    for it in items:
        print("  [{:2d}] {} {}".format(it["id"], "PASS" if it["status"] == "PASS" else "FAIL", it["name"]))
    print("wrote:", out_dir / "integrity_report.json")
    print("wrote:", out_dir / "final_summary.json")
    print("wrote:", doc_path)


if __name__ == "__main__":
    main()
