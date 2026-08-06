#!/bin/bash
# V6 UCIT 六任务正式运行恢复（幂等；中断后直接重跑）
# 依据：configs/v6_ucit_formal_locked.yaml（formal locked config v1）
set -Eeuo pipefail

SEED="${1:-42}"
GPUS="${2:-4,5,6,7}"
PYTHON=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
RUN_ROOT=$REPO/experiments/runs/v6_ucit_engineering/formal/seed_$SEED

echo "== 1. 检查断点（pending 事务 / 恢复节点） =="
$PYTHON - <<PYEOF
import json
from pathlib import Path
from compose.experiments.v6_snapshot import V6Snapshot, analyze_resume
from compose.experts.transaction import CommitTransaction

root = Path("$RUN_ROOT")
for task_dir in sorted(root.glob("task*")):
    snap_dir = task_dir / "snapshots"
    for snap in sorted(snap_dir.glob("task*")):
        try:
            result = analyze_resume(str(snap))
            print(snap, "->", result["resume_node"], "pending:", result["pending_transactions"])
        except Exception as exc:
            print(snap, "-> load error:", exc)
    state_dir = task_dir / "state"
    if state_dir.is_dir():
        registry, incomplete, completed = CommitTransaction.resume(str(state_dir))
        print(task_dir, "incomplete:", list(incomplete), "cleaned:", list(completed))
PYEOF

echo "== 2. 从断点恢复（六任务编排；已完成阶段自动跳过） =="
TRAIN_GPU="${TRAIN_GPU:-4}" bash "$REPO/scripts/v6_ucit/six_task_run.sh" "$SEED" "$GPUS"

echo "== 3. 恢复后验证 =="
echo "   - 不重复提交：registry 中已提交专家 id 不重复（CommitTransaction 拒绝复用）"
echo "   - 不重复 pool_version：registry.pool_version 恢复原值，事务幂等"
echo "   - 旧专家 hash 不变："
$PYTHON - <<PYEOF
import sys
sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA")
from pathlib import Path
from compose.experiments.v6_snapshot import V6Snapshot

root = Path("$RUN_ROOT")
for task_dir in sorted(root.glob("task*")):
    snap_dir = task_dir / "snapshots"
    snaps = sorted(snap_dir.glob("task*"))
    if not snaps:
        continue
    snap = V6Snapshot.load(str(snaps[-1]))
    expected = {e.expert_id: e.checkpoint_sha256 for e in snap.registry.list_all()}
    print(snaps[-1], "mismatches:", snap.verify_expert_hashes_unchanged(expected))
PYEOF
