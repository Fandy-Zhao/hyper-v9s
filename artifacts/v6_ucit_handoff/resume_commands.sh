#!/bin/bash
# V6 UCIT 恢复命令（幂等；中断后直接重跑同一命令）
set -Eeuo pipefail
PYTHON=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
RUN_ROOT=$REPO/experiments/runs/v6_ucit_engineering

echo "== 1. 检查断点（pending 事务 / 恢复节点） =="
$PYTHON - <<'PYEOF'
import json
from pathlib import Path
from compose.experiments.v6_snapshot import V6Snapshot, analyze_resume
from compose.experts.transaction import CommitTransaction
for task_dir in sorted(Path("$RUN_ROOT/formal").glob("task*")) if Path("$RUN_ROOT/formal").exists() else []:
    snap_dir = task_dir / "snapshots"
    for snap in sorted(snap_dir.glob("task*")):
        try:
            result = analyze_resume(str(snap))
            print(snap, "->", result["resume_node"], "pending:", result["pending_transactions"])
        except Exception as exc:
            print(snap, "-> load error:", exc)
    # pending 事务清理（幂等）
    registry, incomplete, completed = CommitTransaction.resume(str(task_dir / "state"))
    print(task_dir, "incomplete:", list(incomplete), "cleaned:", list(completed))
PYEOF

echo "== 2. 从断点恢复（直接重跑；已完成阶段自动跳过） =="
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task1_dry_run \
  --output-root $RUN_ROOT/formal/task0 \
  --gpus 4,5,6,7 --master-port 29681 \
  --config $REPO/configs/v6_ucit_formal_locked.yaml \
  > $RUN_ROOT/formal/task0_runner.log 2>&1 &

echo "== 3. 恢复后验证 =="
echo "   - 不重复提交：registry 中已提交专家 id 不重复（CommitTransaction 拒绝复用）"
echo "   - 不重复 pool_version：registry.pool_version 恢复原值，事务幂等"
echo "   - 旧专家 hash 不变："
$PYTHON - <<'PYEOF'
from compose.experiments.v6_snapshot import V6Snapshot
snap = V6Snapshot.load("$RUN_ROOT/formal/task0/snapshots/task0")
expected = {e.expert_id: e.checkpoint_sha256 for e in snap.registry.list_all()}
print("   mismatches:", snap.verify_expert_hashes_unchanged(expected))
PYEOF
