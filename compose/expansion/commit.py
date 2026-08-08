"""Direct expert commit.

Once a cluster LoRA is trained and its key learned, the expert joins the
registry immediately. There is no contribution validation, no key-recall
commit threshold, no validation-accuracy gate, no positive-Jaccard
rejection, no redundancy rejection, and no provisional/formal promotion:
every valid cluster expert commits.

The commit itself stays crash-safe and idempotent via
``CommitTransaction``: one atomic registry update + exactly one
pool_version bump per commit transaction; resume never re-commits and
never reuses an expert id.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from compose.experts.metadata import (
    ExpertLifecycleStatus,
    ExpertMetadata,
    ExpertStatus,
)
from compose.experts.transaction import CommitTransaction

from .expert_formation import FormedExpert


def build_cluster_expert_metadata(
    formed: FormedExpert,
    task_id: int,
    task_name: str,
    seed: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
    key_path: str,
    key_sha256: str,
    rms_stats_path: Optional[str],
    config_hash: str,
    pool_version: int,
) -> ExpertMetadata:
    """Metadata for a directly committed cluster expert.

    The lifecycle is recorded as FORMAL for checkpoint compatibility, but
    the formal pool never depends on it: the registry's active pool is
    defined by the ``active`` flag and non-archived status.
    """
    return ExpertMetadata(
        expert_id=formed.expert_id,
        adapter_name="expert_{:04d}".format(formed.expert_id),
        rank=8,
        alpha=16.0,
        status=ExpertStatus.FROZEN,
        creation_task=int(task_id),
        creation_task_name=str(task_name),
        created_seed=int(seed),
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=str(checkpoint_sha256),
        key_path=str(key_path),
        key_sha256=str(key_sha256),
        rms_stats_path=str(rms_stats_path) if rms_stats_path else None,
        lifecycle_status=ExpertLifecycleStatus.FORMAL,
        config_hash=str(config_hash),
        pool_version_created=int(pool_version),
        active=True,
        trainable=False,
        extra={
            "cluster_id": formed.cluster_id,
            "key_mode": formed.key_mode,
            "cluster_size": formed.size,
            "cluster_sample_ids": list(formed.sample_ids),
            "commit_rule": "direct_cluster_commit",
        },
    )


def commit_cluster_expert(
    transaction: CommitTransaction,
    formed: FormedExpert,
    metadata: ExpertMetadata,
    artifacts: Dict[str, str],
    condition_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Directly commit one cluster expert (no validation gate).

    ``artifacts`` maps artifact paths (LoRA checkpoint, key, ...) to their
    expected sha256. Returns the transaction result; idempotent under
    crash recovery (see ``CommitTransaction``).
    """
    if metadata.expert_id != formed.expert_id:
        raise ValueError("metadata expert_id disagrees with the formed expert")
    transaction.begin(
        formed.expert_id,
        {"task_id": formed.creation_task, "cluster_id": formed.cluster_id},
    )
    return transaction.complete(
        formed.expert_id,
        artifacts=artifacts,
        condition_record=condition_record or {"rule": "direct_cluster_commit"},
        metadata=metadata,
    )
