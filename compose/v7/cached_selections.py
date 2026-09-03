"""Test-time routing selections from the fixed-query cache (0903 spec §21).

Final evaluation (per-task S6 and the 21-cell matrix) used to run the
frozen CLIP encoder live per record and route through
``V7InferenceRouter``.  With the precomputed cache, every q_i for a
question split is already stored: this module reproduces the *exact*
committed-only Global Top-2 route by calling the very same router object
over cache rows, and writes the per-sample selection manifest that
``eval_task --selection-manifest`` consumes (so the generator loads no
CLIP model and makes zero query-encoder calls).

Routing happens on the caller's device with fp32 rows, matching the live
eval path bit-for-bit (same pool keys, same query values, same router,
same op order on the same device).

The manifest maps ``str(sample_id_of(record))`` -> ``{"global_top2":
[e1, e2]}``, exactly the schema ``route_manifest`` writes for pruning and
exactly the key ``eval_task`` looks up per record.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch

from . import query_cache as qc
from .inference import V7InferenceRouter
from .pool import V7ExpertKeyPool
from .query import FixedQueryProvenance
from .workflow import route_manifest


def write_cached_selections(
    cache_manifest_path: str,
    key_state_path: str,
    questions_path: str,
    question_task_index: int,
    output_path: str,
    *,
    model_task_index: int = -1,
    device: str = "cuda:0",
    backbone_hash: Optional[str] = None,
    impl_hash: Optional[str] = None,
) -> Dict[str, object]:
    """Write a committed-pool selection manifest from cache rows.

    ``key_state_path`` is a committed ``v7_keys.pt`` (task root
    ``committed/``), whose pool routes the question split.  The question
    split itself lives in the cache under ``question_task_index`` (for S6
    that equals the model task; for a lower-triangle cell it is the
    *question* task, which may differ from the model task).

    The question file's sample-id sequence must equal the cache split's
    sequence exactly (fail closed), and the returned audit records the
    routing source, counts and the manifest file hash.

    ``backbone_hash``/``impl_hash`` (when given) bind every task's test
    split contract to the live frozen backbone / fixed-query encoder, so a
    cache whose queries came from a different backbone is rejected before
    any routing happens.
    """
    manifest = qc.V7CacheManifest.locate(cache_manifest_path)
    manifest.validate_runtime_contract(
        required=("test",), backbone_hash=backbone_hash, impl_hash=impl_hash
    )
    records = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    sample_ids = [qc.sample_id_of(record) for record in records]
    reader = manifest.reader(question_task_index, "test")
    if sample_ids != list(reader.sample_ids):
        raise ValueError(
            "question file for task{} does not match the cache test split "
            "(declared {} != cached {})".format(
                question_task_index, len(sample_ids), reader.n
            )
        )
    state = torch.load(key_state_path, map_location="cpu", weights_only=False)
    pool = V7ExpertKeyPool.from_state(state)
    if model_task_index < 0:
        origins = {int(value["origin_task"]) for value in pool.metadata.values()}
        model_task_index = int(sorted(origins)[-1]) if origins else -1
    router = V7InferenceRouter(pool)
    queries = reader.get_batch(sample_ids, device=device)
    with torch.inference_mode():
        result = router.router(queries)
    rows = result.expert_ids.detach().cpu()
    if rows.shape != (reader.n, 2):
        raise ValueError("Global Top-2 routing must return exactly two experts per row")
    selected = route_manifest(sample_ids, rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    selected, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    histogram: Dict[str, int] = {}
    for row in rows.tolist():
        signature = tuple(sorted(int(value) for value in row))
        histogram[str(signature)] = histogram.get(str(signature), 0) + 1
    return {
        "source": "v7_fixed_query_cache_selections",
        "question_task_index": int(question_task_index),
        "model_task_index": int(model_task_index),
        "questions_path": str(Path(questions_path).resolve()),
        "key_state_path": str(Path(key_state_path).resolve()),
        "count": len(sample_ids),
        "sequence_matches_cache": True,
        "visible_expert_ids": [int(value) for value in pool.selectable_ids()],
        "selected_histogram": histogram,
        "selection_manifest": str(target.resolve()),
        "encoder_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--key-state", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--question-task-index", type=int, required=True)
    parser.add_argument("--model-task-index", type=int, default=-1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backbone-path",
        help="frozen CLIP directory; when given, every task's test-split "
        "contract is bound to its live content hash (fail closed)",
    )
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    backbone_hash = None
    if args.backbone_path:
        # Imported lazily: compose.eval.query_features pulls in the CLIP
        # stack at import time; the pure routing path stays lightweight.
        from compose.eval.query_features import query_backbone_provenance

        backbone_hash = query_backbone_provenance(args.backbone_path)["backbone_hash"]
    audit = write_cached_selections(
        args.cache_manifest, args.key_state, args.questions,
        args.question_task_index, args.output,
        model_task_index=args.model_task_index, device=args.device,
        backbone_hash=backbone_hash,
        impl_hash=FixedQueryProvenance().module_hash,
    )
    print(
        "cached selections written to {} ({} samples, {} selectable experts, "
        "encoder_calls=0)".format(
            args.output, audit["count"], len(audit["visible_expert_ids"])
        )
    )
    if args.audit_output:
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_output).write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
