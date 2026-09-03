"""S1 adapter: emit the legacy ``features/<split>.json`` payload from the
precomputed fixed-query cache (0903 spec §5-8).

The V7 run pipeline's S1 stage used to run the frozen CLIP encoder live
over every train/val record.  With the full precomputed cache available,
this module replaces that encoder run: it reads the binary split cache
through the unified reader (``compose.v7.query_cache``), validates the
manifest's runtime contract against the *current* process (git head,
backbone content hash, fixed-query implementation hash), and writes a
payload whose schema is byte-compatible with the legacy S1 output so
every downstream consumer (``validate_query_cache_contract``,
``queries_from_cache``, ``V7QueryDataset``, RMS, pruning) is untouched.

Deliberate differences from the legacy payload (documented in the
adaptation report):

- ``records[sample_id]`` carries ``query`` only; the ``visual_feature`` /
  ``text_feature`` lists are *not* stored in the binary cache and no V7
  consumer reads them (verified by audit).  The matching legacy
  ``feature_hash`` field is therefore omitted;
- ``query_hash`` keeps the exact legacy semantics (canonical json sha256
  over ``{sample_id: query}`` sorted by sample id) and is computed over
  the cache rows in a memory-bounded stream;
- an extra top-level ``query_origin`` block records where the payload
  came from (manifest sha256, split contract hash, cache kind) so every
  consumer can verify the payload is cache-derived, not encoder-derived.

No GPU, no CLIP, no encoder calls are ever involved here.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from . import query_cache as qc
from .query import FixedQueryProvenance

PAYLOAD_ORIGIN_KIND = "v7_fixed_query_cache_derived"


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, check=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).stdout.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_head() -> Optional[str]:
    return _git(["rev-parse", "HEAD"])


def git_branch() -> Optional[str]:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"])


def _query_list_text(query_row) -> str:
    """Compact json text of the bare ``[..1536 floats..]`` query list.

    Exactly what ``json.dumps(list, separators=(",", ":"))`` produces, so
    the recomputed ``query_hash`` is byte-equivalent to the legacy
    ``records_query_hash`` (which hashes ``record["query"]`` - the list,
    not the per-record dict).
    """
    return json.dumps(query_row.tolist(), separators=(",", ":"))


def query_hash_of_rows(reader: qc.V7QuerySplitReader) -> str:
    """Legacy-semantics query hash over cache rows (streaming, bounded RAM).

    sha256 over ``json.dumps({sid: query_list}, sort_keys=True,
    separators=(",", ":"))`` with the digest accumulated per record
    instead of materializing the full mapping.
    """
    order = sorted(range(reader.n), key=lambda index: reader.sample_ids[index])
    digest = hashlib.sha256()
    digest.update(b"{")
    for position, index in enumerate(order):
        if position:
            digest.update(b",")
        sample_id = reader.sample_ids[index]
        digest.update(json.dumps(sample_id).encode("utf-8"))
        digest.update(b":")
        digest.update(_query_list_text(reader.queries[index]).encode("utf-8"))
    digest.update(b"}")
    return digest.hexdigest()


def write_split_payload_from_cache(
    manifest: qc.V7CacheManifest,
    task_index: int,
    split: str,
    declared_json_path: str,
    output_path: str,
    backbone_provenance: Mapping[str, object],
    *,
    encoder_provenance: Optional[Mapping[str, object]] = None,
    encoder_hash: Optional[str] = None,
    stream_chunk: int = 4096,
) -> Dict[str, object]:
    """Write one cache-derived ``features/<split>.json`` payload.

    ``declared_json_path`` is the split file the run actually trains or
    evaluates on (S0's ``train_full.json``/``val_full.json``).  Its sample
    id sequence must equal the cache's stored sequence exactly - any
    missing, extra, or reordered id fails closed (the sample id is the
    primary key; a mismatch would silently route wrong samples otherwise).
    """
    declared = json.loads(Path(declared_json_path).read_text(encoding="utf-8"))
    if not isinstance(declared, list):
        raise ValueError("declared split is not a record list: {}".format(declared_json_path))
    declared_ids = [
        str(record.get("id", record.get("question_id"))) for record in declared
    ]
    reader = manifest.reader(task_index, split)
    if declared_ids != list(reader.sample_ids):
        raise ValueError(
            "split {} declared-id sequence does not match the cache for task{} "
            "(declared {} != cached {}): refusing to emit a payload".format(
                split, task_index, len(declared_ids), reader.n
            )
        )
    impl = FixedQueryProvenance()
    payload_header = {
        "schema_version": qc.QUERY_SCHEMA_VERSION,
        "feature_source": "frozen_clip_l14_336",
        "query_backbone_provenance": dict(backbone_provenance),
        "query_mode": qc.QUERY_MODE,
        "query_encoder_provenance": dict(
            encoder_provenance if encoder_provenance is not None else impl.to_dict()
        ),
        "query_encoder_hash": (
            encoder_hash if encoder_hash is not None else impl.module_hash
        ),
        "query_hash": query_hash_of_rows(reader),
        "query_origin": {
            "kind": PAYLOAD_ORIGIN_KIND,
            "cache_kind": qc.CACHE_KIND,
            "task_index": int(task_index),
            "split": split,
            "manifest_sha256": manifest.manifest_sha256(),
            "manifest_path": manifest.path,
            "split_contract_hash": manifest.split_contract_hash(task_index, split),
            "query_dim": qc.QUERY_DIM,
            "dtype": qc.QUERY_DTYPE,
            "encoder_calls": 0,
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    written_rows = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("{")
            first_key = True
            for key, value in payload_header.items():
                if not first_key:
                    handle.write(",")
                first_key = False
                handle.write(json.dumps(key))
                handle.write(":")
                handle.write(json.dumps(value, separators=(",", ":")))
            handle.write(',"records":{')
            queries = reader.queries
            for offset in range(0, reader.n, stream_chunk):
                stop = min(offset + stream_chunk, reader.n)
                rows = queries[offset:stop].detach().cpu()
                for index in range(offset, stop):
                    if index:
                        handle.write(",")
                    sample_id = reader.sample_ids[index]
                    handle.write(json.dumps(sample_id))
                    handle.write(':{"query":')
                    handle.write(_query_list_text(rows[index - offset]))
                    handle.write("}")
                    written_rows += 1
            handle.write("}}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    if written_rows != reader.n:
        raise ValueError(
            "payload write for task{}.{} wrote {} rows, expected {}".format(
                task_index, split, written_rows, reader.n
            )
        )
    return {
        "task_index": int(task_index),
        "split": split,
        "declared_count": len(declared_ids),
        "cached_count": reader.n,
        "sequence_matches_cache": True,
        "query_hash": payload_header["query_hash"],
        "manifest_sha256": payload_header["query_origin"]["manifest_sha256"],
        "split_contract_hash": payload_header["query_origin"]["split_contract_hash"],
        "encoder_calls": 0,
        "output": str(target.resolve()),
        "output_size_bytes": target.stat().st_size,
    }


def emit_s1_payloads_from_cache(
    manifest_path: str,
    task_index: int,
    backbone_path: str,
    declared_files: Mapping[str, str],
    output_files: Mapping[str, str],
    *,
    backbone_name: str = "clip-vit-large-patch14-336",
    backbone_provenance: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Full S1-from-cache entry point used by ``v7_task_run``.

    Validates the manifest's *content* contract (live backbone content
    hash, fixed-query implementation hash, schema/dim/dtype, declared
    counts and the per-split manifest/metadata contract-hash chain - fail
    closed) and then writes one cache-derived payload per declared split.

    The manifest's recorded producer git (the commit the cache was built
    at) and the *current* runtime git are both returned in the audit for
    the formal report: code changes made after cache construction (S1
    adapters, scheduling) never alter query content, whose invariants are
    the backbone hash + query implementation hash + source data - those
    are exactly what is validated above.  No encoder call ever happens
    (``encoder_calls`` is 0 by construction).

    ``backbone_provenance`` may be injected (tests); otherwise it is
    computed live from ``backbone_path`` via the lazy CLIP-free path.
    """
    if backbone_provenance is None:
        # Imported lazily: compose.eval.query_features pulls in the CLIP
        # stack at import time; the pure adapters above stay lightweight.
        from compose.eval.query_features import query_backbone_provenance

        backbone_provenance = query_backbone_provenance(backbone_path)

    manifest = qc.V7CacheManifest.locate(manifest_path)
    contract = manifest.validate_runtime_contract(
        backbone_hash=backbone_provenance["backbone_hash"],
        impl_hash=FixedQueryProvenance().module_hash,
        required=tuple(declared_files),
    )
    runtime_git = git_head()
    writes = []
    for split, declared_path in declared_files.items():
        audit = write_split_payload_from_cache(
            manifest,
            task_index,
            split,
            declared_path,
            output_files[split],
            backbone_provenance,
        )
        writes.append(audit)
    return {
        "source": PAYLOAD_ORIGIN_KIND,
        "manifest_path": manifest.path,
        "manifest_sha256": manifest.manifest_sha256(),
        "query_mode": qc.QUERY_MODE,
        "backbone_name": backbone_name,
        "backbone_hash": backbone_provenance["backbone_hash"],
        "impl_hash": FixedQueryProvenance().module_hash,
        # provenance pair: producer git (immutable, from the manifest) vs
        # the git this process runs at; drift is reported, never silent.
        "producer_git_sha": manifest.payload.get("git_sha"),
        "producer_git_branch": manifest.payload.get("git_branch"),
        "runtime_git_sha": runtime_git,
        "runtime_git_branch": git_branch(),
        "runtime_contract": contract,
        "splits": writes,
        "encoder_calls": 0,
    }
