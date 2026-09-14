"""Content-addressed cache of per-sample evaluation evidence.

Why
---
One S5 (remove-and-reroute) pruning trajectory asks the same question over and
over: *given this expert set for this validation sample, what is its answer NLL
and what does it generate?*  The trajectory removes one expert at a time, so a
sample whose route the removal did not touch has literally the same evidence in
the next job as it had in the previous one -- and re-running the LLM for it is
pure duplicated work.

This module removes that duplication and nothing else.  It never approximates,
interpolates, rounds or substitutes a proxy metric: a cache hit returns the
exact payload a previous run computed, and a miss runs the real evaluation and
stores its real result.  Evidence that has not been computed once is never
invented.

The fingerprint
---------------
Every key embeds a *fingerprint* of the evaluation context -- checkpoint
weights, checkpoint manifest, question file, runtime contract, model paths and
the scoring knobs that change the answer.  The fingerprint is a directory
component, so a cache populated for checkpoint A is structurally invisible to a
run over checkpoint B: a mismatch cannot produce a hit, it produces a miss.
That is stricter than refusing reuse, because it cannot be defeated by a
stale lookup path.

Concurrency
-----------
S5 jobs run several processes on separate GPUs against one cache directory.
Writes are atomic (unique temp file + ``os.replace``) and keys are immutable,
so a concurrent writer can never expose a partial payload.  Readers keep a
per-process memo so a key is read from disk at most once per job.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

#: Bumped whenever the meaning of a stored payload changes.  A bump makes every
#: previously written entry unreachable, which is the intended failure mode.
CACHE_SCHEMA_VERSION = 1

#: Relative-path prefix for the fingerprint descriptor, for auditability.
DESCRIPTOR_NAME = "fingerprint.json"


def sha256_file(path: str) -> str:
    """Streaming sha256, so a multi-GB checkpoint does not enter memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fingerprint(entries: Mapping[str, object]) -> str:
    """Stable digest of everything that changes what an evaluation produces."""
    payload = json.dumps(
        {"schema_version": CACHE_SCHEMA_VERSION, **entries},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scoring_fingerprint(
    *,
    checkpoint_dir: str,
    question_file: str,
    model_path: str,
    vision_tower: str,
    projector_path: str,
    runtime_contract: Optional[str],
    scoring: Mapping[str, object],
) -> Dict[str, object]:
    """Fingerprint inputs shared by ``nll_eval`` and ``eval_task``.

    Hashing ``compose_experts.bin`` costs a couple of seconds per job and is
    what makes the checkpoint identity exact rather than nominal: a
    re-exported checkpoint with the same path cannot alias a cached answer.
    """
    checkpoint = Path(checkpoint_dir)
    entries: Dict[str, object] = {
        "checkpoint_weights_sha256": sha256_file(str(checkpoint / "compose_experts.bin")),
        "checkpoint_manifest_sha256": sha256_file(str(checkpoint / "compose_experts.json")),
        "question_file_sha256": sha256_file(question_file),
        "model_path": model_path,
        "vision_tower": vision_tower,
        "projector_path": projector_path,
        "runtime_contract_sha256": (
            sha256_file(runtime_contract) if runtime_contract else None
        ),
        "scoring": dict(scoring),
    }
    return entries


class EvidenceCache:
    """Per-sample evidence store keyed by a fingerprint of the context.

    ``enabled=False`` yields a cache that reports every lookup as a miss and
    discards every write, so the same code path can be run with and without
    reuse without a second branch in the caller.
    """

    def __init__(
        self,
        root: Optional[str],
        fingerprint_entries: Union[Mapping[str, object], Callable[[], Mapping[str, object]]],
        enabled: bool = True,
    ) -> None:
        # ``fingerprint_entries`` may be a callable because computing it reads
        # and hashes the checkpoint: a disabled cache must not pay that cost,
        # so the entries are only materialised once we know the cache is on.
        self.enabled = bool(enabled and root)
        self._memo: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        if not self.enabled:
            self.root = None
            self.fingerprint = None
            return
        entries = fingerprint_entries() if callable(fingerprint_entries) else fingerprint_entries
        self.fingerprint = build_fingerprint(entries)
        self.root = Path(root) / self.fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = self.root / DESCRIPTOR_NAME
        if not descriptor.is_file():
            # Written once per fingerprint; a reader can audit exactly which
            # context produced every entry beneath it.
            self._atomic_write(descriptor, dict(entries))

    # -- keys -------------------------------------------------------------

    @staticmethod
    def make_key(*parts: object) -> str:
        """Key over the request identity, and only over the request identity."""
        payload = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- storage ----------------------------------------------------------

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / "{}.json".format(key)

    @staticmethod
    def _atomic_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".{}-".format(path.name),
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        if key in self._memo:
            self.hits += 1
            return self._memo[key]
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A torn or hand-edited entry is treated as absent: recompute.
            self.misses += 1
            return None
        self._memo[key] = payload
        self.hits += 1
        return payload

    def put(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        self._memo[key] = payload
        self._atomic_write(self._path(key), payload)

    # -- reporting --------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "requests": total,
            "hit_rate": (self.hits / total) if total else 0.0,
        }


def evidence_key(sample_id: str, expert_ids: Sequence[int]) -> str:
    """Key for one (sample, expert-set) evaluation request.

    ``expert_ids`` is normalised to a sorted tuple of ints so that two requests
    that route a sample through the same *set* of experts share evidence --
    the expert-set semantics used by ``ComposeSelection`` is a set, not a
    sequence, so this is an identity, not a relaxation.
    """
    return EvidenceCache.make_key("sample", str(sample_id), sorted(int(v) for v in expert_ids))


def split_by_cache(
    cache: EvidenceCache, sample_id: str, requests: Iterable[Tuple[str, Sequence[int]]]
) -> Tuple[Dict[str, Any], List[Tuple[str, Sequence[int]]]]:
    """Partition requests into ``(hits, misses)`` for one sample."""
    hits: Dict[str, Any] = {}
    misses: List[Tuple[str, Sequence[int]]] = []
    for set_key, expert_ids in requests:
        payload = cache.get(evidence_key(sample_id, expert_ids))
        if payload is None:
            misses.append((set_key, expert_ids))
        else:
            hits[set_key] = payload
    return hits, misses
