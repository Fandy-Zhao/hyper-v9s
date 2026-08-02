"""Atomic, sharded, resumable train-only Residual Buffer."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


BUFFER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResidualRecord:
    sample_id: str
    task_id: int
    task_name: str
    split: str
    data_reference: str
    data_manifest_hash: str
    image_reference: str
    prompt_reference: str
    answer_reference: str
    query_feature_hash: str
    predicted_set: Tuple[int, ...]
    teacher_set: Tuple[int, ...]
    teacher_sufficiency: bool
    predicted_sufficiency: float
    empty_nll: float
    selected_historical_nll: float
    residual_gain: float
    router_confidence: float
    creation_timestamp: str
    source_git_commit: str
    config_hash: str
    answer_type: str = "unknown"
    question_subtype: str = "unknown"

    def __post_init__(self):
        if self.split != "train":
            raise ValueError("Residual Buffer only accepts train")
        if not 0.0 <= self.predicted_sufficiency <= 1.0:
            raise ValueError("predicted_sufficiency must be a probability")
        if not self.sample_id or not self.config_hash or not self.source_git_commit:
            raise ValueError("buffer provenance fields are required")


class ResidualBuffer:
    def __init__(self, mode: str) -> None:
        if mode not in ("teacher_buffer", "predicted_buffer"):
            raise ValueError("buffer mode must be teacher_buffer or predicted_buffer")
        self.mode = mode
        self._records: Dict[str, ResidualRecord] = {}

    def add(self, record: ResidualRecord) -> bool:
        if record.sample_id in self._records:
            if self._records[record.sample_id] != record:
                raise ValueError("conflicting duplicate residual sample")
            return False
        self._records[record.sample_id] = record
        return True

    @property
    def records(self):
        return tuple(self._records[key] for key in sorted(self._records))

    def write_shard(self, path, rank: int) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": BUFFER_SCHEMA_VERSION, "mode": self.mode, "rank": int(rank), "records": [asdict(row) for row in self.records]}
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load_shard(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != BUFFER_SCHEMA_VERSION:
            raise ValueError("unsupported Residual Buffer schema")
        result = cls(payload["mode"])
        for value in payload["records"]:
            value["predicted_set"] = tuple(value["predicted_set"])
            value["teacher_set"] = tuple(value["teacher_set"])
            result.add(ResidualRecord(**value))
        return result, int(payload["rank"])

    @classmethod
    def merge_shards(cls, paths, output_path, expected_sample_ids: Iterable[str], mode: str):
        merged, ranks = cls(mode), set()
        for path in paths:
            shard, rank = cls.load_shard(path)
            if shard.mode != mode or rank in ranks:
                raise ValueError("buffer shard mode/rank conflict")
            ranks.add(rank)
            for record in shard.records:
                merged.add(record)
        expected, actual = set(map(str, expected_sample_ids)), set(merged._records)
        if actual != expected:
            raise ValueError("buffer merge missing={} unexpected={}".format(sorted(expected - actual), sorted(actual - expected)))
        merged.write_shard(output_path, rank=-1)
        return merged
