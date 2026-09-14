"""Auditable, mergeable RMS calibration statistics."""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


STATISTIC_VERSION = 1


def stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StatisticKey:
    expert_id: int
    layer_name: str
    module_name: str
    target_module_type: str
    statistic_version: int = STATISTIC_VERSION

    def token(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class OnlineMoments:
    """Float64 Welford accumulator over tensor elements."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.sum_squares = 0.0

    def update(self, values: torch.Tensor) -> None:
        # Inputs are bf16/fp16/fp32. Reduce a batch on-device in fp32, then
        # merge the resulting scalar moments with float64 Welford arithmetic.
        # This avoids materializing transformer activations in fp64.
        data = values.detach().to(dtype=torch.float32).reshape(-1)
        if data.numel() == 0:
            return
        n = int(data.numel())
        mean = float(data.mean().item())
        centered = data - mean
        m2 = float(torch.sum(centered * centered).item())
        sum_squares = float(torch.sum(data * data).item())
        self.merge(n, mean, m2, sum_squares)

    @staticmethod
    def device_moments(values: torch.Tensor):
        """The same three reductions :meth:`update` runs, left on the device.

        Returns ``(numel, mean, m2, sum_squares)`` where the three statistics
        are 0-dim device tensors rather than Python floats, so a caller that
        needs many moments can batch one synchronisation instead of three per
        call.  The arithmetic is character-for-character the same as
        :meth:`update`: the mean is reduced first and subtracted from the
        data as a scalar, so ``data - mean`` computes the identical fp32
        subtraction whether ``mean`` arrives as a Python float or as a 0-dim
        tensor holding that same fp32 value.
        """
        data = values.detach().to(dtype=torch.float32).reshape(-1)
        if data.numel() == 0:
            return None
        mean = data.mean()
        centered = data - mean
        return (
            int(data.numel()),
            mean,
            torch.sum(centered * centered),
            torch.sum(data * data),
        )

    def merge_device(self, numel: int, mean, m2, sum_squares) -> None:
        """:meth:`merge` for the 0-dim tensors of :meth:`device_moments`."""
        self.merge(numel, mean, m2, sum_squares)

    def merge(self, count: int, mean: float, m2: float, sum_squares: float) -> None:
        count = int(count)
        if count <= 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2, self.sum_squares = count, float(mean), float(m2), float(sum_squares)
            return
        total = self.count + count
        delta = float(mean) - self.mean
        self.m2 += float(m2) + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.sum_squares += float(sum_squares)
        self.count = total

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0

    @property
    def rms(self) -> float:
        return math.sqrt(max(self.sum_squares / self.count, 0.0)) if self.count else 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2, "sum_squares": self.sum_squares}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "OnlineMoments":
        result = cls()
        result.merge(int(state["count"]), float(state["mean"]), float(state["m2"]), float(state["sum_squares"]))
        return result

    def all_reduce_(self, device: Optional[torch.device] = None) -> "OnlineMoments":
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return self
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local = torch.tensor([self.count, self.mean * self.count, self.sum_squares], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(local)
        count = int(local[0].item())
        total = float(local[1].item())
        squares = float(local[2].item())
        mean = total / count if count else 0.0
        # Population M2 reconstructed from globally additive first/second moments.
        self.count, self.mean, self.sum_squares = count, mean, squares
        self.m2 = max(squares - count * mean * mean, 0.0)
        return self


class RMSStatistics:
    """Calibration-only statistics plus provenance-based cache validation."""

    def __init__(self, provenance: Mapping[str, Any]) -> None:
        required = {"calibration_split", "checkpoint_hash", "dataset_manifest_hash", "composition_config_hash"}
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError("statistics provenance is missing {}".format(missing))
        split = str(provenance["calibration_split"]).lower()
        if "test" in split:
            raise ValueError("test data cannot be used for RMS calibration")
        self.provenance = dict(provenance)
        self.entries = {}  # type: Dict[str, Dict[str, Any]]

    def entry(self, key: StatisticKey, dtype: str) -> Dict[str, Any]:
        """The accumulator record for ``key``, created on first use.

        ``dtype`` is the recorded delta dtype; it is only honoured when the
        record is created, exactly as the inline ``setdefault`` in
        :meth:`update` treats it.
        """
        token = key.token()
        record = self.entries.get(token)
        if record is None:
            record = {
                "key": asdict(key), "delta": OnlineMoments(), "output": OnlineMoments(),
                "base_output": OnlineMoments(), "dtype": str(dtype), "sample_count": 0,
            }
            self.entries[token] = record
        return record

    def update(self, key: StatisticKey, delta: torch.Tensor, output: torch.Tensor, base_output: Optional[torch.Tensor] = None) -> None:
        entry = self.entry(key, str(delta.dtype))
        entry["delta"].update(delta)
        entry["output"].update(output)
        if base_output is not None:
            entry["base_output"].update(base_output)
        entry["sample_count"] += int(delta.shape[0]) if delta.ndim else 1

    def delta_rms(self, expert_id: int, layer_name: str) -> float:
        matches = [entry for entry in self.entries.values() if int(entry["key"]["expert_id"]) == int(expert_id) and entry["key"]["layer_name"] == layer_name]
        if len(matches) != 1:
            raise KeyError("expected one RMS statistic for expert {} layer {}, found {}".format(expert_id, layer_name, len(matches)))
        return matches[0]["delta"].rms

    def all_reduce_(self, device: Optional[torch.device] = None) -> "RMSStatistics":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for entry in self.entries.values():
            for field in ("delta", "output", "base_output"):
                entry[field].all_reduce_(device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sample_count = torch.tensor(entry["sample_count"], dtype=torch.int64, device=device)
                torch.distributed.all_reduce(sample_count)
                entry["sample_count"] = int(sample_count.item())
        return self

    def all_reduce_batched_(self, device: Optional[torch.device] = None) -> "RMSStatistics":
        """Exact-moment distributed reduction with two collective calls.

        ``all_reduce_`` issues four blocking collectives per statistic entry.
        The accelerated collector has no dependency between those entries, so
        their independent fp64 moment triples and int64 sample counts can be
        packed into contiguous tensors.  Each lane still receives the same
        SUM reduction and reconstruction as the scalar path; only launch and
        synchronization overhead is removed.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return self
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        entries = list(self.entries.values())
        if not entries:
            return self
        packed = []
        sample_counts = []
        for entry in entries:
            for field in ("delta", "output", "base_output"):
                moment = entry[field]
                packed.extend((
                    float(moment.count),
                    float(moment.mean * moment.count),
                    float(moment.sum_squares),
                ))
            sample_counts.append(int(entry["sample_count"]))
        moments_tensor = torch.tensor(packed, dtype=torch.float64, device=device)
        count_tensor = torch.tensor(sample_counts, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(moments_tensor)
        torch.distributed.all_reduce(count_tensor)
        values = moments_tensor.cpu().tolist()
        counts = count_tensor.cpu().tolist()
        cursor = 0
        for entry, sample_count in zip(entries, counts):
            for field in ("delta", "output", "base_output"):
                count = int(values[cursor])
                total = float(values[cursor + 1])
                squares = float(values[cursor + 2])
                cursor += 3
                mean = total / count if count else 0.0
                moment = entry[field]
                moment.count = count
                moment.mean = mean
                moment.sum_squares = squares
                moment.m2 = max(squares - count * mean * mean, 0.0)
            entry["sample_count"] = int(sample_count)
        return self

    def state_dict(self) -> Dict[str, Any]:
        packed = {}
        for token, entry in self.entries.items():
            packed[token] = {key: (value.state_dict() if isinstance(value, OnlineMoments) else value) for key, value in entry.items()}
        return {"statistic_version": STATISTIC_VERSION, "provenance": self.provenance, "entries": packed}

    def save_json(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state_dict(), handle, indent=2, sort_keys=True)
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
    def load_json(cls, path, expected_provenance: Mapping[str, Any], registry=None) -> "RMSStatistics":
        with Path(path).open(encoding="utf-8") as handle:
            state = json.load(handle)
        if int(state.get("statistic_version", -1)) != STATISTIC_VERSION:
            raise ValueError("unsupported RMS statistic version")
        if dict(state.get("provenance", {})) != dict(expected_provenance):
            raise ValueError("RMS statistics cache invalidated by provenance/hash change")
        result = cls(expected_provenance)
        for token, packed in state.get("entries", {}).items():
            if registry is not None:
                expert_id = int(packed["key"]["expert_id"])
                metadata = registry.get(expert_id)
                status = getattr(metadata.status, "value", metadata.status)
                if status == "archived":
                    raise ValueError(
                        "RMS statistics for archived expert {} cannot be loaded".format(expert_id)
                    )
            entry = dict(packed)
            for field in ("delta", "output", "base_output"):
                entry[field] = OnlineMoments.from_state_dict(entry[field])
            result.entries[token] = entry
        return result

    def summary(self) -> Dict[str, Any]:
        rows = []
        for entry in self.entries.values():
            rows.append({**entry["key"], "sample_count": entry["sample_count"], "delta_rms": entry["delta"].rms,
                         "output_rms": entry["output"].rms, "base_output_rms": entry["base_output"].rms,
                         "mean": entry["delta"].mean, "variance": entry["delta"].variance, "dtype": entry["dtype"],
                         "calibration_split": self.provenance["calibration_split"],
                         "source_checkpoint_hash": self.provenance["checkpoint_hash"],
                         "source_dataset_manifest_hash": self.provenance["dataset_manifest_hash"]})
        return {"provenance": self.provenance, "entries": sorted(rows, key=lambda row: (row["layer_name"], row["expert_id"]))}
