"""Leakage and provenance guards for Stage 04."""

from typing import Any, Mapping


REQUIRED_CACHE_FIELDS = {
    "sample_id", "dataset_manifest_hash", "split", "tokenizer_hash",
    "model_identifier", "base_checkpoint_hash", "expert_registry_hash",
    "expert_checkpoint_hashes", "composition_mode", "rms_statistics_hash",
    "oracle_config_hash", "answer_mask_version", "answer_template_hash",
    "target_averaging", "composer_version", "code_version",
}


def validate_oracle_split(split: str) -> str:
    value = str(split).strip().lower()
    if "test" in value:
        raise ValueError("test answers cannot be used by the Oracle teacher")
    if value not in {"train", "validation", "val", "train_calibration"}:
        raise ValueError("unsupported Oracle split: {}".format(split))
    return value


def validate_cache_provenance(value: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_CACHE_FIELDS - set(value))
    if missing:
        raise ValueError("cache provenance is missing {}".format(missing))
    validate_oracle_split(str(value["split"]))
    if value["composition_mode"] not in ("direct_sum", "rms_calibrated"):
        raise ValueError("cache has an invalid composition mode")
    hashes = value["expert_checkpoint_hashes"]
    if not isinstance(hashes, Mapping) or any(not str(item) for item in hashes.values()):
        raise ValueError("expert checkpoint hashes must be a non-empty-string mapping")


def assert_temporal_boundary(expert_ids, task_id: int, temporal_scope: str, creation_tasks: Mapping[int, int]) -> None:
    cutoff = int(task_id) - 1 if temporal_scope == "historical_only" else int(task_id)
    if temporal_scope not in ("historical_only", "post_task_diagnostic"):
        raise ValueError("invalid temporal scope")
    leaking = sorted(int(value) for value in expert_ids if int(creation_tasks[int(value)]) > cutoff)
    if leaking:
        raise ValueError("future expert temporal leakage: {}".format(leaking))
