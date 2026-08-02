"""Frozen historical-only Direct teacher definition for old-expert sufficiency."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SufficiencyLabel:
    sample_id: str
    old_expert_sufficient: bool
    residual_gain: float
    teacher_set: tuple


def teacher_sufficiency(record, minimum_raw_nll_improvement: float = 0.02) -> SufficiencyLabel:
    if record.get("temporal_scope") != "historical_only" or record.get("composition_mode") != "direct_sum":
        raise ValueError("sufficiency teacher requires historical-only Oracle-Direct")
    if "test" in str(record.get("split", "")).lower():
        raise ValueError("test Oracle is forbidden")
    selected = tuple(map(int, record["selected_expert_ids"]))
    empty_nll = float(record["empty"]["mean_nll"])
    selected_nll = float(record["selected_mean_nll"])
    selected_score = float(record["selected_score"])
    empty_score = float(record["empty"]["score"])
    gain = empty_nll - selected_nll
    sufficient = bool(selected) and selected_score < empty_score and gain >= float(minimum_raw_nll_improvement)
    return SufficiencyLabel(str(record["sample_id"]), sufficient, gain, selected)
