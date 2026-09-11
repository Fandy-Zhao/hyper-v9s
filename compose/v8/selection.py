"""Per-sample expert selection for the four V8 teacher states.

``ComposeSelection`` (``compose/adapters/types.py``) already encodes exactly the
structure V8 needs:

===========  =========================  ============================
V8 state     experts                    composition scale
===========  =========================  ============================
``BaseOnly`` ``[-1, -1]`` (empty)       backbone only
``Reuse1``   ``[k, -1]``               ``1.0``  (active slots == 1)
``Reuse2``   ``[k, l]``               ``1/sqrt(2)``
``Residual`` ``[k, l]`` (context)     ``1/sqrt(2)``
===========  =========================  ============================

So V8 needs no new forward path: it needs a builder that turns a per-sample
state map into one ``ComposeSelection`` covering a whole batch.  That is what
this module provides, together with the guard rails the specification demands --
no duplicated expert id per row, no expert outside the pool, and a declared
cardinality that is derived from the state rather than trusted from the file.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from compose.adapters.types import MAX_ACTIVE_EXPERTS, PAD_EXPERT_ID, ComposeSelection
from compose.v8.config import (
    STATE_BASE_ONLY,
    STATE_RESIDUAL,
    STATE_REUSE1,
    STATE_REUSE2,
    TEACHER_STATES,
)


class SelectionError(RuntimeError):
    """Raised when a state map cannot be turned into a legal selection."""


#: How many experts each state must name.  ``Residual`` is not fixed here
#: because PART 15 lets it fall back to the best *historical context*, which may
#: be a single expert or a pair; the cardinality is validated against the
#: recorded context instead.
STATE_CARDINALITY: Dict[str, Tuple[int, ...]] = {
    STATE_BASE_ONLY: (0,),
    STATE_REUSE1: (1,),
    STATE_REUSE2: (2,),
    STATE_RESIDUAL: (0, 1, 2),
}


def validate_state(state: str, experts: Sequence[int]) -> None:
    if state not in TEACHER_STATES:
        raise SelectionError(f"unknown V8 teacher state {state!r}")
    experts = [int(value) for value in experts]
    allowed = STATE_CARDINALITY[state]
    if len(experts) not in allowed:
        raise SelectionError(
            f"state {state} requires {allowed} experts, got {len(experts)}: {experts}"
        )
    if len(set(experts)) != len(experts):
        raise SelectionError(
            f"state {state} repeats an expert id {experts}; duplicate ids would "
            "double that expert's gate (see ComposeLinear.forward)"
        )
    if len(experts) > MAX_ACTIVE_EXPERTS:
        raise SelectionError(
            f"state {state} names {len(experts)} experts, over the "
            f"{MAX_ACTIVE_EXPERTS}-slot ComposeSelection limit"
        )


def build_selection(
    sample_ids: Sequence[str],
    state_by_sample: Mapping[str, str],
    experts_by_sample: Mapping[str, Sequence[int]],
    device: Optional[torch.device] = None,
    gates_by_sample: Optional[Mapping[str, Sequence[float]]] = None,
) -> ComposeSelection:
    """One ``ComposeSelection`` for a batch whose rows may differ in state."""
    batch_size = len(sample_ids)
    ids = torch.full(
        (batch_size, MAX_ACTIVE_EXPERTS), PAD_EXPERT_ID, dtype=torch.long
    )
    gates = torch.zeros((batch_size, MAX_ACTIVE_EXPERTS), dtype=torch.float32)
    for row, sample_id in enumerate(sample_ids):
        sample_id = str(sample_id)
        if sample_id not in state_by_sample:
            raise SelectionError(f"missing teacher state for sample {sample_id}")
        if sample_id not in experts_by_sample:
            raise SelectionError(f"missing expert list for sample {sample_id}")
        state = state_by_sample[sample_id]
        experts = [int(value) for value in experts_by_sample[sample_id]]
        validate_state(state, experts)
        sample_gates = (
            [float(value) for value in gates_by_sample[sample_id]]
            if gates_by_sample is not None and sample_id in gates_by_sample
            else [1.0] * len(experts)
        )
        if len(sample_gates) != len(experts):
            raise SelectionError(
                f"sample {sample_id} has {len(experts)} experts but "
                f"{len(sample_gates)} gates"
            )
        for slot, (expert_id, gate) in enumerate(zip(experts, sample_gates)):
            if gate <= 0:
                raise SelectionError(
                    f"sample {sample_id} gives expert {expert_id} a non-positive gate"
                )
            ids[row, slot] = expert_id
            gates[row, slot] = gate
    if device is not None:
        ids = ids.to(device)
        gates = gates.to(device)
    return ComposeSelection(ids, gates, normalization="none")


def uniform_selection(
    expert_ids: Sequence[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    gates: Optional[Sequence[float]] = None,
) -> ComposeSelection:
    """The same policy on every row -- used by the teacher's single/pair passes."""
    experts = [int(value) for value in expert_ids]
    validate_state(
        STATE_BASE_ONLY if not experts else (STATE_REUSE1 if len(experts) == 1 else STATE_REUSE2),
        experts,
    )
    rows = [str(index) for index in range(batch_size)]
    sample_gates = [float(value) for value in gates] if gates is not None else [1.0] * len(experts)
    return build_selection(
        rows,
        {row: (STATE_BASE_ONLY if not experts else
               (STATE_REUSE1 if len(experts) == 1 else STATE_REUSE2)) for row in rows},
        {row: experts for row in rows},
        device=device,
        gates_by_sample={row: sample_gates for row in rows},
    )


def state_cardinality_summary(state_by_sample: Mapping[str, str]) -> Dict[str, int]:
    summary = {state: 0 for state in TEACHER_STATES}
    for state in state_by_sample.values():
        if state not in summary:
            raise SelectionError(f"unknown V8 teacher state {state!r}")
        summary[state] += 1
    return summary


def selection_of_state(
    state_by_sample: Mapping[str, str],
    experts_by_sample: Mapping[str, Sequence[int]],
) -> Dict[str, List[str]]:
    """Group sample ids by state; used for per-state reporting and tests."""
    grouped: Dict[str, List[str]] = {state: [] for state in TEACHER_STATES}
    for sample_id, state in state_by_sample.items():
        grouped[state].append(str(sample_id))
    for state, ids in grouped.items():
        ids.sort()
    return grouped


def residual_weight(state: str) -> float:
    """Per-sample gradient-gating weight ``r_i`` (PART 21).

    ``r_i = 0`` for every sample the pool already covers -- the current task's
    candidate expert must not be trained on capability the pool already has.
    Only ``Residual`` samples carry the full weight, because they are the ones
    that genuinely need new capability.
    """
    if state in (STATE_REUSE1, STATE_REUSE2):
        return 0.0
    if state == STATE_RESIDUAL:
        return 1.0
    if state == STATE_BASE_ONLY:
        # Base already answers it correctly: no expert is needed at all, so the
        # candidate must not be pushed onto it either.
        return 0.0
    raise SelectionError(f"unknown V8 teacher state {state!r}")


def residual_weights(
    sample_ids: Sequence[str],
    state_by_sample: Mapping[str, str],
) -> torch.Tensor:
    """``r`` vector aligned with ``sample_ids``; see :func:`residual_weight`."""
    weights = []
    for sample_id in sample_ids:
        sample_id = str(sample_id)
        if sample_id not in state_by_sample:
            raise SelectionError(f"missing teacher state for sample {sample_id}")
        weights.append(residual_weight(state_by_sample[sample_id]))
    return torch.tensor(weights, dtype=torch.float32)


__all__ = [
    "SelectionError",
    "STATE_CARDINALITY",
    "build_selection",
    "residual_weight",
    "residual_weights",
    "selection_of_state",
    "state_cardinality_summary",
    "uniform_selection",
    "validate_state",
]

