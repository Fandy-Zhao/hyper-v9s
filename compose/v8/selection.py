"""Per-sample expert selection for the four V8 teacher states.

``ComposeSelection`` (``compose/adapters/types.py``) already encodes exactly the
structure V8 needs:

===========  =========================  ============================
V8 state     experts                    composition scale
===========  =========================  ============================
``BaseOnly`` ``[-1, -1]`` (empty)       backbone only
``Reuse1``   ``[k, -1]``               ``1.0``  (active slots == 1)
``Reuse2``   ``[k, l]``               ``1/sqrt(2)``
``Residual`` ``[k, -1]`` or ``[]``     ``1.0`` / backbone; at most **one**
                                        historical context expert
===========  =========================  ============================

The Residual cardinality cap is not cosmetic.  At inference V8 keeps the fixed
Top-2 budget, and a Residual sample is the one whose second slot belongs to the
current task's candidate expert.  A two-expert historical context would therefore
ask the teacher's composition for a three-expert cardinality that the inference
path can never reproduce -- a composition the model is never allowed to make.

So V8 needs no new forward path: it needs a builder that turns a per-sample
state map into one ``ComposeSelection`` covering a whole batch.  That is what
this module provides, together with the guard rails the specification demands --
no duplicated expert id per row, no expert outside the pool, and a declared
cardinality that is derived from the state rather than trusted from the file.

**Consumers are the teacher and the query-only inference path only.**  Full-data
training does not build a selection here: it routes every micro-batch through
``compose.v7.routing.GlobalTop2Router`` from the fixed query and the active
route keys.  The former ``residual_weight(s)`` helpers, which gated the answer
loss per teacher state, belonged to the retired full-oracle trainer and are gone.
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


#: How many experts each state must name.  ``Residual`` may be empty (no
#: historical pool, or a task with no history) or hold its single best context
#: expert -- never a pair, for the cardinality reason in the module docstring.
STATE_CARDINALITY: Dict[str, Tuple[int, ...]] = {
    STATE_BASE_ONLY: (0,),
    STATE_REUSE1: (1,),
    STATE_REUSE2: (2,),
    STATE_RESIDUAL: (0, 1),
}

#: The largest number of experts a Residual training row may compose: its
#: historical context plus the current-task candidate expert.
MAX_RESIDUAL_ACTIVE_EXPERTS = 2


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
    """How many samples the teacher assigned to each state (reporting only)."""
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
    """Group sample ids by state; used for per-state reporting and tests.

    ``experts_by_sample`` is accepted for symmetry with :func:`build_selection`
    and deliberately unused: grouping is a function of the state map alone.
    """
    grouped: Dict[str, List[str]] = {state: [] for state in TEACHER_STATES}
    for sample_id, state in state_by_sample.items():
        if state not in grouped:
            raise SelectionError(f"unknown V8 teacher state {state!r}")
        grouped[state].append(str(sample_id))
    for ids in grouped.values():
        ids.sort()
    return grouped


__all__ = [
    "MAX_RESIDUAL_ACTIVE_EXPERTS",
    "SelectionError",
    "STATE_CARDINALITY",
    "build_selection",
    "selection_of_state",
    "state_cardinality_summary",
    "uniform_selection",
    "validate_state",
]

