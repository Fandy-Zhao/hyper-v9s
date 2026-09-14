"""Hyper-LLaVA V8: Few-shot capability screening + multi-key Global Top-2.

**This is the only V8 training semantics in the repository.**  The former
sample-level/full-oracle variant, including its residual-gated and per-sample
assignment interfaces, has been retired; see
``docs/reports/V8_FORMAL_METHOD.md``.

One task of V8 is a five-step loop::

    S2  Few-shot answer-supervised capability teacher (small TRAIN subset)
            -> which *historical* experts already solve part of this task
    S3  Task-level reusable set R_t + current-task reuse keys for R_t
            -> the teacher's per-sample selections are used here and nowhere else
    S4  Full-data query-only Global Top-2 co-evolution
            -> all train samples; routing reads queries and active route keys only
    S6  Validation remove-and-reroute pruning
    S7  Commit -> S8 official lower-triangle evaluation

The full-data stage never reads a teacher artifact.  Its only routing inputs are
the fixed query and the active route keys; ground-truth answers enter solely as
the ordinary supervised answer loss.  ``compose/v8/workflow.py`` is the stage
implementation and ``compose/experiments/v8_task_run.py`` the only formal entry.

Package layout:

``config``          explicit, validated configuration (no scattered thresholds)
``pool``            ``MultiKeyExpertPool`` -- reads the committed V7 pool frozen
``routing``         expert-level Multi-Key Router (per-expert max aggregation)
``selection``       per-sample ``ComposeSelection`` for the teacher's 0/1/2 routes
``metric_adapter``  the ``M`` signal, reproducing the official UCIT scores
``teacher``         the Answer-Supervised Expert Teacher
``screening``       teacher subset sampling + the task-level R_t artifact
``workflow``        the teacher-screening stage the V8 pipeline drives
``audit``           candidate recall and full-pool capability audits
``generate``        route -> answer generation with an append-only cache
``query``           the fixed V7 query, re-exported with a contract check
``inference``       query-only routing, AST-verified free of supervision

The V7 kernels (``V7ComposeTrainer``, ``V7ExpertKeyPool``, ``GlobalTop2Router``,
``CandidatePruner``) are **reused numerical implementations, not a second V8
method**: V8 owns the orchestrator, the contract and the teacher, and delegates
the arithmetic to V7 so that the two never drift apart.
"""

from compose.v8.config import (  # noqa: F401
    STATE_BASE_ONLY,
    STATE_RESIDUAL,
    STATE_REUSE1,
    STATE_REUSE2,
    TEACHER_STATES,
    V8Config,
    assert_frozen_contract,
)
from compose.v8.metric_adapter import TaskMetricAdapter  # noqa: F401
from compose.v8.pool import MultiKeyExpertPool, alias_key_init  # noqa: F401
from compose.v8.routing import MultiKeyRouter  # noqa: F401
from compose.v8.screening import aggregate_reusable_experts  # noqa: F401
from compose.v8.selection import build_selection, uniform_selection  # noqa: F401
from compose.v8.teacher import AnswerSupervisedTeacher  # noqa: F401

__all__ = [
    "AnswerSupervisedTeacher",
    "MultiKeyExpertPool",
    "MultiKeyRouter",
    "STATE_BASE_ONLY",
    "STATE_RESIDUAL",
    "STATE_REUSE1",
    "STATE_REUSE2",
    "TEACHER_STATES",
    "TaskMetricAdapter",
    "V8Config",
    "aggregate_reusable_experts",
    "alias_key_init",
    "assert_frozen_contract",
    "build_selection",
    "uniform_selection",
]
