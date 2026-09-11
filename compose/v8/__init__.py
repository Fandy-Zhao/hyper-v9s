"""Hyper-LLaVA V8: Answer-Supervised Multi-Key Expert Composition.

V7 asked *which two experts should be composed*.  V8 asks a sharper question
first -- **which current-task samples can the existing pool already solve?** --
and answers it with the task metric rather than a loss threshold:

    correct.           the task metric decides `solved`; it is the only gate
    NLL ranks.         loss only orders already-solved experts and measures soft
                       contribution
    keys localise.     an expert keeps its origin key and gains a task alias key
                       per later task that actually needs it (1 Expert : N Keys)
    new experts fill.  a candidate expert only learns capability the old pool
                       genuinely lacks (per-sample gradient gating)

Package layout:

``config``          explicit, validated configuration (no scattered thresholds)
``pool``            ``MultiKeyExpertPool`` -- experts and keys as separate
                    identities, plus the byte-exact V7 migration
``routing``         expert-level Multi-Key Router (per-expert max aggregation)
``selection``       per-sample ``ComposeSelection`` for the four teacher states
``metric_adapter``  the ``M`` signal, reproducing the official UCIT scores
``teacher``         the Answer-Supervised Expert Teacher
``key_learning``    alias-key initialisation and the positive/ranking losses
``gating``          per-sample gradient gating (``L_answer_residual``)
``audit``           candidate recall and full-pool capability audits
``pruning``         alias-key and candidate retirement rules
``commit``          end-of-task freeze, V7-compatible key export
``query``           the fixed V7 query, re-exported with a contract check
``inference``       query-only routing, AST-verified free of supervision
``checkpoint``      resume with the full expert/key id mapping
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
from compose.v8.commit import commit_task, write_v7_compatible_keys  # noqa: F401
from compose.v8.metric_adapter import TaskMetricAdapter  # noqa: F401
from compose.v8.pool import MultiKeyExpertPool, alias_key_init  # noqa: F401
from compose.v8.pruning import apply_pruning, plan_key_pruning  # noqa: F401
from compose.v8.routing import MultiKeyRouter  # noqa: F401
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
    "alias_key_init",
    "apply_pruning",
    "assert_frozen_contract",
    "build_selection",
    "commit_task",
    "plan_key_pruning",
    "uniform_selection",
    "write_v7_compatible_keys",
]
