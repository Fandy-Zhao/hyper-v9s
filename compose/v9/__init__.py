"""V9-S: Answer-Guided Responsibility Distillation.

The simplified form of V9: Answer-Guided Key--Expert Co-Evolution.

The closed loop, in one line:

    the key decides where to learn  ->  the expert learns what to do
    ->  the answer judges whether it helped  ->  the answer corrects the key

What V9-S removes from the training loop is the thing that made V8 unable to scale:
there is no per-expert answer enumeration and no per-expert-pair enumeration.
One backbone forward carries every candidate expert through differentiable
gates, one ground-truth answer loss is taken, and one ``autograd.grad`` against
the gate tensor turns that loss into a per-expert *local conditional
contribution estimate* which supervises the routing keys.

Module map:

``config``        every knob, no magic numbers elsewhere
``keys``          ``1 expert : base identity + task-scoped keys``
``retrieval``     frozen-base-key Top-C recall + periodic wide recall, cached once per task
``schedule``      bootstrap -> co-evolution -> discretisation
``router``        independent-sigmoid gates over a dense candidate row
``contribution``  the gate gradient and the responsibility it induces
``losses``        ``L_ans + lambda_key*L_key + lambda_sparse*L_sparse + ...``
``trainer``       one forward, one gate gradient, one backward
``multi_key``     global multi-key aggregation over effective keys
``inference``     query-only deployment routing (purity-checked)
``audit``         task-end task-key audit and candidate commit
``checkpoint``    atomic task-level save/resume
"""

from .audit import (
    ExpertAuditRecord,
    V9AuditError,
    apply_candidate_commit,
    apply_historical_task_key_audit,
    audit_candidates,
    audit_historical_task_keys,
    candidate_usage_entropy,
    max_key_cosine,
    pairwise_key_cosine,
    validation_answer_gain,
)
from .checkpoint import (
    V9_CHECKPOINT_VERSION,
    capture_rng_state,
    load_v9_checkpoint,
    restore_rng_state,
    save_v9_checkpoint,
)
from .data import (
    V9DataError,
    V9QueryCollator,
    V9QueryDataset,
    V9TaskData,
    build_task_retrieval,
    write_retrieval_manifest,
)
from .config import (
    HISTORICAL_AGGREGATION_MULTI_KEY_MAX,
    RESPONSIBILITY_GATE_GRADIENT,
    RESPONSIBILITY_LOSS_BCE,
    ROUTING_INDEPENDENT_SIGMOID,
    V9_COMPOSE_MODE,
    V9_METHOD_NAME,
    V9_METHOD_TITLE,
    V9AuditConfig,
    V9BootstrapConfig,
    V9Config,
    V9ExactOracleConfig,
    V9ExpertConfig,
    V9InferenceConfig,
    V9KeyConfig,
    V9LossConfig,
    V9QueryConfig,
    V9ResponsibilityConfig,
    V9RetrievalConfig,
    V9RoutingConfig,
    V9ScheduleConfig,
    V9ValidationConfig,
    V9WideRetrievalConfig,
    assert_frozen_contract,
    load_v9_config,
)
from .contribution import (
    V9Contribution,
    answer_derived_responsibility,
    calibration_report,
    contribution_statistics,
    exact_removal_contribution,
    gate_gradient,
    local_conditional_contribution,
)
from .inference import (
    V9InferenceError,
    V9InferenceRouter,
    assert_v9_inference_purity,
    validate_inference_policy,
)
from .keys import (
    LEGACY_KEY_TYPE_TASK_RESIDUAL,
    V9KeyPool,
    V9KeyPoolError,
    initialize_candidate_keys,
    spherical_kmeans_keys,
)
from .losses import (
    V9LossTerms,
    budget_loss,
    compose_total_loss,
    key_responsibility_loss,
    sparse_loss,
)
from .multi_key import (
    V9MultiKeyError,
    V9MultiKeyRouteResult,
    V9MultiKeyRouter,
    aggregatable_expert_ids,
    aggregation_diagnostics,
    memory_key_ids,
)
from .retrieval import (
    HistoricalTopC,
    V9RetrievalError,
    build_historical_topc,
    compose_candidate_rows,
    is_wide_step,
    load_or_build_historical_topc,
    pool_fingerprint,
    retrieval_diagnostics,
)
from .router import V9RouteOutput, V9Router, V9RouterError
from .trainer import V9ComposeTrainer, V9TrainerError, answer_loss_key_gradient
from .schedule import (
    STAGE_BOOTSTRAP,
    STAGE_HARD,
    STAGE_SOFT,
    V9StageScheduler,
    V9StageState,
)

__all__ = [
    "ExpertAuditRecord",
    "HistoricalTopC",
    "HISTORICAL_AGGREGATION_MULTI_KEY_MAX",
    "RESPONSIBILITY_GATE_GRADIENT",
    "RESPONSIBILITY_LOSS_BCE",
    "LEGACY_KEY_TYPE_TASK_RESIDUAL",
    "V9DataError",
    "V9QueryCollator",
    "V9QueryDataset",
    "V9TaskData",
    "build_task_retrieval",
    "retrieval_diagnostics",
    "write_retrieval_manifest",
    "ROUTING_INDEPENDENT_SIGMOID",
    "STAGE_BOOTSTRAP",
    "STAGE_HARD",
    "STAGE_SOFT",
    "V9AuditConfig",
    "V9AuditError",
    "V9BootstrapConfig",
    "V9_CHECKPOINT_VERSION",
    "V9Config",
    "V9Contribution",
    "V9ExactOracleConfig",
    "V9ExpertConfig",
    "V9InferenceConfig",
    "V9InferenceError",
    "V9InferenceRouter",
    "V9KeyConfig",
    "V9KeyPool",
    "V9KeyPoolError",
    "V9LossConfig",
    "V9LossTerms",
    "V9MultiKeyError",
    "V9MultiKeyRouteResult",
    "V9MultiKeyRouter",
    "V9QueryConfig",
    "V9ResponsibilityConfig",
    "V9RetrievalConfig",
    "V9RetrievalError",
    "V9RouteOutput",
    "V9Router",
    "V9RouterError",
    "V9WideRetrievalConfig",
    "V9_COMPOSE_MODE",
    "V9_METHOD_TITLE",
    "V9ComposeTrainer",
    "V9TrainerError",
    "answer_loss_key_gradient",
    "V9RoutingConfig",
    "V9ScheduleConfig",
    "V9StageScheduler",
    "V9StageState",
    "V9ValidationConfig",
    "aggregatable_expert_ids",
    "aggregation_diagnostics",
    "answer_derived_responsibility",
    "apply_candidate_commit",
    "apply_historical_task_key_audit",
    "assert_frozen_contract",
    "assert_v9_inference_purity",
    "audit_candidates",
    "audit_historical_task_keys",
    "budget_loss",
    "build_historical_topc",
    "calibration_report",
    "candidate_usage_entropy",
    "capture_rng_state",
    "compose_candidate_rows",
    "compose_total_loss",
    "contribution_statistics",
    "exact_removal_contribution",
    "gate_gradient",
    "initialize_candidate_keys",
    "key_responsibility_loss",
    "is_wide_step",
    "load_or_build_historical_topc",
    "load_v9_checkpoint",
    "load_v9_config",
    "local_conditional_contribution",
    "max_key_cosine",
    "memory_key_ids",
    "pairwise_key_cosine",
    "pool_fingerprint",
    "restore_rng_state",
    "retrieval_diagnostics",
    "save_v9_checkpoint",
    "sparse_loss",
    "spherical_kmeans_keys",
    "validate_inference_policy",
    "validation_answer_gain",
]
