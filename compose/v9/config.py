"""Explicit V9-S configuration: Answer-Guided Responsibility Distillation.

Every knob the specification names is a field here, so no threshold is a magic
number scattered through the implementation.  Field defaults are the
specification defaults; experiment configs override them.

The class is frozen and round-trips through ``to_dict``/``from_dict`` exactly
like ``compose.v8.config.V8Config``, so the whole recipe can be written into
every checkpoint, run contract and audit record.

Three V9 v1 ingredients are absent by construction, and their absence is what
the defaults below encode:

* **no residual-key decomposition** -- a historical expert that is reused on a
  later task carries an independent, absolute current-task key (the V8
  ``task_alias`` role).  There is no ``base + gamma * delta`` mixing, so there
  is no ``gamma`` to configure and no base key to re-anchor;
* **no direct answer gradient to any key** -- ``routing.direct_answer_gradient
  _to_key`` must stay ``False``.  The gate tensor that drives the forward
  composition is detached from the key graph, so ``L_ans`` reaches *no* key
  parameter.  Keys are supervised by exactly one term, ``L_key``, whose target
  is the answer-derived responsibility (see :mod:`compose.v9.contribution`);
* **no fixed exploration expert** -- exposure for historical experts that the
  Top-C recall misses comes from periodic *wide retrieval*
  (:class:`V9WideRetrievalConfig`), not from a permanently reserved slot.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, Mapping

from compose.v8.config import V7_QUERY_MODULE_HASH


#: The one formal V9-S method name, used as the ``method:`` value in every
#: config and written into the run contract, the committed pool manifest and
#: every audit record.
V9_METHOD_NAME = "v9s"

#: The ``--compose_mode`` value that selects the V9-S training loop.
#:
#: It is defined here, next to the method name, because two files need it: the
#: training entry point that branches on it and the task orchestrator that
#: passes it.  A literal duplicated across the two would be a drift waiting to
#: happen, and the failure mode -- a task that silently trains the wrong loop --
#: is exactly the kind that a run would not notice for hours.
V9_COMPOSE_MODE = "v9s_responsibility_distillation"

#: The human-readable name of the method, for reports and logs.  The
#: specification's accepted terminology is either this or "Answer-Guided
#: Key-Expert Co-Evolution (Simplified)"; the older "Task Residual Key" /
#: "Base + Residual Key" vocabulary is deliberately not used anywhere on the
#: V9-S main path.
V9_METHOD_TITLE = "Answer-Guided Responsibility Distillation"

#: Query: the frozen V7/V8 coordinate system, unchanged.
V9_QUERY_TYPE = "fixed_visual_text_concat"

#: Routing: independent per-expert sigmoid.  Experts are not mutually
#: exclusive classes -- a sample may need none, one, or two -- so a softmax
#: over the candidate set would force a constant total mass onto every sample.
ROUTING_INDEPENDENT_SIGMOID = "independent_sigmoid"

#: Candidate-key initialisation strategies.
CANDIDATE_INIT_KMEANS = "spherical_kmeans"
CANDIDATE_INIT_PERTURBED_MEAN = "task_mean_perturbed"
CANDIDATE_INITS = (CANDIDATE_INIT_KMEANS, CANDIDATE_INIT_PERTURBED_MEAN)

#: Key aggregation at inference, inherited from V8: ``max`` over an expert's
#: key memory, then Top-K over distinct experts.
AGGREGATION_MAX = "max"

#: The historical-retrieval aggregation contract.  ``preserve_v8`` means the
#: Top-C recall keeps V8's frozen-base-key geometry exactly: the recall set is
#: ranked by ``cosine(q_i, e_k_base)`` over keys that never move during the
#: task, so it is computed once and cached.
HISTORICAL_AGGREGATION_MULTI_KEY_MAX = "frozen_multi_key_max"


@dataclass(frozen=True)
class V9QueryConfig:
    """The fixed multimodal query is reused, never re-derived (spec §4)."""

    type: str = V9_QUERY_TYPE
    source_module_hash: str = V7_QUERY_MODULE_HASH
    visual_dim: int = 768
    text_dim: int = 768
    query_dim: int = 1536
    cache: bool = True
    trainable_parameter_count: int = 0

    def __post_init__(self) -> None:
        if self.type != V9_QUERY_TYPE:
            raise ValueError(
                f"V9 keeps the fixed query: type must be {V9_QUERY_TYPE!r}"
            )
        if self.source_module_hash != V7_QUERY_MODULE_HASH:
            raise ValueError("V9 query module hash must match the V7 fixed query")
        if (self.visual_dim, self.text_dim, self.query_dim) != (768, 768, 1536):
            raise ValueError("V9 query dimensions must be 768 + 768 = 1536")
        if self.trainable_parameter_count != 0:
            raise ValueError("V9 query must have zero trainable parameters")


@dataclass(frozen=True)
class V9ExpertConfig:
    """Capability carriers (spec §3)."""

    rank: int = 8
    alpha: float = 16.0
    #: Two, not four.  The main method's claim is that the answer can *judge* a
    #: small candidate set; a wider default multiplies the per-step multi-LoRA
    #: cost and the number of experts competing for the same answer signal
    #: without adding a distinct decision.  ``M=4`` stays available as the
    #: declared ablation.
    num_current_candidates: int = 2
    lora_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("expert rank must be positive")
        if self.alpha <= 0:
            raise ValueError("expert alpha must be positive")
        if self.num_current_candidates < 1:
            raise ValueError("num_current_candidates must be at least 1")


@dataclass(frozen=True)
class V9RetrievalConfig:
    """Cheap frozen-base-key recall that bounds the per-step candidate set.

    The section is named ``historical_retrieval`` in the config file, after what
    it recalls: the *historical* experts.  The current candidates are not
    retrieved -- they are always present.
    """

    top_c: int = 4
    aggregation: str = HISTORICAL_AGGREGATION_MULTI_KEY_MAX
    cache: bool = True

    def __post_init__(self) -> None:
        if self.top_c < 0:
            raise ValueError("historical_retrieval.top_c must be non-negative")
        if self.aggregation != HISTORICAL_AGGREGATION_MULTI_KEY_MAX:
            raise ValueError(
                "V9-S keeps V8's historical recall: the candidate set is ranked "
                "by cosine against frozen base keys, computed once per task"
            )


@dataclass(frozen=True)
class V9WideRetrievalConfig:
    """Periodic wide recall: exposure without a permanently reserved slot.

    A historical expert whose base key ranks ``C+1`` never enters the Top-C, so
    it never receives answer gradient and its key can never be corrected -- a
    failure that is both invisible and self-confirming.  V9 v1 held a slot open
    for such an expert on *every* row, which pays for the guarantee on every
    step and hands the answer an expert it never asked for.

    V9-S widens the recall instead: on a fraction ``ratio`` of optimizer steps
    the row carries ``wide_top_c`` historical slots rather than
    ``historical_retrieval.top_c``.  The door stays open, the persistent cost is one extra
    cache column per row rather than one extra active expert per row, and the
    wide steps are exactly where the historical-reuse diagnostics are read.
    """

    enabled: bool = True
    ratio: float = 0.05
    top_c: int = 8

    def __post_init__(self) -> None:
        if not (0.0 <= self.ratio <= 1.0):
            raise ValueError("wide_retrieval.ratio must lie in [0, 1]")
        if self.top_c < 1:
            raise ValueError("wide_retrieval.top_c must be positive")
        if self.enabled and self.ratio <= 0.0:
            raise ValueError(
                "wide retrieval is enabled with ratio 0: no step would ever "
                "widen, so the recall door would stay shut"
            )


@dataclass(frozen=True)
class V9KeyConfig:
    """Key geometry and initialisation (spec §3, §5).

    There is no ``gamma`` here: V9-S has no residual mixing.  A reused
    historical expert's current-task key is an independent absolute key, and it
    is initialised at the expert's own base key so the expert enters the new
    task routing exactly as itself.
    """

    aggregation: str = AGGREGATION_MAX
    candidate_init: str = CANDIDATE_INIT_KMEANS
    candidate_init_perturbation: float = 0.01
    candidate_init_samples: int = 4096
    candidate_init_seed: int = 42

    def __post_init__(self) -> None:
        if self.aggregation != AGGREGATION_MAX:
            raise ValueError(
                "V9 inherits the V8 aggregation: max over an expert's key "
                "memory, then Top-K over distinct experts"
            )
        if self.candidate_init not in CANDIDATE_INITS:
            raise ValueError(f"candidate_init must be one of {CANDIDATE_INITS}")
        if self.candidate_init_perturbation < 0:
            raise ValueError("candidate_init_perturbation must be non-negative")
        if self.candidate_init_samples < 1:
            raise ValueError("candidate_init_samples must be positive")


@dataclass(frozen=True)
class V9RoutingConfig:
    """Sparse differentiable routing (spec §7)."""

    type: str = ROUTING_INDEPENDENT_SIGMOID
    temperature_start: float = 1.0
    temperature_mid: float = 0.5
    temperature_end: float = 0.2
    #: ``K`` in the deployed Top-K rule (spec §35's ``max_inference_experts``).
    #: Named after where it applies rather than after a schedule stage: it *is*
    #: the inference rule, and the discretisation stage simply stops pretending
    #: otherwise.
    max_inference_experts: int = 2
    learnable_bias: bool = False
    bias_init: float = 0.0
    #: Must remain ``False``; ``True`` is refused rather than supported.
    #:
    #: The gate tensor is still differentiable w.r.t. the keys -- that is how
    #: the *contribution* is measured -- but the copy that drives the forward
    #: composition is detached from the key graph.  Without the detach a key
    #: would be pushed by two different objectives at once: directly by
    #: ``dL_ans/dkey`` through the gate, and indirectly by the responsibility
    #: teacher through ``L_key``.  Those two do not agree in general (the
    #: teacher is positive-only and per-sample normalised), so the key would be
    #: trained by a sum of a gradient and a target signal.  Detaching leaves
    #: exactly one answer-side path into the keys, and it is the auditable one.
    direct_answer_gradient_to_key: bool = False

    def __post_init__(self) -> None:
        if self.type != ROUTING_INDEPENDENT_SIGMOID:
            raise ValueError(
                "V9-S routing is independent sigmoid: experts are not "
                "mutually exclusive, so a softmax would force a constant "
                "total gate mass onto every sample"
            )
        if self.direct_answer_gradient_to_key:
            raise ValueError(
                "direct_answer_gradient_to_key must be False: the answer may "
                "reach a key only through contribution -> responsibility -> "
                "L_key, never through a direct gate gradient.  Two "
                "simultaneous answer-side gradients on one key is the "
                "supervision conflict V9-S exists to remove"
            )
        if self.learnable_bias:
            raise ValueError(
                "V9-S does not permit a learnable routing bias: deployment ranks "
                "by retained-key cosine, so learning a bias would create a "
                "train/inference routing mismatch"
            )
        if not (0.0 < self.temperature_end <= self.temperature_mid <= self.temperature_start):
            raise ValueError(
                "temperatures must satisfy 0 < end <= mid <= start"
            )
        if self.max_inference_experts < 1:
            raise ValueError("max_inference_experts must be at least 1")


@dataclass(frozen=True)
class V9BootstrapConfig:
    """Cold-start exposure floor (spec §15, §35).

    A freshly created expert has ``Delta_theta ~ 0``, hence
    ``dL/da_new ~ 0``: without a floor it can never earn the answer gradient
    that would give it capability.  The floor is an *exposure* device only --
    it breaks symmetry, it never defines a permanent expert boundary.

    The floor is named after what it is, a floor on the gate, and not after the
    mechanism that used to be here.  It has no relation to the removed
    exploration expert: the route it holds open is the *current candidates*',
    which are the experts a fresh task has no evidence about yet.

    How long the floor is applied for is ``schedule.bootstrap_ratio``.  That is
    deliberately the only place the bootstrap duration is written down: two
    configurable copies of one number is a way to have them disagree.
    """

    gate_floor: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.gate_floor < 1.0):
            raise ValueError("gate_floor must lie in [0, 1)")


@dataclass(frozen=True)
class V9ScheduleConfig:
    """Three-stage schedule (spec §16)."""

    bootstrap_ratio: float = 0.08
    soft_ratio: float = 0.67
    hard_ratio: float = 0.25

    def __post_init__(self) -> None:
        ratios = (self.bootstrap_ratio, self.soft_ratio, self.hard_ratio)
        if any(value < 0 for value in ratios):
            raise ValueError("stage ratios must be non-negative")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(
                f"stage ratios must sum to 1.0, got {sum(ratios)}"
            )
        if self.hard_ratio <= 0:
            raise ValueError(
                "the discretisation stage is required: deployment is Top-2"
            )


@dataclass(frozen=True)
class V9LossConfig:
    """Total objective (spec §13, §14, §35).

        L_total = L_ans + lambda_key * L_key + lambda_sparse * L_sparse
                  [+ lambda_budget * L_budget]

    The budget term is **off by default**.  ``L_sparse`` already states "use few
    experts"; adding a one-sided penalty at the same time is two objectives on
    one quantity, and with ``max_inference_experts = 2`` the deployment rule already
    enforces the budget at inference.  ``use_budget_loss: true`` restores it as
    the declared ablation.
    """

    lambda_key: float = 0.1
    lambda_sparse: float = 0.01
    lambda_budget: float = 0.01
    use_budget_loss: bool = False
    responsibility_epsilon: float = 1e-8
    sparse_budget: float = 2.0

    def __post_init__(self) -> None:
        for name in ("lambda_key", "lambda_sparse", "lambda_budget"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.responsibility_epsilon <= 0:
            raise ValueError("responsibility_epsilon must be positive")
        if self.sparse_budget < 1:
            raise ValueError("sparse_budget must be at least 1")


@dataclass(frozen=True)
class V9CompositionConfig:
    """RMS-calibrated LoRA composition (spec §8)."""

    rms_calibration: bool = True
    #: ``"none"``: h <- base + sum_k a_k * kappa_k * u_k.  V8's 1/sqrt(N)
    #: cardinality factor would scale the whole expert delta by the number of
    #: retrieved candidates rather than by their fitted gates.
    cardinality_scale: str = "none"

    def __post_init__(self) -> None:
        if self.cardinality_scale not in ("v8", "none"):
            raise ValueError("cardinality_scale must be 'v8' or 'none'")


@dataclass(frozen=True)
class V9ValidationConfig:
    """Exact-contribution calibration (spec §30).

    Used only on a small held-out sample to check that the gate-gradient
    estimate is a usable proxy.  The exact oracle is never a training
    dependency.
    """

    contribution_calibration: bool = True
    exact_contribution_samples: int = 64
    calibration_split: str = "validation"
    #: How many expert *pairs* the calibration measures when the run declares
    #: ``pair_rerank``.  One backbone forward per pair on the bounded held-out
    #: sample -- the only place a pair can be afforded at all.
    pair_rerank_pairs: int = 4

    def __post_init__(self) -> None:
        if self.exact_contribution_samples < 1:
            raise ValueError("exact_contribution_samples must be positive")
        if self.calibration_split not in ("validation", "train"):
            raise ValueError("calibration_split must be 'validation' or 'train'")
        if self.pair_rerank_pairs < 2:
            raise ValueError(
                "pair_rerank_pairs must be at least 2: one deployed pair plus a "
                "challenger is the smallest comparison that means anything"
            )


@dataclass(frozen=True)
class V9AuditConfig:
    """Task-end retention decisions (spec §18, §19)."""

    #: Historical current-task key audit.
    min_key_usage_rate: float = 0.02
    min_key_mean_positive_contribution: float = 0.0
    #: Candidate commit.
    min_candidate_usage_rate: float = 0.01
    min_candidate_positive_contribution: float = 0.0
    min_candidate_positive_rate: float = 0.0
    min_validation_answer_gain: float = 0.0
    redundancy_cosine_threshold: float = 0.98
    validation_gain_samples: int = 128
    #: Minimum number of experts that must survive a task, so the pool can
    #: always supply the deployed Top-2.
    min_committed_experts: int = 2

    def __post_init__(self) -> None:
        for name in (
            "min_key_usage_rate",
            "min_candidate_usage_rate",
            "min_candidate_positive_rate",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.validation_gain_samples < 1:
            raise ValueError("validation_gain_samples must be positive")
        if self.min_committed_experts < 1:
            raise ValueError("min_committed_experts must be positive")


#: The estimator behind the responsibility teacher.  The gate gradient is a
#: *local, conditional* quantity: it answers "if this gate moved, what would the
#: answer loss do?", holding the other gates and the whole backbone fixed.  It
#: is not the exact marginal contribution of an expert, which would require one
#: removal forward pass per expert per sample.
RESPONSIBILITY_GATE_GRADIENT = "gate_gradient"

#: The responsibility loss.  BCE, because the routing output is an independent
#: per-expert sigmoid and BCE is that family's proper scoring rule.  A
#: normalised divergence would compare a distribution against something that is
#: not one.
RESPONSIBILITY_LOSS_BCE = "BCE"


@dataclass(frozen=True)
class V9ResponsibilityConfig:
    """How the answer becomes a supervision target (spec §9, §10, §35)."""

    estimator: str = RESPONSIBILITY_GATE_GRADIENT
    #: Only positive contributions survive into the target: a negative estimate
    #: says "removing this expert would have helped", which is not a claim that
    #: some *other* expert should serve the sample instead.  It is evidence for
    #: reducing this expert's propensity, not evidence for raising anyone else's.
    positive_only: bool = True
    loss: str = RESPONSIBILITY_LOSS_BCE

    def __post_init__(self) -> None:
        if self.estimator != RESPONSIBILITY_GATE_GRADIENT:
            raise ValueError(
                "V9-S responsibility is the gate-gradient local conditional "
                "contribution: {} is not implemented and the exact removal "
                "quantity is calibration-only".format(self.estimator)
            )
        if not self.positive_only:
            raise ValueError(
                "responsibility must be positive-only: rows with no positive "
                "contribution carry no teacher and are excluded, rather than "
                "given an invented target"
            )
        if self.loss != RESPONSIBILITY_LOSS_BCE:
            raise ValueError(
                "the responsibility loss is BCE on the independent sigmoid "
                "gates; {} would compare a distribution against a non-"
                "distribution".format(self.loss)
            )


@dataclass(frozen=True)
class V9InferenceConfig:
    """What the deployed forward is allowed to look at (spec §35, §38)."""

    #: No task id at inference.  A run that needed the task id would be
    #: answering "which task is this?" rather than "which expert does this
    #: sample need", and the lower-triangular evaluation would be measuring the
    #: task label rather than transfer.
    task_id: bool = False
    #: V8's Global Multi-Key rule, inherited unchanged: an expert is recalled
    #: when *any* of its retained keys matches, max over that expert's key
    #: memory, then Top-K over distinct experts.
    global_multi_key: bool = True
    #: Pair reranking is the declared ablation, never the main rule.  The main
    #: experiment's deployed pair is the two highest gates.
    pair_rerank: bool = False

    def __post_init__(self) -> None:
        if self.task_id:
            raise ValueError("V9-S inference must not consume a task id")
        if not self.global_multi_key:
            raise ValueError(
                "V9-S inherits V8's Global Multi-Key inference; a run without "
                "it is not this method"
            )


@dataclass(frozen=True)
class V9ExactOracleConfig:
    """Where the exact removal quantity may appear (spec §30, §38).

    ``G_exact(k) = L(S \\ E_k) - L(S)`` needs one extra backbone forward per
    expert per sample.  It is the measurement the whole method is judged
    against, and it is also the thing that must never enter the training loop --
    a training signal derived from it would be a different, far more expensive
    method.  Both facts are stated here so a reader can check them without
    reading the trainer, and so a config that asked for anything else would fail
    to load rather than quietly train something else.
    """

    training: bool = False
    validation_calibration_only: bool = True

    def __post_init__(self) -> None:
        if self.training:
            raise ValueError(
                "the exact oracle must not appear in the training loop: it is "
                "calibration-only"
            )
        if not self.validation_calibration_only:
            raise ValueError(
                "the exact oracle is enabled only for validation calibration"
            )


@dataclass(frozen=True)
class V9Config:
    schema_version: int = 1
    method: str = V9_METHOD_NAME
    seed: int = 42
    query: V9QueryConfig = field(default_factory=V9QueryConfig)
    expert: V9ExpertConfig = field(default_factory=V9ExpertConfig)
    historical_retrieval: V9RetrievalConfig = field(
        default_factory=V9RetrievalConfig
    )
    wide_retrieval: V9WideRetrievalConfig = field(
        default_factory=V9WideRetrievalConfig
    )
    key: V9KeyConfig = field(default_factory=V9KeyConfig)
    routing: V9RoutingConfig = field(default_factory=V9RoutingConfig)
    bootstrap: V9BootstrapConfig = field(default_factory=V9BootstrapConfig)
    schedule: V9ScheduleConfig = field(default_factory=V9ScheduleConfig)
    loss: V9LossConfig = field(default_factory=V9LossConfig)
    composition: V9CompositionConfig = field(default_factory=V9CompositionConfig)
    validation: V9ValidationConfig = field(default_factory=V9ValidationConfig)
    audit: V9AuditConfig = field(default_factory=V9AuditConfig)
    responsibility: V9ResponsibilityConfig = field(
        default_factory=V9ResponsibilityConfig
    )
    inference: V9InferenceConfig = field(default_factory=V9InferenceConfig)
    exact_oracle: V9ExactOracleConfig = field(
        default_factory=V9ExactOracleConfig
    )
    SECTIONS: ClassVar[Dict[str, type]] = {}

    def __post_init__(self) -> None:
        if self.method != V9_METHOD_NAME:
            raise ValueError(f"V9 requires method: {V9_METHOD_NAME}")
        # A YAML loader hands sections over as plain dicts.  Coerce here so each
        # section's own invariants actually run; otherwise a dict would be
        # stored as-is and every guarantee below would be silently skipped.
        for name, section_type in self.SECTIONS.items():
            current = getattr(self, name)
            if isinstance(current, Mapping):
                object.__setattr__(self, name, section_type(**dict(current)))
            elif not isinstance(current, section_type):
                raise TypeError(
                    f"V9 config section {name!r} must be a {section_type.__name__} "
                    f"or a mapping, got {type(current).__name__}"
                )

    # ------------------------------------------------------------------
    # derived geometry
    # ------------------------------------------------------------------
    @property
    def candidate_count(self) -> int:
        return int(self.expert.num_current_candidates)

    @property
    def historical_slots(self) -> int:
        """Historical columns in the cached recall row.

        The cache is built once per task at the *widest* recall the task ever
        uses, so a wide step needs no rebuild: the base Top-C is a prefix of the
        wide Top-C by construction (both are the same ranking truncated at
        different lengths).
        """
        width = int(self.historical_retrieval.top_c)
        if self.wide_retrieval.enabled:
            width = max(width, int(self.wide_retrieval.top_c))
        return width

    @property
    def selection_slots(self) -> int:
        """Width of one routing row: ``historical_slots + candidates``."""
        return self.historical_slots + self.candidate_count

    def active_historical_slots(self, wide: bool) -> int:
        """How many historical columns are live on a given step."""
        if wide and self.wide_retrieval.enabled:
            return min(int(self.wide_retrieval.top_c), self.historical_slots)
        return min(int(self.historical_retrieval.top_c), self.historical_slots)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "V9Config":
        if not isinstance(value, Mapping):
            raise TypeError("V9Config.from_dict expects a mapping")
        sections: Dict[str, Any] = {}
        for name, section_type in cls.SECTIONS.items():
            raw = value.get(name)
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                raise TypeError(f"V9 config section {name!r} must be a mapping")
            sections[name] = section_type(**dict(raw))
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            method=str(value.get("method", V9_METHOD_NAME)),
            seed=int(value.get("seed", 42)),
            **sections,
        )


V9Config.SECTIONS = {
    "query": V9QueryConfig,
    "expert": V9ExpertConfig,
    "historical_retrieval": V9RetrievalConfig,
    "wide_retrieval": V9WideRetrievalConfig,
    "key": V9KeyConfig,
    "routing": V9RoutingConfig,
    "bootstrap": V9BootstrapConfig,
    "schedule": V9ScheduleConfig,
    "loss": V9LossConfig,
    "composition": V9CompositionConfig,
    "validation": V9ValidationConfig,
    "audit": V9AuditConfig,
    "responsibility": V9ResponsibilityConfig,
    "inference": V9InferenceConfig,
    "exact_oracle": V9ExactOracleConfig,
}


def load_v9_config(path: str) -> V9Config:
    """Load a V9 YAML config; the section layout mirrors the dataclass."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = V9Config.from_dict(raw)
    assert_frozen_contract(config)
    return config


def assert_frozen_contract(config: V9Config) -> None:
    """Re-assert the hard constraints the method's claim rests on.

    These are the facts a reader is entitled to assume from the config alone:
    the query is parameter-free and frozen, routing is independent per-expert
    sigmoid whose forward gates are detached from the key graph, composition is
    RMS-calibrated without a cardinality factor, the schedule ends in a
    discretisation stage, and the slot geometry is consistent with the routing
    row actually built.
    """
    if config.query.trainable_parameter_count != 0:
        raise ValueError("the query must remain parameter-free")
    if config.query.cache is not True:
        raise ValueError(
            "the fixed query must be cached: re-encoding it per epoch would "
            "let the query coordinate system drift"
        )
    if config.routing.type != ROUTING_INDEPENDENT_SIGMOID:
        raise ValueError("V9-S routing must be independent sigmoid")
    if config.routing.direct_answer_gradient_to_key:
        raise ValueError(
            "the answer must not reach a key through a direct gate gradient"
        )
    if config.composition.cardinality_scale != "none":
        raise ValueError(
            "V9-S composes h <- base + sum_k a_k * kappa_k * u_k; a cardinality "
            "factor would scale experts by retrieval width"
        )
    if config.schedule.hard_ratio <= 0:
        raise ValueError("the discretisation stage is required")
    if config.key.aggregation != AGGREGATION_MAX:
        raise ValueError("V9-S inference keeps max-per-expert multi-key aggregation")
    if config.historical_retrieval.aggregation != HISTORICAL_AGGREGATION_MULTI_KEY_MAX:
        raise ValueError("V9-S keeps V8's historical recall geometry")
    if config.selection_slots < 1:
        raise ValueError("the V9-S candidate set must be non-empty")
    if config.wide_retrieval.enabled:
        if config.wide_retrieval.top_c < int(config.historical_retrieval.top_c):
            raise ValueError(
                "wide_retrieval.top_c ({}) is narrower than "
                "historical_retrieval.top_c ({}): a wide step would recall "
                "fewer experts than a normal one".format(
                    config.wide_retrieval.top_c,
                    config.historical_retrieval.top_c,
                )
            )
    if config.inference.pair_rerank and not config.validation.contribution_calibration:
        raise ValueError(
            "pair_rerank without the calibration pass would declare a "
            "comparison the run never makes: the only place V9-S can afford a "
            "per-pair answer forward is the bounded held-out calibration"
        )


__all__ = [
    "AGGREGATION_MAX",
    "CANDIDATE_INIT_KMEANS",
    "CANDIDATE_INIT_PERTURBED_MEAN",
    "CANDIDATE_INITS",
    "HISTORICAL_AGGREGATION_MULTI_KEY_MAX",
    "ROUTING_INDEPENDENT_SIGMOID",
    "V9_METHOD_NAME",
    "V9_METHOD_TITLE",
    "V9_QUERY_TYPE",
    "V9AuditConfig",
    "V9BootstrapConfig",
    "V9CompositionConfig",
    "V9Config",
    "V9ExpertConfig",
    "V9KeyConfig",
    "V9LossConfig",
    "V9QueryConfig",
    "V9RetrievalConfig",
    "V9RoutingConfig",
    "V9ScheduleConfig",
    "V9ValidationConfig",
    "V9WideRetrievalConfig",
    "assert_frozen_contract",
    "load_v9_config",
]
