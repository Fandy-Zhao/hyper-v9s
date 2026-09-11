"""Explicit V8 configuration.

Every knob named in the V8 specification (PART 28) is a field here so that no
threshold is a magic number scattered through the implementation.  Field
defaults are the specification defaults; the experiment configs override them.

The configuration is frozen and round-trips through ``to_dict``/``from_dict``
exactly like ``compose.v7.config.V7Config`` so it can be written into every
checkpoint and teacher-cache provenance record.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple


#: The V8 query is the *same object* as V7's.  The hash is copied verbatim so a
#: cache produced by either method is provably over the same coordinate system.
V7_QUERY_MODULE_HASH = "v7_fixed_layernorm_concat_l2_v1"

KEY_TYPE_ORIGIN = "origin"
KEY_TYPE_TASK_ALIAS = "task_alias"
KEY_TYPES = (KEY_TYPE_ORIGIN, KEY_TYPE_TASK_ALIAS)

#: Sample teacher states (PART 16).
STATE_BASE_ONLY = "BaseOnly"
STATE_REUSE1 = "Reuse1"
STATE_REUSE2 = "Reuse2"
STATE_RESIDUAL = "Residual"
TEACHER_STATES = (STATE_BASE_ONLY, STATE_REUSE1, STATE_REUSE2, STATE_RESIDUAL)

#: Key-target roles.  An alternative solved expert is IGNORE: it must never be
#: pushed away from a query it can actually solve.
#:
#: SOLVER_POSITIVE and CONTEXT_POSITIVE are *both* attraction signals but rest on
#: different evidence, and the whole point of keeping them apart is that
#: "not independently solved" is not the same statement as "not useful as
#: composition context".  A Residual sample's context expert must therefore
#: attract its alias key rather than repel it -- the reasoning that makes an
#: unsolved single a legitimate NEGATIVE on a Reuse1 sample does not transfer.
#:
#: ``TARGET_POSITIVE`` keeps its historical string value ``"positive"`` so that
#: teacher records written before this distinction existed stay readable.
TARGET_SOLVER_POSITIVE = "positive"
TARGET_POSITIVE = TARGET_SOLVER_POSITIVE
TARGET_CONTEXT_POSITIVE = "context_positive"
TARGET_NEGATIVE = "negative"
TARGET_IGNORE = "ignore"
TARGETS = (TARGET_SOLVER_POSITIVE, TARGET_CONTEXT_POSITIVE, TARGET_NEGATIVE, TARGET_IGNORE)


@dataclass(frozen=True)
class V8QueryConfig:
    """The fixed query is reused, never re-derived (PART 3.1)."""

    type: str = "fixed_concat"
    source_module_hash: str = V7_QUERY_MODULE_HASH
    visual_dim: int = 768
    text_dim: int = 768
    query_dim: int = 1536
    trainable_parameter_count: int = 0

    def __post_init__(self) -> None:
        if self.type != "fixed_concat":
            raise ValueError("V8 keeps the V7 fixed query: type must be fixed_concat")
        if self.source_module_hash != V7_QUERY_MODULE_HASH:
            raise ValueError("V8 query module hash must match the V7 fixed query")
        if (self.visual_dim, self.text_dim, self.query_dim) != (768, 768, 1536):
            raise ValueError("V8 query dimensions must be 768 + 768 = 1536")
        if self.trainable_parameter_count != 0:
            raise ValueError("V8 query must have zero trainable parameters")


#: The teacher may only be a Capability Discovery Oracle: the answer supervision
#: it hands out must not be limited by what the current keys happen to recall.
#: A bounded-recall teacher makes key quality a *precondition* for capability
#: discovery, so a capable expert that the origin keys rank badly is never
#: scored, never becomes a solver, and never earns the alias key that would have
#: fixed its ranking -- the failure is invisible and self-confirming.
TEACHER_SEARCH_FULL_HISTORY = "full_history_single_oracle"

#: Bounded pair composition search over the K_s best historical singles, or a
#: strictly exhaustive sweep over every visible historical pair.  Bounded is the
#: default because it is the established budget; exhaustive is opt-in because it
#: is C(H,2) generations per pending sample.
PAIR_SEARCH_BOUNDED = "bounded"
PAIR_SEARCH_EXHAUSTIVE = "exhaustive"


@dataclass(frozen=True)
class V8TeacherConfig:
    use_base_first: bool = True
    #: Which experts the teacher is allowed to score as singles.  Only the
    #: full-history oracle is implemented: key recall is a diagnostic here, never
    #: a filter on who receives answer supervision.
    search_mode: str = TEACHER_SEARCH_FULL_HISTORY
    pair_search_mode: str = PAIR_SEARCH_BOUNDED
    #: DIAGNOSTIC ONLY.  The router's Top-M is still computed and recorded (it
    #: is the quantity the keys are judged by), but it no longer decides which
    #: experts the teacher tests, which states are assigned, which experts get
    #: alias support, or which samples the candidate trains on.
    historical_top_m: int = 8
    pair_top_k_single: int = 4
    max_pairs: int = 6
    solved_signal: str = "task_metric"
    #: NLL is a ranking / confidence / residual-context signal ONLY.
    #: ``use_as_solved_threshold`` must stay False: V8 never defines "solved"
    #: by an NLL threshold (PART 8.1, PART 46 item 1).
    nll_use_for_ranking: bool = True
    nll_use_for_confidence: bool = True
    nll_use_for_residual_context: bool = True
    nll_use_as_solved_threshold: bool = False
    #: Minimum task-metric gain for a pair to count as a real marginal
    #: contribution (PART 14).  Task-specific, never a global NLL threshold.
    pair_min_metric_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.nll_use_as_solved_threshold:
            raise ValueError(
                "V8 forbids an NLL solved threshold: capability is decided by "
                "task metric correctness only (PART 8.1 / PART 46 item 1)"
            )
        if self.solved_signal != "task_metric":
            raise ValueError("V8 solved signal must be task_metric")
        if self.use_base_first is not True:
            raise ValueError("V8 requires base-first minimal-capacity search")
        if self.search_mode != TEACHER_SEARCH_FULL_HISTORY:
            raise ValueError(
                "the training teacher must be a full-history capability oracle: "
                "key recall may not limit which experts receive answer "
                "supervision (DECISION-1)"
            )
        if self.pair_search_mode not in (PAIR_SEARCH_BOUNDED, PAIR_SEARCH_EXHAUSTIVE):
            raise ValueError(
                f"pair_search_mode must be {PAIR_SEARCH_BOUNDED!r} or "
                f"{PAIR_SEARCH_EXHAUSTIVE!r}"
            )
        if self.historical_top_m < 1:
            raise ValueError("historical_top_m must be positive")
        if self.pair_top_k_single < 2:
            raise ValueError("pair_top_k_single must be at least 2")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be positive")

    @property
    def full_history_singles(self) -> bool:
        """True when every visible historical expert is scored as a single."""
        return self.search_mode == TEACHER_SEARCH_FULL_HISTORY

    def pair_budget(self, candidate_count: int) -> int:
        """Number of pairs actually enumerated over the shortlist."""
        if self.pair_search_mode == PAIR_SEARCH_EXHAUSTIVE:
            count = max(int(candidate_count), 0)
            return count * (count - 1) // 2
        shortlist = min(int(candidate_count), int(self.pair_top_k_single))
        return min(shortlist * (shortlist - 1) // 2, int(self.max_pairs))


@dataclass(frozen=True)
class V8ExpertConfig:
    historical_lora_frozen: bool = True
    historical_key_frozen: bool = True
    candidate_rank: int = 8
    candidate_alpha: float = 16.0
    candidate_count: int = 4

    def __post_init__(self) -> None:
        if not self.historical_lora_frozen:
            raise ValueError("V8 freezes historical LoRA in its first version")
        if not self.historical_key_frozen:
            raise ValueError("V8 freezes historical committed keys")
        if self.candidate_count != 4 or self.candidate_rank != 8:
            raise ValueError("V8 keeps the V7 four rank-8 candidate recipe")


@dataclass(frozen=True)
class V8KeyConfig:
    mode: str = "multi_key"
    alias_key_trainable: bool = True
    lazy_alias_creation: bool = True
    positive_loss: str = "cosine"
    ranking_loss: str = "margin"
    alternative_solved_policy: str = "ignore"
    #: ``L_pos`` splits by evidence source: an expert the teacher selected solved
    #: the sample, an expert it kept as Residual context did not.  Both attract
    #: the alias key; the two weights are separate so an ablation can tell the
    #: two contributions apart without a code change.  V8-v1 runs them equal.
    lambda_solver_positive: float = 1.0
    lambda_context_positive: float = 1.0
    lambda_rank: float = 0.1
    ranking_margin: float = 0.2
    learning_rate: float = 3.0e-4

    @property
    def lambda_pos(self) -> float:
        """Legacy name for the solver-positive weight (pre-oracle semantics)."""
        return self.lambda_solver_positive

    def __post_init__(self) -> None:
        if self.mode != "multi_key":
            raise ValueError("V8 key mode must be multi_key")
        if not self.alias_key_trainable:
            raise ValueError("V8 must be able to train current-task alias keys")
        if not self.lazy_alias_creation:
            raise ValueError(
                "V8 creates alias keys lazily from teacher positives only; "
                "unconditional per-expert creation is forbidden (PART 46 item 7)"
            )
        if self.alternative_solved_policy != "ignore":
            raise ValueError(
                "an alternative solved expert must be ignored, never a negative"
            )
        if self.positive_loss != "cosine":
            raise ValueError("V8 v1 positive loss is cosine attraction")
        if self.ranking_loss != "margin":
            raise ValueError("V8 v1 ranking loss is a margin loss")
        if self.lambda_solver_positive < 0 or self.lambda_rank < 0:
            raise ValueError("key loss weights must be non-negative")
        if self.lambda_context_positive < 0:
            raise ValueError("key loss weights must be non-negative")
        if self.ranking_margin < 0:
            raise ValueError("ranking margin must be non-negative")


@dataclass(frozen=True)
class V8ResidualConfig:
    sample_filtering: bool = False
    gradient_gating: bool = True
    use_historical_context: bool = True

    def __post_init__(self) -> None:
        if self.sample_filtering:
            raise ValueError(
                "V8 never builds a residual-only dataset; every sample gets a "
                "reuse decision and only gradients are gated (PART 21)"
            )
        if not self.gradient_gating:
            raise ValueError("V8 requires per-sample gradient gating")


@dataclass(frozen=True)
class V8RoutingConfig:
    key_similarity: str = "cosine"
    expert_aggregation: str = "max"
    distinct_expert_topk: bool = True
    top_k: int = 2

    def __post_init__(self) -> None:
        if self.key_similarity != "cosine":
            raise ValueError("V8 key similarity is cosine")
        if self.expert_aggregation != "max":
            raise ValueError(
                "V8 aggregates all keys of an expert with max before Top-K"
            )
        if not self.distinct_expert_topk:
            raise ValueError(
                "one expert must not occupy two Top-K slots (PART 6)"
            )
        if self.top_k != 2:
            raise ValueError("V8 keeps the V7 Top-2 expert budget")


@dataclass(frozen=True)
class V8AuditConfig:
    full_pool_single_audit: bool = True
    full_pool_audit_samples: int = 128
    recall_ks: Tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        if not self.full_pool_single_audit:
            raise ValueError(
                "V8 must audit full-pool recall before concluding that an old "
                "expert has no capability (PART 26 / PART 46 item 17)"
            )
        if any(k < 1 for k in self.recall_ks):
            raise ValueError("recall ks must be positive")


@dataclass(frozen=True)
class V8PruningConfig:
    alias_support_threshold: int = 1
    alias_gain_threshold: float = 0.0
    alias_redundancy_threshold: float = 0.995
    candidate_prune_enabled: bool = True
    candidate_min_removal_gain: float = 0.0
    candidate_redundancy_cosine: float = 0.98

    def __post_init__(self) -> None:
        if self.alias_support_threshold < 1:
            raise ValueError("alias_support_threshold must be at least 1")
        if not 0.0 <= self.alias_redundancy_threshold <= 1.0:
            raise ValueError("alias_redundancy_threshold must be in [0, 1]")
        if not 0.0 <= self.candidate_redundancy_cosine <= 1.0:
            raise ValueError("candidate_redundancy_cosine must be in [0, 1]")


@dataclass(frozen=True)
class V8TrainingConfig:
    lambda_key: float = 0.1
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    seed: int = 42
    bf16: bool = True
    tf32: bool = True
    model_max_length: int = 2048

    def __post_init__(self) -> None:
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


#: Renamed fields, so a config dict written by an earlier revision still loads
#: instead of raising ``TypeError`` on an unexpected keyword.  Value is
#: ``{section: {old_name: new_name}}``.
_RENAMED_SECTION_FIELDS: Dict[str, Dict[str, str]] = {
    "key": {"lambda_pos": "lambda_solver_positive"},
}


def _section_kwargs(name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a raw section dict to the current dataclass field names."""
    if name == "audit" and "recall_ks" in raw:
        raw["recall_ks"] = tuple(raw["recall_ks"])
    for old, new in _RENAMED_SECTION_FIELDS.get(name, {}).items():
        if old in raw:
            raw.setdefault(new, raw.pop(old))
    return raw


@dataclass(frozen=True)
class V8Config:
    schema_version: int = 1
    method: str = "v8_answer_supervised_multikey"
    seed: int = 42
    query: V8QueryConfig = field(default_factory=V8QueryConfig)
    teacher: V8TeacherConfig = field(default_factory=V8TeacherConfig)
    expert: V8ExpertConfig = field(default_factory=V8ExpertConfig)
    key: V8KeyConfig = field(default_factory=V8KeyConfig)
    residual: V8ResidualConfig = field(default_factory=V8ResidualConfig)
    routing: V8RoutingConfig = field(default_factory=V8RoutingConfig)
    audit: V8AuditConfig = field(default_factory=V8AuditConfig)
    pruning: V8PruningConfig = field(default_factory=V8PruningConfig)
    training: V8TrainingConfig = field(default_factory=V8TrainingConfig)

    #: Section name -> dataclass, used both by ``from_dict`` and by the
    #: coercion in ``__post_init__``.  Keeping one table means a plain dict
    #: section can never slip through unvalidated.
    SECTIONS: ClassVar[Dict[str, type]] = {}

    def __post_init__(self) -> None:
        if self.method != "v8_answer_supervised_multikey":
            raise ValueError("V8 requires method: v8_answer_supervised_multikey")
        # A caller may pass a plain dict for a section (that is what the YAML
        # loader produces).  Coerce it to its dataclass here so the section's
        # own invariants actually run; otherwise a dict would be stored as-is
        # and every guarantee below would be silently skipped.
        for name, section_type in self.SECTIONS.items():
            current = getattr(self, name)
            if isinstance(current, Mapping):
                object.__setattr__(
                    self, name, section_type(**_section_kwargs(name, dict(current)))
                )
            elif not isinstance(current, section_type):
                raise TypeError(
                    f"V8 config section {name!r} must be a {section_type.__name__} "
                    f"or a mapping, got {type(current).__name__}"
                )
        assert_frozen_contract(self)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["audit"]["recall_ks"] = list(self.audit.recall_ks)
        return payload

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "V8Config":
        sections: Dict[str, Any] = {}
        for name, section in cls.SECTIONS.items():
            sections[name] = section(
                **_section_kwargs(name, dict(value.get(name, {}) or {}))
            )
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            method=str(value.get("method", "v8_answer_supervised_multikey")),
            seed=int(value.get("seed", 42)),
            **sections,
        )


V8Config.SECTIONS = {
    "query": V8QueryConfig,
    "teacher": V8TeacherConfig,
    "expert": V8ExpertConfig,
    "key": V8KeyConfig,
    "residual": V8ResidualConfig,
    "routing": V8RoutingConfig,
    "audit": V8AuditConfig,
    "pruning": V8PruningConfig,
    "training": V8TrainingConfig,
}


def assert_frozen_contract(config: V8Config) -> None:
    """Re-assert the hard PART 4 constraints; used by the trainer and tests."""
    if config.expert.historical_lora_frozen is not True:
        raise ValueError("historical LoRA must be frozen")
    if config.expert.historical_key_frozen is not True:
        raise ValueError("historical committed keys must be frozen")
    if config.teacher.nll_use_as_solved_threshold is not False:
        raise ValueError("NLL must never define solved")
    if config.residual.sample_filtering is not False:
        raise ValueError("no residual sample filtering is permitted")
    if config.query.trainable_parameter_count != 0:
        raise ValueError("the query must remain parameter-free")


__all__ = [
    "KEY_TYPE_ORIGIN",
    "KEY_TYPE_TASK_ALIAS",
    "KEY_TYPES",
    "STATE_BASE_ONLY",
    "STATE_REUSE1",
    "STATE_REUSE2",
    "PAIR_SEARCH_BOUNDED",
    "PAIR_SEARCH_EXHAUSTIVE",
    "STATE_RESIDUAL",
    "STATE_REUSE1",
    "STATE_REUSE2",
    "TARGET_CONTEXT_POSITIVE",
    "TARGET_IGNORE",
    "TARGET_NEGATIVE",
    "TARGET_POSITIVE",
    "TARGET_SOLVER_POSITIVE",
    "TARGETS",
    "TEACHER_SEARCH_FULL_HISTORY",
    "TEACHER_STATES",
    "V7_QUERY_MODULE_HASH",
    "V8AuditConfig",
    "V8Config",
    "V8ExpertConfig",
    "V8KeyConfig",
    "V8PruningConfig",
    "V8QueryConfig",
    "V8ResidualConfig",
    "V8RoutingConfig",
    "V8TeacherConfig",
    "V8TrainingConfig",
    "assert_frozen_contract",
]
