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

#: The one formal V8 method name.  It is written into the run contract, the
#: screening artifact and the committed pool manifest, so a reader can always
#: tell a teacher-screened run from the retired full-oracle one.
V8_METHOD_NAME = "v8_teacher_screened_global_top2"

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


def _section_kwargs(name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a raw section dict to the current dataclass field names."""
    if name == "audit" and "recall_ks" in raw:
        raw["recall_ks"] = tuple(raw["recall_ks"])
    return raw


@dataclass(frozen=True)
class V8Config:
    schema_version: int = 1
    method: str = V8_METHOD_NAME
    seed: int = 42
    query: V8QueryConfig = field(default_factory=V8QueryConfig)
    teacher: V8TeacherConfig = field(default_factory=V8TeacherConfig)
    routing: V8RoutingConfig = field(default_factory=V8RoutingConfig)
    audit: V8AuditConfig = field(default_factory=V8AuditConfig)

    #: Section name -> dataclass, used both by ``from_dict`` and by the
    #: coercion in ``__post_init__``.  Keeping one table means a plain dict
    #: section can never slip through unvalidated.
    SECTIONS: ClassVar[Dict[str, type]] = {}

    def __post_init__(self) -> None:
        if self.method != V8_METHOD_NAME:
            raise ValueError(f"V8 requires method: {V8_METHOD_NAME}")
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
            method=str(value.get("method", V8_METHOD_NAME)),
            seed=int(value.get("seed", 42)),
            **sections,
        )


V8Config.SECTIONS = {
    "query": V8QueryConfig,
    "teacher": V8TeacherConfig,
    "routing": V8RoutingConfig,
    "audit": V8AuditConfig,
}


def assert_frozen_contract(config: V8Config) -> None:
    """Re-assert the method's hard constraints.

    These are the facts the formal V8 claim rests on.  Historical LoRA and the
    keys committed by earlier tasks are frozen; ``solved`` is decided by the
    task metric and never by an NLL threshold; the query is parameter-free.  The
    full-data stage freezes historical parameters through the V7 trainer's own
    audit (``compose/v7/hf_trainer.py``), not through a switch here.
    """
    if config.teacher.nll_use_as_solved_threshold is not False:
        raise ValueError("NLL must never define solved")
    if config.teacher.search_mode != TEACHER_SEARCH_FULL_HISTORY:
        raise ValueError("the teacher must score every visible historical expert")
    if config.query.trainable_parameter_count != 0:
        raise ValueError("the query must remain parameter-free")
    if config.routing.expert_aggregation != "max" or not config.routing.distinct_expert_topk:
        raise ValueError(
            "routing is max-per-expert then distinct-expert Top-K; key-level "
            "Top-K followed by de-duplication is forbidden"
        )


__all__ = [
    "KEY_TYPES",
    "KEY_TYPE_ORIGIN",
    "KEY_TYPE_TASK_ALIAS",
    "PAIR_SEARCH_BOUNDED",
    "PAIR_SEARCH_EXHAUSTIVE",
    "STATE_BASE_ONLY",
    "STATE_RESIDUAL",
    "STATE_REUSE1",
    "STATE_REUSE2",
    "TARGETS",
    "TARGET_CONTEXT_POSITIVE",
    "TARGET_IGNORE",
    "TARGET_NEGATIVE",
    "TARGET_POSITIVE",
    "TARGET_SOLVER_POSITIVE",
    "TEACHER_SEARCH_FULL_HISTORY",
    "TEACHER_STATES",
    "V7_QUERY_MODULE_HASH",
    "V8AuditConfig",
    "V8Config",
    "V8QueryConfig",
    "V8RoutingConfig",
    "V8TeacherConfig",
    "assert_frozen_contract",
]
