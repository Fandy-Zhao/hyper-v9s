"""``V8-Exact-Accelerated`` execution mode: declaration, application, guard.

The mode exists to make one promise checkable: **every flag it turns on is an
execution change and nothing else.**  The produced numbers -- losses, routes,
gradients, weights -- must be identical to the frozen baseline, and the
baseline itself must still run when the flags are off.

This module is the enforcement point.

* :data:`EXECUTION_FLAGS` is the complete set of legal flags, each with the
  argument it sets and the ablation step it implements (S1..S11).
* :data:`PROHIBITED_KEYS` is a fail-closed deny-list.  Any of the twelve
  accelerations the V8 brief forbids -- residual-only training, candidate
  truncation before the answer-NLL oracle, teacher freezing, reuse-sample
  subsampling, cross-epoch route freezing, recipe changes, approximate NLL
  proxies -- is rejected by name, so a config file cannot smuggle one in.
* :data:`RECIPE_INVARIANTS` are the recipe knobs that must equal the frozen
  baseline.  :func:`assert_recipe_invariants` compares the *resolved*
  ``TrainingArguments`` (after the config and the command line are merged), so
  an unlisted flag that quietly moves the global batch is caught too.

Load with ``--compose_v8_config``; see ``configs/v8_exact_accelerated.yaml``.
"""

import os
from typing import Dict, Mapping, Optional, Sequence

#: ``name -> (ModelArguments attribute, kind, description)``.
#: ``kind`` tells the applier what to do with the YAML value.
EXECUTION_FLAGS: Dict[str, tuple] = {
    # S1 -- query cache normalisation.
    "cache_queries": (
        "compose_v7_query_tensor",
        "path",
        "Load the fixed 1536-D queries from the pipeline's queries.pt split "
        "artefact (mmap, fingerprint-checked) instead of parsing the ~1.3 GB "
        "per-sample JSON. Bit-identical; verified by "
        "compose.experiments.verify_query_tensor.",
    ),
    # S5 -- remove the per-layer device synchronisation.
    "compose_selection_plan": (
        "compose_selection_plan",
        "bool",
        "Share one selection decomposition across all ComposeLinear layers per "
        "micro-step instead of recomputing it (with ~3 device syncs) per layer.",
    ),
    # Profiling (P0) -- measurement only, never part of the trained recipe.
    "profile_training": ("profile_training", "bool", "Write profile_steps*.jsonl."),
    "profile_sync": (
        "profile_sync",
        "bool",
        "Sync-pair the coarse phase boundaries. True gives attribution, False "
        "gives a low-disturbance wall clock; neither changes the maths.",
    ),
    "profile_flush_every": ("profile_flush_every", "int", "JSONL flush cadence."),
    # Recipe-preserving runtime knobs (S6/S8): these are legal *only* through
    # the invariant check below, which proves the effective batch is unchanged.
    "effective_batch": (
        "effective_batch",
        "derived",
        "The product micro-batch x accumulation x world size. Declare it and "
        "leave gradient_accumulation_steps out, and accumulation is derived for "
        "whatever world size the run actually uses -- a config that hard-codes "
        "accumulation is only correct on one world size.",
    ),
    "micro_batch_size": (
        "per_device_train_batch_size",
        "train:int",
        "Micro-batch width; the scheduling knob that S6 sweeps. Accumulation "
        "moves inversely so the effective batch stays at the declared value.",
    ),
    "gradient_accumulation_steps": (
        "gradient_accumulation_steps",
        "train:int",
        "Accumulation steps, explicit. Optional: prefer ``effective_batch``, "
        "which stays correct across world sizes. If both are given they must "
        "agree.",
    ),
    "length_bucketing": (
        "group_by_modality_length",
        "train:bool",
        "Length-grouped sampler (the frozen baseline already runs it; the flag "
        "exists so an intentionally un-bucketed control run is possible).",
    ),
    "attn_implementation": (
        "compose_attn_implementation",
        "str",
        "Attention kernel ('' = transformers' default, 'eager', "
        "'sdpa', 'flash_attention_2'). Numerically validated against the eager "
        "path before it may be enabled; see the equivalence report.",
    ),
}

#: Names the brief uses for capabilities this repository names differently.
#: These are deliberately **not** accepted as flags -- the mapping exists so that
#: an unknown-flag rejection can point at the flag that does the job rather than
#: leaving the reader to conclude the capability was forgotten.  Accepting a
#: ``use_flash_attention`` alias, for instance, would invite enabling a kernel
#: that the attention sweep measured as not worth it here.
FLAG_ALIASES: Dict[str, str] = {
    "use_flash_attention": "attn_implementation",
    "microbatch_autotune": "micro_batch_size",
}

#: Fail-closed deny-list: the twelve prohibited accelerations, by name.
PROHIBITED_KEYS: Dict[str, str] = {
    "residual_only_training": "brief §12.1 (1): residual-only LoRA training",
    "residual_only": "brief §12.1 (1): residual-only LoRA training",
    "skip_current_expert": "brief §12.1 (1)(2): dropping Current-Expert competition",
    "skip_current_expert_on_reuse": "brief §12.1 (2): old-only samples lose competition",
    "freeze_teacher": "brief §12.1 (3): freezing the dynamic teacher",
    "teacher_refresh_interval": "brief §12.1 (4): lowering the teacher/NLL refresh rate",
    "teacher_refresh_every": "brief §12.1 (4): lowering the teacher/NLL refresh rate",
    "truncate_candidates": "brief §12.1 (5): Key Top-M truncation before the oracle",
    "candidate_top_m": "brief §12.1 (5): changing the candidate limit",
    "subsample_reuse": "brief §12.1 (6): subsampling high-confidence reuse samples",
    "reuse_subsample_ratio": "brief §12.1 (6): subsampling high-confidence reuse samples",
    "freeze_routes_across_epochs": "brief §12.1 (7): cross-epoch route freezing",
    "inflate_key_batch": "brief §12.1 (8): inflating the Key training batch",
    "approx_nll": "brief §12.1 (11): approximate NLL predictors/proxies",
    "nll_proxy": "brief §12.1 (11): approximate NLL predictors/proxies",
    "skip_end_of_task_lifecycle": "brief §12.1 (12): candidate lifecycle changes",
    "drop_early_stopping": "brief §12.1 (12): candidate lifecycle changes",
}

#: Frozen recipe values.  Keys are ``TrainingArguments`` attribute names; the
#: values are the baseline recipe's.  ``None`` means "not checked here".
RECIPE_INVARIANTS: Mapping[str, object] = {
    "learning_rate": 0.0002,
    "num_train_epochs": 1.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.0,
    "seed": 42,
    "model_max_length": 2048,
    "bf16": True,
    "tf32": True,
    "remove_unused_columns": False,
    "gradient_checkpointing": True,
}

#: Effective (global) batch the frozen baseline trains with.
BASELINE_EFFECTIVE_BATCH = 64


class ProhibitedFlagError(ValueError):
    """Raised when a config requests an acceleration the brief forbids."""


def load_config(path: str) -> Dict[str, object]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("V8 accelerated config must be a mapping: {}".format(path))
    return payload


def check_no_prohibited_keys(payload: Mapping[str, object], path: str = "<config>") -> None:
    """Reject a config that asks for a forbidden acceleration, by name."""
    found = []
    for key in _walk_keys(payload):
        if key in PROHIBITED_KEYS:
            found.append("{} ({})".format(key, PROHIBITED_KEYS[key]))
    if found:
        raise ProhibitedFlagError(
            "{} requests accelerations the V8 brief prohibits: {}".format(
                path, "; ".join(sorted(found))
            )
        )


def _walk_keys(node) -> list:
    keys = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            keys.append(str(key))
            keys.extend(_walk_keys(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            keys.extend(_walk_keys(value))
    return keys


def _apply_flag(name: str, value, model_args, training_args, path: str) -> Dict[str, object]:
    attribute, kind, _description = EXECUTION_FLAGS[name]
    if kind == "path":
        if value in (None, "", False):
            return {}
        resolved = os.path.abspath(str(value))
        setattr(model_args, attribute, resolved)
        return {attribute: resolved}
    if kind == "bool":
        resolved = bool(value)
        setattr(model_args, attribute, resolved)
        return {attribute: resolved}
    if kind == "int":
        resolved = int(value)
        setattr(model_args, attribute, resolved)
        return {attribute: resolved}
    if kind == "str":
        resolved = "" if value is None else str(value)
        setattr(model_args, attribute, resolved)
        return {attribute: resolved}
    if kind.startswith("train:"):
        resolved = value
        if kind == "train:int":
            resolved = int(value)
        elif kind == "train:bool":
            resolved = bool(value)
        setattr(training_args, attribute, resolved)
        return {attribute: resolved}
    if kind == "derived":
        # Recorded, not applied: it feeds the accumulation derivation below.
        return {"{}_declared".format(name): int(value)}
    raise ValueError("unknown flag kind {} for {}".format(kind, name))


def _derive_accumulation(
    declared: Mapping[str, object],
    training_args,
    world_size: int,
    path: str,
) -> Dict[str, object]:
    """Turn ``effective_batch`` + ``micro_batch_size`` into an accumulation.

    Accumulation is the only one of the three that depends on how many GPUs the
    run happens to use, so it is the one the config should not be stating.  A
    config that hard-codes ``accumulation: 32`` is correct at world size 2 and
    silently wrong at any other size -- the failure mode the whole invariant
    gate exists to prevent.  Declaring the effective batch instead keeps one
    file correct on one, two or four ranks.
    """
    declared_batch = declared.get("effective_batch_declared")
    if declared_batch is None:
        return {}
    explicit = declared.get("gradient_accumulation_steps")
    micro = int(training_args.per_device_train_batch_size)
    denominator = micro * int(world_size)
    if denominator <= 0:
        raise ValueError("{}: micro-batch {} x world size {}".format(path, micro, world_size))
    if declared_batch % denominator:
        raise ValueError(
            "{}: effective_batch {} is not divisible by micro-batch {} x world "
            "size {} = {}".format(path, declared_batch, micro, world_size, denominator)
        )
    accumulation = declared_batch // denominator
    if explicit is not None and int(explicit) != accumulation:
        raise ValueError(
            "{}: gradient_accumulation_steps {} contradicts effective_batch {} at "
            "world size {} (micro-batch {} needs {}). Remove one of them.".format(
                path, explicit, declared_batch, world_size, micro, accumulation
            )
        )
    training_args.gradient_accumulation_steps = accumulation
    return {
        "gradient_accumulation_steps": accumulation,
        "effective_batch": micro * accumulation * int(world_size),
        "world_size": int(world_size),
    }


def _unknown_flag_message(
    unknown: Sequence[str], payload: Mapping[str, object], path: str
) -> str:
    """Reject unknown flags, but say what this repository does instead.

    A fail-closed rejection is only useful if it is actionable.  The brief names
    several capabilities this repository deliberately does not implement -- the
    token/label cache, the historical-NLL cache, the vectorised expert
    evaluation -- and the measured reason for each is already written in the same
    config file's ``not_applicable:`` block.  Quoting it here keeps a reader from
    concluding the flag was forgotten, and keeps the fix (delete the flag) local
    rather than requiring a hunt through the reports.
    """
    lines = [
        "{}: unknown execution flags {} (legal: {})".format(
            path, list(unknown), sorted(EXECUTION_FLAGS)
        )
    ]
    documented = payload.get("not_applicable", {})
    if not isinstance(documented, Mapping):
        documented = {}
    for name in unknown:
        entry = documented.get(name)
        reason = entry.get("reason") if isinstance(entry, Mapping) else entry
        if reason:
            # One line: the reasons in the shipped config are multi-line prose.
            lines.append(
                "  {}: documented as not applicable -- {}".format(
                    name, " ".join(str(reason).split())
                )
            )
        elif name in FLAG_ALIASES:
            lines.append(
                "  {}: this repository spells it {} -- see EXECUTION_FLAGS".format(
                    name, FLAG_ALIASES[name]
                )
            )
    return "\n".join(lines)


def apply_config(
    payload: Mapping[str, object],
    model_args,
    training_args,
    path: str = "<config>",
    world_size: Optional[int] = None,
) -> Dict[str, object]:
    """Apply a V8 accelerated config in place; return the resolved flag values.

    Only the ``flags:`` block is applied.  ``not_applicable:`` is documentation
    of what this repository deliberately does *not* do and is validated for
    prohibited content but otherwise ignored.

    ``world_size`` is required as soon as the config declares an
    ``effective_batch`` or a ``micro_batch_size``, because accumulation is
    derived from it -- see :func:`_derive_accumulation`.  When omitted it falls
    back to ``training_args.world_size``, which is only trustworthy once the
    Trainer exists.
    """
    check_no_prohibited_keys(payload, path)
    flags = payload.get("flags", {})
    if not isinstance(flags, Mapping):
        raise ValueError("{}: 'flags' must be a mapping".format(path))
    unknown = sorted(set(flags) - set(EXECUTION_FLAGS))
    if unknown:
        raise ValueError(_unknown_flag_message(unknown, payload, path))
    resolved: Dict[str, object] = {}
    for name, value in flags.items():
        resolved.update(_apply_flag(name, value, model_args, training_args, path))
    size = int(world_size or getattr(training_args, "world_size", 0) or 1)
    resolved.update(_derive_accumulation(resolved, training_args, size, path))
    resolved.pop("effective_batch_declared", None)
    return resolved


def assert_recipe_invariants(training_args, model_args=None, world_size=None) -> Dict[str, object]:
    """Prove the resolved run still trains the frozen recipe.

    Returns the observed invariants so the caller can log them; raises
    :class:`ValueError` on the first violation.  This is the gate that makes
    "same recipe, different execution" a checked property rather than a claim.
    """
    observed: Dict[str, object] = {}
    violations = []
    for name, expected in RECIPE_INVARIANTS.items():
        actual = getattr(training_args, name, None)
        observed[name] = actual
        if expected is None:
            continue
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - float(expected)) > 1e-12:
                violations.append("{}={!r} (baseline {!r})".format(name, actual, expected))
        elif actual != expected:
            violations.append("{}={!r} (baseline {!r})".format(name, actual, expected))

    # Explicit world size wins over ``TrainingArguments.world_size``: the latter
    # is a property over the accelerator's distributed state, which may still be
    # uninitialised at the point this runs (before the Trainer is built), in
    # which case it reports 1 and would make the effective-batch guard fire
    # spuriously on a correctly-configured multi-process run.
    size = int(world_size or getattr(training_args, "world_size", 0) or 1)
    effective = (
        int(training_args.per_device_train_batch_size)
        * int(training_args.gradient_accumulation_steps)
        * size
    )
    observed["effective_batch"] = effective
    observed["world_size"] = size
    if effective != BASELINE_EFFECTIVE_BATCH:
        violations.append(
            "effective batch {} != baseline {}".format(effective, BASELINE_EFFECTIVE_BATCH)
        )

    if model_args is not None:
        for name in ("compose_rank", "compose_alpha"):
            observed[name] = getattr(model_args, name, None)

    if violations:
        raise ValueError(
            "V8-Exact-Accelerated recipe violated -- these knobs are frozen by "
            "the brief: " + "; ".join(violations)
        )
    return observed


def default_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs",
        "v8_exact_accelerated.yaml",
    )
