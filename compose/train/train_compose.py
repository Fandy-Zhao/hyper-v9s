"""Compose conditional-residual expert training (single unified entry).

``--compose-mode`` selects the routing strategy:

- ``fixed``: zero through four fixed experts compose the forward for every sample
  (foundation / baseline training);
- ``cluster_expert``: per-sample cluster-wise conditional-residual training.
  Each sample routes to ``old_teacher_set + cluster_expert`` via the
  ``compose_selections`` batch key (ComposeSelectionDataset /
  ComposeSelectionCollator); only the cluster expert receives gradient; the
  base model, vision encoder, projector and all historical experts stay
  frozen. Old experts are loaded from ``--compose-checkpoint``, new cluster
  experts are registered as trainable, and the training job ends by writing
  a standalone per-expert state dict (``expert_<id>.pt``) for the commit
  transaction.

This module is the only formal Compose trainer: the legacy candidate-only
entry is merged here and removed.
"""

import json
import os
import sys
from typing import List, Optional

import torch
import transformers

from llava import conversation as conversation_lib

from compose.adapters import (
    ExpertManager,
    inject_compose_adapters,
    validate_compose_injection,
)
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.model import ComposeLlavaForCausalLM, load_compose_config

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .profiler import (
    TrainingProfiler,
    attach_llava_hooks,
    build_step_callback,
    count_compose_linear_calls,
    count_lora_expert_calls,
)
from .data import (
    ComposeSelectionCollator,
    ComposeSelectionDataset,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    V7QueryCollator,
    V7QueryDataset,
    make_supervised_data_module,
)
from .trainer import ComposeTrainer
from compose.v9.config import V9_COMPOSE_MODE


def _csv_ints(value: str) -> List[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expert id list must not be empty")
    return [int(item) for item in values]


def _csv_floats(value: str) -> Optional[List[float]]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in values] if values else None


def _csv_seed_map(value: str):
    mapping = {}
    for item in (entry.strip() for entry in value.split(",") if entry.strip()):
        expert_id, separator, seed = item.partition("=")
        if not separator or not seed.strip():
            raise ValueError("compose_expert_seeds entries must use EXPERT_ID=SEED")
        expert_id = int(expert_id.strip())
        if expert_id < 0:
            raise ValueError("expert ids in compose_expert_seeds must be non-negative")
        mapping[expert_id] = int(seed.strip())
    return mapping


def _expert_origin_mapping(value: str):
    mapping = {}
    for item in (entry.strip() for entry in value.split(",") if entry.strip()):
        expert_id, separator, origin_task_id = item.partition("=")
        if not separator or not origin_task_id.strip():
            raise ValueError(
                "compose_existing_expert_origins entries must use EXPERT_ID=TASK_ID"
            )
        mapping[int(expert_id.strip())] = origin_task_id.strip()
    return mapping


def _load_selection_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    manifest = {}
    for row in rows:
        manifest[str(row["sample_id"])] = row
    return manifest


def _launch_world_size() -> int:
    """Processes in this launch, readable before the Trainer exists.

    ``--compose_v8_config`` is resolved before anything is built, and the
    effective-batch guard needs the real world size there.  ``torchrun`` exports
    ``WORLD_SIZE`` for every rank, so the environment is authoritative; the
    process group is consulted first only because it is the stricter source when
    it happens to be initialised already.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _load_query_tensor(path: str, fallback_path: str):
    """(queries, rows, value_hash) from ``queries.pt``, or None to fall back.

    Existence detection is deliberately soft: a missing artefact logs a warning
    and returns ``None`` so the caller parses the JSON cache instead.  Anything
    that *is* present but fails a fingerprint, contract or structural check
    raises -- a stale or corrupted cache must never be consumed silently.  See
    :func:`compose.v7.query_cache.load_split_cache_for_training`.
    """
    from compose.v7.query_cache import load_split_cache_for_training

    target = path
    if os.path.isdir(target):
        target = os.path.join(target, "queries.pt")
    if not os.path.isfile(target):
        print(
            "[v8-accelerated] query tensor {} not found; falling back to the "
            "JSON cache {}".format(target, fallback_path),
            file=sys.stderr,
            flush=True,
        )
        return None
    queries, rows, value_hash, metadata = load_split_cache_for_training(
        os.path.dirname(target) or "."
    )
    print(
        "[v8-accelerated] query tensor {}: {} x {} {}, value_hash={}".format(
            target, queries.shape[0], queries.shape[1], queries.dtype, value_hash[:12]
        ),
        file=sys.stderr,
        flush=True,
    )
    return queries, rows, value_hash


def _resolve_expert_roles(active_experts, trainable_value: str):
    active = [int(value) for value in active_experts]
    trainable = _csv_ints(trainable_value) if trainable_value.strip() else list(active)
    if not set(trainable).issubset(set(active)):
        raise ValueError("compose_trainable_expert_ids must be a subset of active experts")
    return active, trainable


def _enable_non_reentrant_checkpointing() -> None:
    """4-GPU DDP only: switch activation checkpointing to non-reentrant.

    ``find_unused_parameters=True`` is required for cluster training
    (per-sample cluster routing leaves some new experts unused on a rank
    in a given step), but reentrant checkpointing (torch default,
    transformers 4.33 ``torch.utils.checkpoint.checkpoint``) is
    incompatible with DDP unused-parameter detection: the recompute
    pass marks parameters ready twice ('Expected to mark a variable
    ready only once', observed on the 4-GPU smoke audit). Non-reentrant
    checkpointing (torch >= 2.0) is the supported combination. The
    single-GPU path keeps the reentrant default so it mirrors the formal
    single-GPU reference exactly; the recomputed math is identical.
    """
    import torch.utils.checkpoint as checkpoint_mod

    original = checkpoint_mod.checkpoint

    def _non_reentrant(*args, **kwargs):
        if "use_reentrant" not in kwargs:
            kwargs["use_reentrant"] = False
        return original(*args, **kwargs)

    checkpoint_mod.checkpoint = _non_reentrant


def _prepare_cluster_expert_backward() -> None:
    """Make per-sample selections visible to checkpoint recomputation.

    Reentrant activation checkpointing (the torch default used by LLaVA)
    recomputes each checkpointed segment during ``loss.backward()``.
    Multithreaded autograd (on by default) dispatches that recomputation
    to autograd-engine worker threads, and ``ContextVar`` values do not
    propagate to those threads: ``use_selection`` becomes invisible, the
    recomputed graph is backbone-only, and the cluster expert receives no
    gradients (``lora_B`` stays at its zero initialization). The real
    smoke observed exactly this: ``Finite-gradient LoRA-B count: 0`` and
    committed experts whose ``lora_B.weight`` was exactly zero everywhere.

    Two guards, in order of importance:

    1. Pin the backward to the calling thread so the recompute runs with
       the selection context active.
    2. Refuse single-process multi-GPU DataParallel: ``nn.DataParallel``
       runs the forward on its own worker threads, where the selection
       context is invisible in *both* passes, silently training
       backbone-only experts. Distributed DDP (world_size > 1 via
       torchrun) is safe: each rank is its own process, its forward and
       backward run on the main thread, and the selection context stays
       visible in both passes (verified by the 4-GPU gradient-audit
       smoke, spec §9).
    """
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed and torch.cuda.device_count() > 1:
        raise ValueError(
            "cluster_expert mode requires exactly one visible GPU per "
            "process (CUDA_VISIBLE_DEVICES with a single device), or a "
            "distributed launch (torchrun, one GPU per rank): "
            "nn.DataParallel runs the model forward on worker threads "
            "where the per-sample selection context is invisible, which "
            "silently trains backbone-only experts. Got {} visible GPUs.".format(
                torch.cuda.device_count()
            )
        )
    torch.autograd.set_multithreading_enabled(False)


def _build_model(model_args, training_args):
    config = load_compose_config(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    )
    if model_args.compose_attn_implementation:
        # Execution-only: the attention kernel is chosen here and nowhere else.
        # ``transformers`` 4.33.3 has no TrainingArguments field for this, so it
        # must be set on the config before the weights are loaded.
        config._attn_implementation = model_args.compose_attn_implementation
    config.mm_vision_tower = model_args.vision_tower
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.mm_projector_type = model_args.mm_projector_type
    model = ComposeLlavaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.config.use_cache = False
    model.get_model().initialize_vision_modules(model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    vision_tower.to(device=training_args.device, dtype=vision_dtype)
    return model, vision_tower


def _inject_and_pool(model, model_args):
    adapter_config = ComposeAdapterConfig(
        rank=model_args.compose_rank,
        alpha=model_args.compose_alpha,
        dropout=model_args.compose_dropout,
        target_modules=[
            item.strip()
            for item in model_args.compose_target_modules.split(",")
            if item.strip()
        ],
    )
    injected = inject_compose_adapters(model, adapter_config)
    injection_summary = validate_compose_injection(model, injected)
    manager = ExpertManager(model)
    pool = ExpertPool(manager)
    return injected, injection_summary, manager, pool


def _load_old_checkpoint(pool, model_args, training_args):
    if not model_args.compose_checkpoint:
        return None
    loaded_manifest = load_expert_checkpoint(pool, model_args.compose_checkpoint)
    load_summary = loaded_manifest["load_summary"]
    calibration = loaded_manifest.get("rms_calibration") or {}
    if calibration:
        from compose.lora.rms import apply_kappa_calibration

        apply_kappa_calibration(pool.manager.model, calibration)
        load_summary["rms_calibration_applied"] = True
    for expert_id, origin_task_id in _expert_origin_mapping(
        model_args.compose_existing_expert_origins
    ).items():
        metadata = pool.get(expert_id)
        if metadata.origin_task_id not in (None, origin_task_id):
            raise ValueError(
                "expert {} origin task mismatch; checkpoint={!r}, requested={!r}".format(
                    expert_id, metadata.origin_task_id, origin_task_id
                )
            )
        metadata.origin_task_id = origin_task_id
    if training_args.local_rank in (-1, 0):
        print("Compose checkpoint load summary: {}".format(load_summary))
    return load_summary


def _v9_total_steps(trainer) -> int:
    """Optimizer steps the V9 stage schedule is defined over.

    The three stages are ratios of the whole task, so a total that disagrees
    with the dataloader the trainer runs moves every boundary at once and the
    discretisation stage -- the one deployment actually depends on -- lands in
    the wrong place.  This mirrors HF's own ``max_steps`` derivation rather than
    approximating it.
    """
    import math

    args = trainer.args
    if int(getattr(args, "max_steps", 0)) > 0:
        return int(args.max_steps)
    loader = trainer.get_train_dataloader()
    accumulation = max(int(args.gradient_accumulation_steps), 1)
    per_epoch = max(len(loader) // accumulation, 1)
    epochs = float(args.num_train_epochs)
    return max(int(math.ceil(epochs * per_epoch)), 1)


def _run_v9_calibration(
    trainer, model_args, training_args, v9_config, tokenizer, data_args
) -> None:
    """Spec §30: check the gate-gradient proxy against exact removal, once.

    Runs only on rank 0.  It reuses the model that is already resident, so the
    check costs a handful of forwards on a small held-out batch instead of a
    second 7B load.  Ranks that do not run it wait at a barrier rather than
    racing ahead into checkpoint writing.
    """
    import torch as _torch

    from compose.v9.data import V9QueryCollator, V9QueryDataset
    from compose.v9.retrieval import HistoricalTopC

    manifest_path = model_args.compose_v9_calibration
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    world_size = int(getattr(training_args, "world_size", 1))
    rank = int(getattr(training_args, "process_index", 0))
    if rank != 0:
        if world_size > 1 and _torch.distributed.is_initialized():
            _torch.distributed.barrier()
        return
    with open(manifest["query_cache"], "r", encoding="utf-8") as handle:
        query_cache = json.load(handle)
    dataset = V9QueryDataset(
        manifest["data_path"],
        tokenizer,
        data_args,
        query_cache,
        None,
        historical_topc=HistoricalTopC.load(manifest["retrieval_cache"]),
    )
    loader = _torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training_args.per_device_train_batch_size),
        shuffle=False,
        # The same collator the training loop uses.  ``V7QueryCollator`` would
        # drop the per-sample historical recall row -- it only knows about the
        # fixed query -- and the calibration forward would then route without
        # it.
        collate_fn=V9QueryCollator(tokenizer),
        num_workers=0,
    )
    report = trainer.calibration_pass(
        loader, int(manifest.get("sample_budget", 64))
    )
    calibration_dir = training_args.output_dir
    if report is not None:
        with open(
            os.path.join(calibration_dir, "v9_contribution_calibration.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("V9 contribution calibration: {}".format(report), flush=True)
    with open(
        os.path.join(calibration_dir, "v9_candidate_validation_gain.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            {str(k): float(v) for k, v in trainer.v9_validation_gain.items()},
            handle, indent=2, sort_keys=True,
        )
        handle.write("\n")
    if world_size > 1 and _torch.distributed.is_initialized():
        _torch.distributed.barrier()


def _register_new_experts(pool, expert_ids, model_args):
    seed_map = _csv_seed_map(model_args.compose_expert_seeds)
    unknown_seed_ids = sorted(set(seed_map) - set(int(value) for value in expert_ids))
    if unknown_seed_ids:
        raise ValueError(
            "compose_expert_seeds contains ids not being registered: {}".format(
                unknown_seed_ids
            )
        )
    for expert_id in expert_ids:
        if expert_id not in pool.expert_ids():
            def register():
                pool.register(
                    expert_id,
                    name=(model_args.compose_expert_name or "expert-{:04d}".format(expert_id)),
                    origin_task_id=model_args.compose_origin_task_id,
                    source_checkpoint=model_args.compose_checkpoint,
                    tags=[
                        value.strip()
                        for value in model_args.compose_expert_tags.split(",")
                        if value.strip()
                    ],
                )

            if expert_id not in seed_map:
                register()
                continue
            # Expert construction consumes both CPU and (depending on when
            # adapters are injected) CUDA RNG.  Restore the caller's RNG state
            # after each expert so the only initialization difference is the
            # explicitly recorded expert seed.
            cpu_state = torch.random.get_rng_state()
            cuda_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            torch.manual_seed(seed_map[expert_id])
            register()
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def _check_expected_parameters(model, model_args):
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if (
        model_args.expected_adapter_parameters is not None
        and trainable_parameter_count != model_args.expected_adapter_parameters
    ):
        raise ValueError(
            "trainable parameter count mismatch; expected={}, actual={}".format(
                model_args.expected_adapter_parameters, trainable_parameter_count
            )
        )
    return trainable_parameter_count


def _normalize_compose_argv(argv):
    """Map the hyphenated Compose CLI contract (spec §12, e.g.
    ``--compose-mode``) onto the underscore flags that transformers 4.33's
    HfArgumentParser registers (``--compose_mode``). Only ``--compose-*``
    tokens are rewritten; every other argument passes through untouched."""
    return [
        "--" + token[2:].replace("-", "_") if token.startswith("--compose-") else token
        for token in argv
    ]


#: The one formal V9-S mode.  Kept as a module constant rather than repeated as
#: a literal because the name decides eleven branches below; a typo in any one
#: of them would silently fall through to a different training loop.
V9_MODE = V9_COMPOSE_MODE

#: The V9 v1 spelling of the same mode.  Accepted only so a saved command line
#: still resolves; ``train()`` prints a warning and continues as V9-S.
#: legacy_v9_only.
V9_LEGACY_MODE = "v9_global_coevolution"

V9_MODES = (V9_MODE, V9_LEGACY_MODE)


def train() -> None:
    sys.argv = _normalize_compose_argv(sys.argv)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.vision_tower is None:
        raise ValueError("Compose training requires --vision_tower")

    mode = model_args.compose_mode
    if mode == V9_LEGACY_MODE:  # legacy_v9_only
        print(
            "[v9s] --compose-mode {} is the V9 v1 spelling; running {} "
            "instead".format(V9_LEGACY_MODE, V9_MODE),
            file=sys.stderr,
            flush=True,
        )
        mode = V9_MODE
    if mode not in (
        "fixed",
        "cluster_expert",
        "v7_global_coevolution",
        V9_MODE,
    ):
        raise ValueError(
            "--compose-mode must be fixed, cluster_expert, v7_global_coevolution, "
            "or {}".format(V9_MODE)
        )
    if model_args.compose_v8_config:
        # Applied after the command line and before anything is built, so the
        # config can only ever *add* execution flags.  Recipe knobs it moves are
        # checked against the frozen baseline immediately afterwards.
        from .v8_flags import apply_config, assert_recipe_invariants, load_config

        config_path = os.path.abspath(model_args.compose_v8_config)
        declared = load_config(config_path)
        world_size = _launch_world_size()
        resolved_flags = apply_config(
            declared, model_args, training_args, config_path, world_size=world_size
        )
        invariants = assert_recipe_invariants(
            training_args, model_args, world_size=world_size
        )
        print(
            "[v8-exact-accelerated] {} -> {}".format(
                config_path,
                json.dumps(
                    {"flags": resolved_flags, "invariants": invariants},
                    sort_keys=True,
                    default=str,
                ),
            ),
            file=sys.stderr,
            flush=True,
        )
    if mode == V9_MODE:
        # Loaded before anything is built: a malformed V9 recipe must fail before
        # a 7B checkpoint is paged in, and the candidate count it declares is
        # what the expert registration below is checked against.
        from compose.v9 import V9Config, assert_frozen_contract

        import yaml as _yaml

        with open(model_args.compose_v9_config, "r", encoding="utf-8") as handle:
            v9_config = V9Config.from_dict(_yaml.safe_load(handle) or {})
        assert_frozen_contract(v9_config)
    saved_expert_ids = []
    if mode == "cluster_expert":
        if not model_args.compose_selection_manifest:
            raise ValueError(
                "cluster_expert mode requires --compose-selection-manifest"
            )
        if not model_args.compose_cluster_expert_ids.strip():
            raise ValueError(
                "cluster_expert mode requires --compose-cluster-expert-ids"
            )
    elif mode == "v7_global_coevolution":
        required = {
            "compose_v7_key_state": model_args.compose_v7_key_state,
            "compose_v7_query_cache": model_args.compose_v7_query_cache,
            "compose_v7_config": model_args.compose_v7_config,
            "compose_v7_metrics_path": model_args.compose_v7_metrics_path,
            "compose_v7_runtime_contract": model_args.compose_v7_runtime_contract,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("V7 mode requires {}".format(", ".join(missing)))
        if not model_args.compose_cluster_expert_ids.strip():
            raise ValueError("V7 mode requires four --compose-cluster-expert-ids")
        if model_args.max_samples is not None:
            raise ValueError("V7 training forbids max_samples; provide an explicit smoke split")
        if model_args.tune_mm_mlp_adapter:
            raise ValueError("V7 trains only current Candidate LoRA and Key parameters")
    elif mode == V9_MODE:
        required = {
            "compose_v9_config": model_args.compose_v9_config,
            "compose_v9_key_state": model_args.compose_v9_key_state,
            "compose_v9_query_cache": model_args.compose_v9_query_cache,
            "compose_v9_retrieval_cache": model_args.compose_v9_retrieval_cache,
            "compose_v9_metrics_path": model_args.compose_v9_metrics_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("V9 mode requires {}".format(", ".join(missing)))
        if not model_args.compose_cluster_expert_ids.strip():
            raise ValueError("V9 mode requires the current candidate ids")
        if model_args.max_samples is not None:
            raise ValueError("V9 training forbids max_samples; provide an explicit smoke split")
        if model_args.tune_mm_mlp_adapter:
            raise ValueError(
                "V9 trains only the current Candidate LoRA, the routing keys and "
                "the routing bias; the projector stays frozen"
            )
    else:
        if not model_args.compose_expert_ids.strip():
            raise ValueError("fixed mode requires --compose-expert-ids")

    profile_path = None
    if model_args.profile_training:
        profile_path = model_args.profile_path or os.path.join(
            training_args.output_dir, "profile_steps.jsonl"
        )
        if training_args.world_size > 1:
            # One JSONL per rank: concurrent appends from several ranks would
            # interleave partial lines.
            stem, suffix = os.path.splitext(profile_path)
            profile_path = "{}.rank{}{}".format(stem, training_args.process_index, suffix)
    profiler = TrainingProfiler(
        path=profile_path,
        flush_every=model_args.profile_flush_every,
        sync=model_args.profile_sync,
        extra={
            "task_index": int(model_args.compose_v7_task_index),
            "world_size": int(training_args.world_size),
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
        },
    )
    if model_args.compose_selection_plan:
        # S5 must be decided before the first forward; it changes execution only.
        from compose.adapters.lora import set_fast_selection

        set_fast_selection(True)
    with profiler.startup_timer("model_build"):
        model, vision_tower = _build_model(model_args, training_args)
    with profiler.startup_timer("adapter_injection"):
        injected, injection_summary, manager, pool = _inject_and_pool(model, model_args)

    if mode in ("cluster_expert", "v7_global_coevolution", V9_MODE):
        _prepare_cluster_expert_backward()
        _load_old_checkpoint(pool, model_args, training_args)
        cluster_expert_ids = _csv_ints(model_args.compose_cluster_expert_ids)
        if mode == "v7_global_coevolution" and len(cluster_expert_ids) != 4:
            raise ValueError("V7 requires exactly four current candidate IDs")
        if mode == V9_MODE and len(cluster_expert_ids) != (
            v9_config.candidate_count
        ):
            raise ValueError(
                "V9 config declares {} current candidates but {} ids were given".format(
                    v9_config.candidate_count, len(cluster_expert_ids)
                )
            )
        _register_new_experts(pool, cluster_expert_ids, model_args)
        saved_expert_ids = list(cluster_expert_ids)
        pool.train_only(cluster_expert_ids)
        # Reentrant activation checkpointing (torch default) re-runs each
        # layer under torch.no_grad() in backward and only connects the graph
        # when at least one checkpoint input requires grad.  The whole base
        # is frozen (embeddings included), so re-enable grad on the
        # embeddings only: they stay out of the optimizer (LoRA-only update)
        # but give each checkpointed layer a grad-requiring input, the
        # standard HF LoRA + gradient-checkpointing arrangement.
        model.gradient_checkpointing_enable()
        model.model.embed_tokens.weight.requires_grad_(True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # DDP + find_unused_parameters=True (see S6 4-GPU launch)
            # requires non-reentrant checkpointing.
            _enable_non_reentrant_checkpointing()
        if mode in ("v7_global_coevolution", V9_MODE):
            # Keep frozen embeddings out of the optimizer while still making
            # checkpointed layer inputs require grad.
            model.model.embed_tokens.weight.requires_grad_(False)
            model.enable_input_require_grads()
    else:
        _load_old_checkpoint(pool, model_args, training_args)
        selected_experts, trainable_experts = _resolve_expert_roles(
            _csv_ints(model_args.compose_expert_ids),
            model_args.compose_trainable_expert_ids,
        )
        gates = _csv_floats(model_args.compose_gates)
        if len(selected_experts) > 4:
            raise ValueError("Compose fixed mode trains at most four experts")
        _register_new_experts(pool, selected_experts, model_args)
        saved_expert_ids = list(selected_experts)
        pool.train_only(trainable_experts)
        manager.set_default_selection(
            selected_experts,
            gates,
            normalization=model_args.compose_gate_normalization,
        )

    trainable_parameter_count = _check_expected_parameters(model, model_args)
    if model_args.tune_mm_mlp_adapter:
        model.get_model().mm_projector.requires_grad_(True)
    if training_args.gradient_checkpointing and mode == "fixed":
        model.enable_input_require_grads()

    if training_args.local_rank in (-1, 0):
        print("Compose mode: {}".format(mode))
        print("Compose injection summary: {}".format(injection_summary))
        print("Compose expert count: {}".format(len(pool.expert_ids())))
        print("Trainable expert IDs: {}".format(sorted(pool.trainable_expert_ids)))
        print("Trainable parameter count: {}".format(trainable_parameter_count))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        model_args.version, conversation_lib.conv_templates["vicuna_v1"]
    )
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(model_args, tokenizer)
    if mode in ("v7_global_coevolution", V9_MODE):
        from compose.v7.provenance import (
            build_runtime_contract,
            load_runtime_contract,
            validate_runtime_contract,
        )

        actual_runtime = build_runtime_contract(
            image_aspect_ratio=data_args.image_aspect_ratio,
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            mm_vision_select_feature=model_args.mm_vision_select_feature,
            mm_projector_type=model_args.mm_projector_type,
            projector_path=model_args.pretrain_mm_mlp_adapter,
        )
        contract_path = (
            model_args.compose_v7_runtime_contract
            if mode == "v7_global_coevolution"
            else model_args.compose_v9_runtime_contract
        )
        if contract_path:
            # V9 checks the runtime contract when one is supplied: the query
            # cache it routes with is only meaningful under the same image
            # handling, vision tower and projector that produced it.
            validate_runtime_contract(
                load_runtime_contract(contract_path), actual_runtime, "training"
            )
        elif mode == V9_MODE:
            print(
                "V9 runtime contract not supplied; the query cache is trusted as-is",
                file=sys.stderr,
                flush=True,
            )

    if mode == "cluster_expert":
        selections = _load_selection_manifest(model_args.compose_selection_manifest)
        dataset = ComposeSelectionDataset(
            data_args.data_path, tokenizer, data_args, selections
        )
        if model_args.max_samples and len(dataset.records) > model_args.max_samples:
            kept_ids = {
                str(record.get("id", index))
                for index, record in enumerate(dataset.records[: model_args.max_samples])
            }
            dataset.records = dataset.records[: model_args.max_samples]
            dataset.selections = {
                sample_id: selection
                for sample_id, selection in dataset.selections.items()
                if sample_id in kept_ids
            }
        data_collator = ComposeSelectionCollator(tokenizer)
        data_module = {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": data_collator,
        }
    elif mode == "v7_global_coevolution":
        from compose.v7.config import V7Config
        from compose.v7.pool import V7ExpertKeyPool

        import yaml

        with open(model_args.compose_v7_config, "r", encoding="utf-8") as handle:
            v7_config = V7Config.from_dict(yaml.safe_load(handle))
        key_state = torch.load(
            model_args.compose_v7_key_state, map_location="cpu", weights_only=False
        )
        v7_key_pool = V7ExpertKeyPool.from_state(key_state)
        if set(v7_key_pool.current_ids) != set(cluster_expert_ids):
            raise ValueError("V7 current candidate IDs do not match key state")
        if set(v7_key_pool.historical_ids) != (
            set(pool.expert_ids()) - set(cluster_expert_ids)
        ):
            raise ValueError("V7 historical key and LoRA registries do not match")
        reusable_historical_ids = None
        if model_args.compose_v8_reusable_screening:
            from compose.v8.screening import load_reusable_screening

            screening = load_reusable_screening(
                model_args.compose_v8_reusable_screening,
                expected_task=model_args.compose_v7_task_index,
                historical_ids=v7_key_pool.historical_ids,
            )
            reusable_historical_ids = screening[
                "reusable_historical_expert_ids"
            ]
            print("===== Teacher Screening =====")
            print("TeacherSamples: {}".format(screening["teacher_sample_count"]))
            print("HistoricalExpertCount: {}".format(len(v7_key_pool.historical_ids)))
            print("ReusableHistoricalExperts: {}".format(reusable_historical_ids))
            print("===== Full Training Routing =====")
            print("Selectable Old Experts: {}".format(reusable_historical_ids))
            print("Selectable New Candidates: {}".format(list(v7_key_pool.current_ids)))
            print("FullTrainingOracleEvalSampleCount: 0")
        # The reuse keys were created by S2 with the same initializer as the
        # candidates.  Fail closed if the pool and the screening artifact
        # disagree about who may own a learnable current-task key.
        reuse_key_ids = sorted(
            key_id for key_id in v7_key_pool.key_ids
            if v7_key_pool.route_keys[key_id].key_type == "reuse"
        )
        expected_reuse_key_ids = sorted(
            v7_key_pool.reuse_key_id(int(expert_id), int(model_args.compose_v7_task_index))
            for expert_id in (reusable_historical_ids or ())
        )
        if reuse_key_ids != expected_reuse_key_ids:
            raise ValueError(
                "V7 reuse-key registry {} does not match the screening reusable set "
                "{}".format(reuse_key_ids, expected_reuse_key_ids)
            )
        for key_id in reuse_key_ids:
            entry = v7_key_pool.route_keys[key_id]
            if entry.lifecycle != "current" or not entry.trainable:
                raise ValueError("reuse key {} must be learnable for this task".format(key_id))
            if v7_key_pool.metadata[entry.expert_id]["lifecycle"] != "historical":
                raise ValueError(
                    "reuse key {} must belong to a frozen historical expert".format(key_id)
                )
        if reuse_key_ids:
            print("===== Reuse Keys =====")
            print("ReusableHistoricalReuseKeys: {}".format(reuse_key_ids))
            print(
                "HistoricalLoraTrainable: False; HistoricalCanonicalKeysTrainable: False"
            )
        # Registering this module on the model makes current keys optimizer
        # parameters and moves/checkpoints them with the training model.
        model.v7_key_pool = v7_key_pool
        query_cache = None
        query_tensor = None
        if model_args.compose_v7_query_tensor:
            with profiler.startup_timer("query_load"):
                query_tensor = _load_query_tensor(
                    model_args.compose_v7_query_tensor,
                    model_args.compose_v7_query_cache,
                )
        if query_tensor is None:
            with profiler.startup_timer("query_load"):
                with open(model_args.compose_v7_query_cache, "r", encoding="utf-8") as handle:
                    query_cache = json.load(handle)
        with profiler.startup_timer("dataset_build"):
            dataset = V7QueryDataset(
                data_args.data_path, tokenizer, data_args, query_cache, query_tensor
            )
            data_collator = V7QueryCollator(tokenizer)
        data_module = {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": data_collator,
        }
    elif mode == V9_MODE:
        from compose.v9 import (
            V9KeyPool,
            V9QueryCollator,
            V9QueryDataset,
            V9Router,
        )
        from compose.v9.retrieval import HistoricalTopC

        task_index = int(model_args.compose_v9_task_index)
        with profiler.startup_timer("v9_key_state"):
            key_state = torch.load(
                model_args.compose_v9_key_state, map_location="cpu", weights_only=False
            )
            v9_key_pool = V9KeyPool.from_state(key_state, current_task=task_index)
            v9_key_pool.validate()
            # Everything not on this task is frozen before the optimizer exists:
            # base keys, earlier task keys, and the discarded keys of pruned
            # experts.  Only this task's candidates and this task's historical
            # task keys
            # may train.
            v9_key_pool.freeze_historical(current_task=task_index)
        if set(v9_key_pool.current_ids) != set(cluster_expert_ids):
            raise ValueError(
                "V9 current candidate ids {} do not match the key state {}".format(
                    sorted(cluster_expert_ids), sorted(v9_key_pool.current_ids)
                )
            )
        if set(v9_key_pool.historical_ids) != (
            set(pool.expert_ids()) - set(cluster_expert_ids)
        ):
            raise ValueError(
                "V9 historical key registry {} and LoRA registry {} disagree".format(
                    sorted(v9_key_pool.historical_ids),
                    sorted(set(pool.expert_ids()) - set(cluster_expert_ids)),
                )
            )
        v9_router = V9Router(
            config=v9_config,
            key_pool=v9_key_pool,
            candidate_ids=cluster_expert_ids,
            historical_ids=v9_key_pool.historical_ids,
            task_index=task_index,
        )
        # The pool is a submodule of the router and the router is a submodule of
        # the model, so one assignment registers every key as an optimizer
        # parameter, moves it with the model and exposes it to the
        # trainable-parameter audit.  Nothing here may be held off to the side.
        model.v9_router = v9_router
        manager.set_cardinality_scale(v9_config.composition.cardinality_scale)

        query_cache = None
        query_tensor = None
        if model_args.compose_v7_query_tensor:
            with profiler.startup_timer("query_load"):
                query_tensor = _load_query_tensor(
                    model_args.compose_v7_query_tensor,
                    model_args.compose_v9_query_cache,
                )
        if query_tensor is None:
            with profiler.startup_timer("query_load"):
                with open(model_args.compose_v9_query_cache, "r", encoding="utf-8") as handle:
                    query_cache = json.load(handle)
        with profiler.startup_timer("dataset_build"):
            historical_topc = HistoricalTopC.load(model_args.compose_v9_retrieval_cache)
            if historical_topc.task_index != task_index:
                raise ValueError(
                    "historical Top-C cache belongs to task {} but this run is "
                    "task {}".format(historical_topc.task_index, task_index)
                )
            dataset = V9QueryDataset(
                data_args.data_path,
                tokenizer,
                data_args,
                query_cache,
                query_tensor,
                historical_topc=historical_topc,
            )
            data_collator = V9QueryCollator(tokenizer)
        data_module = {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": data_collator,
        }
    else:
        data_module = make_supervised_data_module(tokenizer, data_args)

    if mode == "v7_global_coevolution":
        from compose.v7.hf_trainer import (
            V7ComposeTrainer,
            attach_v7_ddp_key_anchor,
        )

        attach_v7_ddp_key_anchor(model, v7_key_pool)

        _profiler_handles = []
        if profiler.enabled:
            _profiler_handles.extend(attach_llava_hooks(model, profiler))
            _profiler_handles.extend(count_compose_linear_calls(manager, profiler))
            _profiler_handles.extend(count_lora_expert_calls(manager, profiler))

        trainer = V7ComposeTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            expert_pool=pool,
            v7_key_pool=v7_key_pool,
            v7_config=v7_config,
            v7_task_index=model_args.compose_v7_task_index,
            v7_metrics_path=model_args.compose_v7_metrics_path,
            v7_require_full_coverage=model_args.compose_v7_require_full_coverage,
            v7_reusable_historical_ids=reusable_historical_ids,
            v8_reuse_quality_enabled=bool(model_args.compose_v8_reuse_quality_enabled),
            v8_reuse_quality_temperature=float(model_args.compose_v8_reuse_quality_temperature),
            v8_reuse_quality_floor=float(model_args.compose_v8_reuse_quality_floor),
            v7_profiler=profiler,
            **data_module
        )
        _profiler_callback = build_step_callback(profiler)
        if _profiler_callback is not None:
            trainer.add_callback(_profiler_callback)
    elif mode == V9_MODE:
        from compose.v9.trainer import V9ComposeTrainer

        _profiler_handles = []
        if profiler.enabled:
            _profiler_handles.extend(attach_llava_hooks(model, profiler))
            _profiler_handles.extend(count_compose_linear_calls(manager, profiler))
            _profiler_handles.extend(count_lora_expert_calls(manager, profiler))

        trainer = V9ComposeTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            expert_pool=pool,
            v9_config=v9_config,
            v9_router=v9_router,
            v9_task_index=task_index,
            v9_metrics_path=model_args.compose_v9_metrics_path,
            v9_total_steps=int(model_args.compose_v9_total_steps),
            v9_require_full_coverage=bool(model_args.compose_v9_require_full_coverage),
            v9_profiler=profiler,
            **data_module
        )
        # The schedule boundaries must be derived from the dataloader the
        # trainer will actually run, not from an estimate passed on the command
        # line: bootstrap/soft/discretisation are ratios, so a wrong total moves
        # every boundary at once and the last stage would land in the wrong
        # place.  ``max_steps > 0`` overrides, matching HF.
        if int(model_args.compose_v9_total_steps) <= 0:
            trainer.set_total_steps(_v9_total_steps(trainer))
        _profiler_callback = build_step_callback(profiler)
        if _profiler_callback is not None:
            trainer.add_callback(_profiler_callback)
    else:
        trainer = ComposeTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            expert_pool=pool,
            **data_module
        )
    checkpoints = [
        name
        for name in os.listdir(training_args.output_dir)
        if name.startswith("checkpoint-")
    ] if os.path.isdir(training_args.output_dir) else []
    resumable = ("v7_global_coevolution", V9_MODE)
    if checkpoints and mode not in resumable:
        raise ValueError(
            "output_dir contains checkpoint-* entries; automatic resume is disabled "
            "because Compose checkpoints are adapter-only: {}".format(sorted(checkpoints))
        )
    distributed_audits = None
    if mode in resumable:
        trainer.create_optimizer()
        distributed_audits = {
            "before_training": trainer.distributed_state_audit("before_training")
        }
    with profiler.startup_timer("train_wall"):
        trainer.train(
            resume_from_checkpoint=True if mode in resumable and checkpoints else None
        )
    if mode in resumable:
        profiler.close()
        distributed_audits["after_training"] = trainer.distributed_state_audit(
            "after_training"
        )
        trainer.distributed_barrier()
    trainer.save_state()
    coverage_audit = (
        trainer.full_data_coverage_audit(len(data_module["train_dataset"]))
        if mode in resumable
        else None
    )
    if mode == V9_MODE and model_args.compose_v9_calibration:
        _run_v9_calibration(
            trainer, model_args, training_args, v9_config, tokenizer, data_args
        )
    model.config.use_cache = True
    if mode in resumable:
        trainer.distributed_barrier()
    # ``global_task_statistics`` all-gathers across ranks, so every rank has to
    # enter it.  It used to be called inside the ``should_save`` block below,
    # which runs on rank 0 only: rank 1 walked out of ``train()`` and exited
    # while rank 0 sat in the collective, and the run died at teardown with a
    # gloo "connection closed by peer" that names neither the audit nor the
    # file.  The reduce happens here, on every rank; only the write is rank 0's.
    v9_global_statistics = (
        trainer.global_task_statistics() if mode == V9_MODE else None
    )
    if training_args.should_save:
        if mode == V9_MODE:
            with open(
                os.path.join(training_args.output_dir, "v9_distributed_audit.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(distributed_audits, handle, indent=2, sort_keys=True)
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v9_full_data_coverage.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(coverage_audit, handle, indent=2, sort_keys=True)
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v9_trainable_parameter_audit.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(trainer.trainable_parameter_audit(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v9_task_statistics.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    v9_global_statistics,
                    handle, indent=2, sort_keys=True,
                )
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v9_freeze_audit.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(trainer.assert_task_freeze_integrity(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            # Spec §41 (A): the claim that ``L_ans`` trains the Candidate LoRA
            # and no key is checked on the live training graph, once per task.
            # Until this file existed the check ran and would have raised, but
            # the report it produced was discarded, so the evidence for the
            # report's central claim was a *silence* -- and a silence is what a
            # check that never ran also looks like.  Written on rank 0 only,
            # like the audits beside it: ``should_save`` is rank 0's condition
            # and this block must contain no collective.
            with open(
                os.path.join(training_args.output_dir, "v9_answer_key_isolation.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(trainer.v9_answer_key_isolation, handle, indent=2, sort_keys=True)
                handle.write("\n")
            torch.save(
                v9_key_pool.export_state(),
                os.path.join(training_args.output_dir, "v9_key_pool.pt"),
            )
        if mode == "v7_global_coevolution":
            with open(
                os.path.join(training_args.output_dir, "v7_distributed_audit.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(distributed_audits, handle, indent=2, sort_keys=True)
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v7_full_data_coverage.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(coverage_audit, handle, indent=2, sort_keys=True)
                handle.write("\n")
            freeze_audit = trainer.assert_task_freeze_integrity()
            with open(
                os.path.join(training_args.output_dir, "v7_freeze_audit.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(freeze_audit, handle, indent=2, sort_keys=True)
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v7_trainable_parameter_audit.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(trainer.trainable_parameter_audit(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            with open(
                os.path.join(training_args.output_dir, "v7_training_diagnostics.json"),
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    trainer.final_diagnostics(len(data_module["train_dataset"])),
                    handle, indent=2, sort_keys=True,
                )
                handle.write("\n")
            torch.save(
                v7_key_pool.export_state(),
                os.path.join(training_args.output_dir, "v7_key_pool.pt"),
            )
        pool.sync_training_step(trainer.state.global_step)
        pool.train_only([])
        model.config.save_pretrained(training_args.output_dir)
        if mode == "v7_global_coevolution":
            from compose.lora.rms import runtime_kappa_calibration

            save_expert_checkpoint(
                pool,
                training_args.output_dir,
                rms_calibration=runtime_kappa_calibration(model),
            )
        elif mode == V9_MODE:
            # Same as V7: the RMS calibration is part of the recipe, not a
            # convenience.  Dropping it here would silently un-calibrate every
            # historical expert on the next task's composition.
            from compose.lora.rms import runtime_kappa_calibration

            save_expert_checkpoint(
                pool,
                training_args.output_dir,
                rms_calibration=runtime_kappa_calibration(model),
            )
        else:
            save_expert_checkpoint(pool, training_args.output_dir)
        # Standalone per-expert state dicts are written for both fixed and
        # cluster modes so every capacity-chain point can be assembled and
        # evaluated through the same loader.
        for expert_id in saved_expert_ids:
            payload = {}
            for layer_name, layer in manager.layers.items():
                expert = layer.experts[str(expert_id)]
                payload["{}.lora_A.weight".format(layer_name)] = expert.lora_A.weight
                payload["{}.lora_B.weight".format(layer_name)] = expert.lora_B.weight
            torch.save(
                {"expert_id": expert_id, "state_dict": payload},
                os.path.join(training_args.output_dir, "expert_{:04d}.pt".format(expert_id)),
            )
    if training_args.local_rank in (-1, 0):
        supervision_summary = data_module["data_collator"].supervision_summary()
        if training_args.dataloader_num_workers:
            supervision_summary["scope"] = "main-process-only"
            supervision_summary["note"] = (
                "worker collators enforce zero-supervision errors but do not share counters; "
                "use compose.train.audit_supervision for dataset-wide statistics"
            )
        print("Compose supervision summary: {}".format(supervision_summary))
        print("Trainable LoRA-B count: {}".format(trainer.trainable_lora_b_count))
        print("Finite-gradient LoRA-B count: {}".format(
            trainer.max_finite_gradient_lora_b_count
        ))
        print("Injected {} Compose layers; experts={}".format(len(injected), pool.expert_ids()))


if __name__ == "__main__":
    train()
