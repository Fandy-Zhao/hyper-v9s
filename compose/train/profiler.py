"""Low-overhead phase profiler for the Compose training step.

Design contract
---------------
* **Off by default.**  When :class:`TrainingProfiler` is constructed with
  ``path=None`` every entry point degrades to a no-op (``_NoopPhase``), so the
  un-profiled code path carries no synchronisation and no allocation.
* **Measurement never changes the numbers it measures.**  ``timed`` boundaries
  are ``torch.cuda.synchronize()`` + ``time.perf_counter()`` pairs placed
  strictly *around* existing calls, and only a handful of them per optimizer
  step.  ``cpu`` marks time inner loops (per-layer, per-expert) without any
  device synchronisation, so they cannot stall the GPU; they answer "how much
  of the step is CPU-bound in this module", which is exactly the question the
  fine-grained buckets exist for.  No computed value, loss term, gradient or
  route is touched by either.
* **One row per optimizer step**, aggregated over the accumulation window, with
  buffered flush (``flush_every``) so logging never becomes the bottleneck it is
  supposed to be measuring.

Phase names follow the V8 acceleration brief so the baseline report and the
speedup report compare field by field.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _NoopPhase:
    """Zero-cost stand-in used whenever profiling is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NoopPhase":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


_NOOP = _NoopPhase()


class _Phase:
    __slots__ = ("profiler", "name", "sync", "started")

    def __init__(self, profiler: "TrainingProfiler", name: str, sync: bool) -> None:
        self.profiler = profiler
        self.name = name
        self.sync = sync
        self.started = 0.0

    def __enter__(self) -> "_Phase":
        if self.sync:
            _sync()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> bool:
        if self.sync:
            _sync()
        self.profiler.record(self.name, time.perf_counter() - self.started)
        return False


class _StartupPhase(_Phase):
    __slots__ = ()

    def __exit__(self, *_exc) -> bool:
        if self.sync:
            _sync()
        self.profiler.startup[self.name] = round(
            self.profiler.startup.get(self.name, 0.0)
            + time.perf_counter()
            - self.started,
            6,
        )
        return False


class TrainingProfiler:
    """Aggregate per-optimizer-step phase timings into a JSONL stream."""

    #: Sync-paired phases summed over one accumulation window.  ``step_body_time``
    #: is the *whole* ``ComposeTrainer.training_step`` call (forward + backward +
    #: gradient audit); ``routing_time``, ``allreduce_time``, ``audit_time`` and
    #: ``metrics_time`` all happen *inside* it, so they are sub-phases and are
    #: excluded from the accounted total.  What is genuinely disjoint from the
    #: step body is ``optimizer_time`` (``optimizer.step`` in the Trainer loop)
    #: and ``data_wait_time`` (the gap between consecutive steps, logged but not
    #: counted).  ``accounted_step_time`` is therefore the real partition.
    PHASES = (
        "step_body_time",
        "routing_time",
        "allreduce_time",
        "optimizer_time",
        "metrics_time",
        "data_wait_time",
    )

    #: Phases that are disjoint from ``step_body_time``; only these may be
    #: added to it when computing ``accounted_step_time``.
    OUTSIDE_STEP_BODY = ("optimizer_time",)

    #: Derived (never measured twice) and sub-phases, reported for attribution
    #: only.  A sub-phase is a subset of one of ``PHASES`` and is excluded from
    #: the accounted total.
    DERIVED = ("backward_time", "llm_forward_time")
    SUB_PHASES = (
        "current_expert_forward_time",
        "vision_forward_time",
        "historical_expert_forward_time",
        "audit_time",
        "compose_linear_cpu_time",
        "lora_expert_cpu_time",
    )

    def __init__(
        self,
        path: Optional[str],
        flush_every: int = 25,
        sync: bool = True,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        self.enabled = path is not None
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.sync = bool(sync)
        self.extra = dict(extra or {})
        self._phases: Dict[str, float] = {}
        self._maxima: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._micro_steps = 0
        self._rows = 0
        self._buffer: List[Dict[str, object]] = []
        self._tokens = 0
        self._seq_len_total = 0
        self._samples = 0
        self._valid_tokens = 0
        self._padded_tokens = 0
        self._last_step_end: Optional[float] = None
        self._step_begin: Optional[float] = None
        self._run_begin = time.perf_counter()
        self._peak_allocated = 0
        self._peak_reserved = 0
        #: One-shot startup costs (model build, cache parse, dataset build), kept
        #: out of the per-step phases so they survive ``begin_step``'s reset.
        self.startup: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Timer API (no-ops when disabled)
    # ------------------------------------------------------------------

    @property
    def enabled_or_noop(self) -> object:
        return _NOOP

    def timed(self, name: str) -> object:
        """Sync-paired phase timer.  Reserved for a handful of coarse boundaries."""
        if not self.enabled:
            return _NOOP
        return _Phase(self, name, self.sync)

    def cpu(self, name: str) -> object:
        """CPU-only probe for inner loops.  Never synchronises the device."""
        if not self.enabled:
            return _NOOP
        return _Phase(self, name, False)

    def record(self, name: str, seconds: float) -> None:
        self._phases[name] = self._phases.get(name, 0.0) + seconds
        if seconds > self._maxima.get(name, 0.0):
            self._maxima[name] = seconds

    def bump(self, name: str, value: int = 1) -> None:
        if self.enabled:
            self._counters[name] = self._counters.get(name, 0) + int(value)

    def micro_step(self) -> None:
        if self.enabled:
            self._micro_steps += 1

    def startup_timer(self, name: str) -> object:
        """One-shot startup phase; written to the startup sidecar, not the rows."""
        if not self.enabled:
            return _NOOP
        return _StartupPhase(self, name, self.sync)

    def observe_sequence(self, batch_tokens: int) -> None:
        """Tokens in one forward batch -- ``batch x sequence``, not the width.

        ``mean_seq_len`` is then a *per-sample* length, so it means the same
        thing at every micro-batch width.  Reading only ``logits.shape[1]``
        under-counted by the batch factor, which would have inflated the
        tokens/sec column by 8x on the micro-batch-8 arms.
        """
        if self.enabled:
            self._seq_len_total += int(batch_tokens)
            self._tokens += int(batch_tokens)

    def observe_batch(self, samples: int, valid_tokens: int, padded_tokens: int) -> None:
        """Per-micro-batch truth for the samples/padding columns.

        ``observe_sequence`` only sees the padded width, so micro-batches wider
        than one need the attention mask to report a real padding ratio.
        """
        if self.enabled:
            self._samples += int(samples)
            self._valid_tokens += int(valid_tokens)
            self._padded_tokens += int(padded_tokens)

    # ------------------------------------------------------------------
    # Optimizer-step window
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._phases = {}
        self._counters = {}
        self._micro_steps = 0
        self._tokens = 0
        self._seq_len_total = 0
        self._samples = 0
        self._valid_tokens = 0
        self._padded_tokens = 0
        self._step_begin = now
        self.record(
            "data_wait_time",
            0.0 if self._last_step_end is None else now - self._last_step_end,
        )

    def end_step(self, step_index: int, extra: Optional[Dict[str, object]] = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._last_step_end = now
        row: Dict[str, object] = {"kind": "step", "step": int(step_index)}
        row.update(self.extra)
        for name in self.PHASES + self.SUB_PHASES:
            row[name] = round(self._phases.get(name, 0.0), 6)
        forward = self._phases.get("current_expert_forward_time", 0.0)
        row["backward_time"] = round(
            max(0.0, self._phases.get("step_body_time", 0.0) - forward), 6
        )
        row["llm_forward_time"] = round(
            max(0.0, forward - self._phases.get("vision_forward_time", 0.0)), 6
        )
        row["phase_max_sec"] = {k: round(v, 6) for k, v in sorted(self._maxima.items())}
        row["micro_steps"] = int(self._micro_steps)
        # ``samples``/``padding_ratio`` come from the attention mask when the
        # trainer reported it; micro-batch one never pads, so the fallback is
        # exact for the frozen baseline.
        row["samples"] = int(self._samples) if self._samples else int(self._micro_steps)
        # ``tokens`` is the post-expansion count the LLM actually processed
        # (a single ``<image>`` placeholder becomes 576 patch tokens inside the
        # model); ``text_tokens`` below is the collator's pre-expansion count.
        row["tokens"] = int(self._tokens)
        row["text_valid_tokens"] = int(self._valid_tokens)
        row["text_padded_tokens"] = int(self._padded_tokens)
        row["mean_seq_len"] = (
            round(self._tokens / row["samples"], 3) if row["samples"] else 0.0
        )
        row["padding_ratio"] = (
            round(1.0 - self._valid_tokens / self._padded_tokens, 6)
            if self._padded_tokens
            else 0.0
        )
        row["window_wall_time"] = (
            None if self._step_begin is None else round(now - self._step_begin, 6)
        )
        covered = sum(float(row[name]) for name in self.OUTSIDE_STEP_BODY) + float(
            row["step_body_time"]
        )
        row["accounted_step_time"] = round(covered, 6)
        row["unaccounted_step_time"] = round(
            0.0 if row["window_wall_time"] is None else float(row["window_wall_time"]) - covered,
            6,
        )
        row["since_run_start_sec"] = round(now - self._run_begin, 6)
        row.update({k: int(v) for k, v in self._counters.items()})
        if torch.cuda.is_available():
            self._peak_allocated = max(self._peak_allocated, int(torch.cuda.max_memory_allocated()))
            self._peak_reserved = max(self._peak_reserved, int(torch.cuda.max_memory_reserved()))
        row["peak_allocated_bytes"] = self._peak_allocated
        row["peak_reserved_bytes"] = self._peak_reserved
        if extra:
            row.update(extra)
        self._buffer.append(row)
        self._rows += 1
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buffer or self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for row in self._buffer:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._buffer = []

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush()
        if self.path is None:
            return
        sidecar = self.path.with_suffix(self.path.suffix + ".startup.json")
        with sidecar.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "startup": self.startup,
                    "startup_total_sec": round(sum(self.startup.values()), 6),
                    "profiled_optimizer_steps": self._rows,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def summary(self) -> Dict[str, object]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "profiled_optimizer_steps": self._rows,
            "rows_flushed_to": None if self.path is None else str(self.path),
            "startup": dict(self.startup),
        }


def build_step_callback(profiler: TrainingProfiler):
    """Return a ``TrainerCallback`` wired to the profiler's step boundary.

    ``on_step_begin`` fires once per accumulation window (transformers
    trainer.py:1834) and ``on_step_end`` fires after ``optimizer.step``
    (trainer.py:1928), so the gap between them is exactly one optimizer step and
    the gap *between* consecutive ``on_step_begin`` calls is the time the
    trainer spent outside the step (dataloader stall, logging, checkpointing).
    """
    if not profiler.enabled:
        return None
    from transformers import TrainerCallback

    class _ProfilerCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            profiler.begin_step()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            profiler.end_step(state.global_step)
            return control

    return _ProfilerCallback()


def wrap_optimizer_step(optimizer, profiler: TrainingProfiler):
    """Time ``optimizer.step`` in place.  Purely additive: the call is unchanged.

    The replacement must stay a *bound method*: ``torch.optim.lr_scheduler``
    calls ``with_counter(self.optimizer.step)`` at construction time, which
    dereferences ``method.__self__`` and ``__func__``.  A plain closure would
    raise ``AttributeError`` there, so ``types.MethodType`` is required.
    """
    import types

    if not profiler.enabled or getattr(optimizer, "_profiler_wrapped", False):
        return optimizer
    original = optimizer.step

    def _profiled_step(self, *args, **kwargs):
        with profiler.timed("optimizer_time"):
            return original(*args, **kwargs)

    optimizer.step = types.MethodType(_profiled_step, optimizer)
    optimizer._profiler_wrapped = True
    return optimizer


def attach_llava_hooks(model, profiler: TrainingProfiler) -> List[object]:
    """Time the frozen vision tower and the LLM trunk without altering either.

    Returns the registered handles so the caller can keep them alive (and
    ``remove()`` them).  Every lookup is defensive: a model without a vision
    tower simply gets no vision phase rather than an exception.
    """
    if not profiler.enabled:
        return []
    handles: List[object] = []
    inner = getattr(model, "get_model", lambda: None)()
    vision = getattr(inner, "get_vision_tower", lambda: None)() if inner is not None else None

    if vision is not None:
        def _vision_pre(_module, _inputs):
            profiler._vision_mark = time.perf_counter()

        def _vision_post(_module, _inputs, _output):
            mark = getattr(profiler, "_vision_mark", None)
            if mark is not None:
                profiler.record("vision_forward_time", time.perf_counter() - mark)
                profiler._vision_mark = None

        handles.append(vision.register_forward_pre_hook(_vision_pre))
        handles.append(vision.register_forward_hook(_vision_post))

    def _model_pre(_module, _inputs):
        profiler._model_mark = time.perf_counter()
        profiler.bump("model_forwards")

    def _model_post(_module, _inputs, output):
        mark = getattr(profiler, "_model_mark", None)
        if mark is not None:
            profiler.record("current_expert_forward_time", time.perf_counter() - mark)
            profiler._model_mark = None
        logits = getattr(output, "logits", None)
        if logits is not None and getattr(logits, "ndim", 0) == 3:
            profiler.observe_sequence(int(logits.shape[0]) * int(logits.shape[1]))

    handles.append(model.register_forward_pre_hook(_model_pre))
    handles.append(model.register_forward_hook(_model_post))
    return handles


def count_lora_expert_calls(manager, profiler: TrainingProfiler) -> List[object]:
    """Count and CPU-time every per-expert LoRA evaluation in the adapter pool."""
    if not profiler.enabled:
        return []
    from compose.adapters.lora import LoRAExpert

    def _pre(_module, _inputs, _profiler=profiler):
        # Must return None: a pre-hook returning a tuple replaces the forward
        # arguments (torch.nn.Module._call_impl).
        _profiler.bump("lora_expert_evaluations")
        _profiler._expert_mark = time.perf_counter()

    def _post(_module, _inputs, _output, _profiler=profiler):
        mark = getattr(_profiler, "_expert_mark", None)
        if mark is not None:
            _profiler.record("lora_expert_cpu_time", time.perf_counter() - mark)

    handles = []
    for layer in manager.layers.values():
        for expert in layer.experts.values():
            if isinstance(expert, LoRAExpert):
                handles.append(expert.register_forward_pre_hook(_pre))
                handles.append(expert.register_forward_hook(_post))
    return handles


def count_compose_linear_calls(manager, profiler: TrainingProfiler) -> List[object]:
    """Count and CPU-time every ``ComposeLinear.forward`` (including its syncs)."""
    if not profiler.enabled:
        return []
    from compose.adapters.lora import ComposeLinear

    def _pre(_module, _inputs, _profiler=profiler):
        # Must return None: a pre-hook returning a tuple replaces the forward
        # arguments (torch.nn.Module._call_impl).
        _profiler.bump("compose_layers_forward")
        _profiler._compose_mark = time.perf_counter()

    def _post(_module, _inputs, _output, _profiler=profiler):
        mark = getattr(_profiler, "_compose_mark", None)
        if mark is not None:
            _profiler.record("compose_linear_cpu_time", time.perf_counter() - mark)

    handles = []
    for layer in manager.layers.values():
        if not isinstance(layer, ComposeLinear):
            continue
        handles.append(layer.register_forward_pre_hook(_pre))
        handles.append(layer.register_forward_hook(_post))
    return handles
