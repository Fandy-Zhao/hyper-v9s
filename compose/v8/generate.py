"""Generation and answer-NLL over **per-sample** expert selections.

V7's evaluator can express exactly two routing shapes: one fixed expert, or a
precomputed two-expert route per sample.  The V8 policy needs a third --
per-sample cardinality ``0`` (BaseOnly) / ``1`` (Reuse1) / ``2`` (Reuse2) -- and
``compose/eval/eval_task.py`` refuses anything that is not two *distinct*
experts (``"V7 validation selection must contain two distinct experts"``).  That
restriction is a property of the V7 harness, not of the model: ``ComposeLinear``
already handles every cardinality through ``ComposeSelection``.  This module is
the harness that exposes it, so V8 does not have to fork the model code.

Two things are deliberately *not* optimised:

* **Batch size stays 1.**  Every official V7/UCIT number on this pool was
  produced one sample at a time.  Batching would change padding and therefore
  the logits, so a V8 number produced in batches would not be comparable to the
  V7 baseline it is measured against.  The cost is wall-clock, which is the
  honest price of comparability.
* **The prompt is built exactly as ``compose/eval/eval_task.py`` builds it.**
  ``_prompt`` below is a copy, not an import, because importing it would drag in
  the whole V7 argparse surface; the copy is asserted equal to the V7 one by the
  test suite.

Everything generated is written into an append-only cache keyed by
``(route, sample_id)``, so a crashed run resumes without regenerating and the
exact answers behind every reported number remain on disk for audit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

from compose.data.records import question_text
from compose.v8.config import STATE_BASE_ONLY, STATE_REUSE1, STATE_REUSE2
from compose.v8.selection import build_selection


def route_key(experts: Sequence[int]) -> str:
    """Canonical cache key for a composition route.

    Sorted, because the composed delta is a sum over experts and the pair scale
    is symmetric -- ``[a, b]`` and ``[b, a]`` are the same forward pass, and
    collapsing them into one cache entry makes the teacher's ``combinations``
    ordering irrelevant.
    """
    values = tuple(sorted(int(value) for value in experts))
    if not values:
        return "base"
    return "e" + "_".join("{:02d}".format(value) for value in values)


def experts_from_route(route: str) -> List[int]:
    """Inverse of :func:`route_key`; raises on anything that is not one.

    ``route_key`` writes an ``e`` prefix and then joins two-digit fields with
    ``_``, so the route for the pair ``[10, 13]`` is ``"e10_13"``.  Splitting
    that on ``_`` and dropping the first field yields ``["13"]`` -- a *single*
    expert.  That is not a hypothetical: a seed spot-check built its expert list
    that way, so every seeded *pair* was silently re-measured as the single
    expert in its second slot and the check aborted a correct run.  The prefix
    has to come off before the split, and the result has to round-trip.
    """
    text = str(route)
    if text == "base":
        return []
    if not text.startswith("e"):
        raise GenerationError("malformed route {!r}".format(route))
    fields = text[1:].split("_")
    if not fields or not all(field.isdigit() for field in fields):
        raise GenerationError("malformed route {!r}".format(route))
    experts = [int(field) for field in fields]
    if len(set(experts)) != len(experts) or route_key(experts) != text:
        raise GenerationError("malformed route {!r}".format(route))
    return experts


def _record_id(record: Mapping[str, object]) -> str:
    return str(record.get("id", record.get("question_id")))


def _prompt(record, model_config, conv_mode):
    """Byte-identical to ``compose/eval/eval_task.py:_prompt``.

    Kept in sync by the V8 test suite; if the V7 evaluator changes its prompt,
    that test fails rather than the numbers silently drifting.
    """
    question = question_text(record)
    if DEFAULT_IMAGE_TOKEN not in question:
        image_token = DEFAULT_IMAGE_TOKEN
        if model_config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        question = image_token + "\n" + question
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


class GenerationError(RuntimeError):
    """Raised when a route cannot be generated or a cached answer is missing."""


class RouteGenerationCache:
    """Append-only ``(route, sample_id) -> text`` store.

    Append-only on purpose: a re-run must never silently overwrite the answers
    that a previous run's metrics were computed from.  Rewriting a route would
    invalidate every number derived from it, so this class has no update path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._texts: Dict[Tuple[str, str], str] = {}
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._texts[(str(row["route"]), str(row["sample_id"]))] = str(row["text"])
        self._handle = None

    def __len__(self) -> int:
        return len(self._texts)

    def __bool__(self) -> bool:
        """An empty cache is still a cache.

        ``__len__`` exists for reporting, and Python derives truthiness from it:
        without this override a freshly opened cache evaluates to ``False``, so
        every ``if cache:`` guard silently skips writing.  That is exactly the
        bug this method was added to fix -- a run reported 31 generated answers
        and left no cache file behind.
        """
        return True

    def __contains__(self, item: Tuple[str, str]) -> bool:
        return item in self._texts

    def get(self, sample_id: str, experts: Sequence[int]) -> Optional[str]:
        return self._texts.get((route_key(experts), str(sample_id)))

    def put(self, sample_id: str, experts: Sequence[int], text: str) -> None:
        key = (route_key(experts), str(sample_id))
        if key in self._texts:
            if self._texts[key] != text:
                raise GenerationError(
                    "route {} sample {} already cached with different text".format(*key)
                )
            return
        self._texts[key] = str(text)
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(
            {"route": key[0], "sample_id": key[1], "text": str(text)},
            ensure_ascii=False, sort_keys=True,
        ) + "\n")
        self._handle.flush()

    def rows(self) -> List[Dict[str, str]]:
        return [
            {"route": route, "sample_id": sample_id, "text": text}
            for (route, sample_id), text in sorted(self._texts.items())
        ]


class GenerationEngine:
    """Deterministic (greedy) generation under an arbitrary per-sample policy."""

    def __init__(
        self,
        bundle,
        image_folder: str,
        device: str = "cuda:0",
        max_new_tokens: int = 128,
        conv_mode: str = "vicuna_v1",
        cache_path: Optional[str | Path] = None,
    ) -> None:
        self.bundle = bundle
        self.image_folder = str(image_folder)
        self.device = str(device)
        self.max_new_tokens = int(max_new_tokens)
        self.conv_mode = str(conv_mode)
        self.cache = RouteGenerationCache(cache_path) if cache_path else None
        self._encoded: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.generated_count = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------
    def _encode(self, record: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = _record_id(record)
        cached = self._encoded.get(sample_id)
        if cached is not None:
            return cached
        prompt = _prompt(record, self.bundle.model.config, self.conv_mode)
        input_ids = tokenizer_image_token(
            prompt, self.bundle.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)
        image = Image.open(
            os.path.join(self.image_folder, str(record["image"]))
        ).convert("RGB")
        image_tensor = process_images(
            [image], self.bundle.image_processor, self.bundle.model.config
        )[0].unsqueeze(0)
        # Kept on CPU: 256 samples of 336x336 bf16 is ~170 MB of device memory
        # that buys nothing, since the transfer is negligible next to a forward.
        self._encoded[sample_id] = (input_ids, image_tensor)
        return input_ids, image_tensor

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_one(self, record: Mapping[str, object], experts: Sequence[int]) -> str:
        input_ids, image_tensor = self._encode(record)
        manager = self.bundle.expert_pool.manager
        selection = build_selection(
            ["row"],
            {"row": _state_for(experts)},
            {"row": [int(value) for value in experts]},
        )
        with manager.selection_context(selection):
            output_ids = self.bundle.model.generate(
                input_ids=input_ids.to(self.device),
                images=image_tensor.to(device=self.device, dtype=torch.bfloat16),
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        input_length = input_ids.shape[1]
        return self.bundle.tokenizer.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True
        )[0].strip()

    # ------------------------------------------------------------------
    def generate_route(
        self,
        selection: Mapping[str, Sequence[int]],
        records_by_id: Mapping[str, Mapping[str, object]],
        progress=None,
    ) -> Dict[str, str]:
        """Answers for every ``sample_id -> experts`` in ``selection``.

        Iteration is grouped by route so that the (dominant) base-only and
        single-expert routes are walked in contiguous stretches; each sample is
        still generated on its own, as the V7 harness does.
        """
        groups: Dict[str, List[str]] = {}
        for sample_id, experts in selection.items():
            groups.setdefault(route_key(experts), []).append(str(sample_id))

        answers: Dict[str, str] = {}
        total = len(selection)
        done = 0
        for route in sorted(groups):
            sample_ids = sorted(groups[route])
            for sample_id in sample_ids:
                experts = [int(value) for value in selection[sample_id]]
                cached = (
                    self.cache.get(sample_id, experts)
                    if self.cache is not None else None
                )
                if cached is None:
                    record = records_by_id.get(sample_id)
                    if record is None:
                        raise GenerationError(
                            "no validation record for sample {}".format(sample_id)
                        )
                    cached = self._generate_one(record, experts)
                    self.generated_count += 1
                    if self.cache is not None:
                        self.cache.put(sample_id, experts, cached)
                else:
                    self.cache_hits += 1
                answers[sample_id] = cached
                done += 1
                if progress is not None and done % 50 == 0:
                    progress("route {}: {}/{} samples".format(route, done, total))
        return answers


def _state_for(experts: Sequence[int]) -> str:
    count = len(list(experts))
    if count == 0:
        return STATE_BASE_ONLY
    if count == 1:
        return STATE_REUSE1
    return STATE_REUSE2


__all__ = [
    "GenerationEngine",
    "GenerationError",
    "RouteGenerationCache",
    "experts_from_route",
    "route_key",
]
