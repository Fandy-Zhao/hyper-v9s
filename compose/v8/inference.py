"""V8 inference: query -> multi-key router -> LoRA composition.

At inference time V8 may use **only**:

* the input (image + question),
* the fixed multimodal query,
* the committed expert keys (origin and alias alike),
* the Multi-Key Router,
* RMS / LoRA composition.

It may **not** use the ground-truth answer, any answer-correctness signal, an
answer NLL against the ground truth, the teacher cache, an oracle, or the task
label.  This is a design constraint that has to survive future edits, so it is
enforced mechanically: :func:`assert_inference_purity` parses this module's own
AST and fails if a forbidden identifier or string appears outside of
documentation.  The check reads *code*, not prose -- docstrings are excluded by
construction -- so the ban can be described here without tripping the test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from compose.adapters.types import ComposeSelection
from compose.v8.config import (
    STATE_BASE_ONLY,
    STATE_REUSE1,
    STATE_REUSE2,
    V8RoutingConfig,
)
from compose.v8.pool import MultiKeyExpertPool
from compose.v8.routing import MultiKeyRouteResult, MultiKeyRouter
from compose.v8.selection import build_selection


class InferenceError(RuntimeError):
    """Raised when inference is asked to consult something it must not."""


#: Identifiers that must never appear in executable inference code.
FORBIDDEN_INFERENCE_IDENTIFIERS = frozenset({
    "ground_truth",
    "groundtruth",
    "gt_answer",
    "answer_text",
    "gold_answer",
    "reference_answer",
    "teacher_cache",
    "teacher_labels",
    "answer_correctness",
    "answer_nll",
    "oracle",
    "task_label",
    "true_label",
})

#: String literals that would signal a supervision shortcut.
FORBIDDEN_INFERENCE_STRINGS = frozenset({
    "ground_truth",
    "answer_nll",
    "teacher_cache",
})


#: The two names that *define* what is forbidden.  Their own literals have to be
#: exempt from the scan, or the module could never state the rule it enforces --
#: an exemption keyed on the assignment target, so it cannot be used to smuggle
#: a banned word into any other binding.
EXEMPT_DEFINITION_NAMES = frozenset({
    "FORBIDDEN_INFERENCE_IDENTIFIERS",
    "FORBIDDEN_INFERENCE_STRINGS",
})


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_exempt_definition(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return False
    return isinstance(target, ast.Name) and target.id in EXEMPT_DEFINITION_NAMES


def collect_code_identifiers(source: str) -> Tuple[set, set]:
    """Names and string literals appearing in executable code (no comments/docstrings).

    Prose is excluded by construction: docstrings are dropped and the two
    ``FORBIDDEN_*`` definitions are dropped, so the ban can be *described* here
    without the description reading as a violation.
    """
    tree = ast.parse(source)
    identifiers: set = set()
    strings: set = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            identifiers.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            identifiers.add(node.attr)
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> None:
            identifiers.add(node.arg)
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str):
                strings.add(node.value)
            self.generic_visit(node)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and _is_docstring(node.body[0]):
                node.body = node.body[1:]
            node.body = [
                statement for statement in node.body
                if not _is_exempt_definition(statement)
            ]
    Visitor().visit(tree)
    return identifiers, strings


def assert_inference_purity(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Prove the inference module's executable code touches no supervision."""
    path = Path(path) if path is not None else Path(__file__)
    source = path.read_text(encoding="utf-8")
    identifiers, strings = collect_code_identifiers(source)
    bad_ids = sorted(identifiers & FORBIDDEN_INFERENCE_IDENTIFIERS)
    bad_strings = sorted(strings & FORBIDDEN_INFERENCE_STRINGS)
    if bad_ids or bad_strings:
        raise InferenceError(
            f"inference code references supervision: identifiers={bad_ids} "
            f"strings={bad_strings}"
        )
    return {
        "path": str(path),
        "identifiers_scanned": len(identifiers),
        "strings_scanned": len(strings),
        "forbidden_identifiers_found": bad_ids,
        "forbidden_strings_found": bad_strings,
        "pure": True,
    }


class V8InferenceRouter(nn.Module):
    """Query-only routing.  Holds no supervision of any kind."""

    def __init__(
        self,
        pool: MultiKeyExpertPool,
        routing_config: Optional[V8RoutingConfig] = None,
        excluded_experts: Iterable[int] = (),
    ) -> None:
        super().__init__()
        self.pool = pool
        self.router = MultiKeyRouter(routing_config)
        self.excluded_experts = tuple(int(value) for value in excluded_experts)

    def forward(self, queries: torch.Tensor) -> MultiKeyRouteResult:
        return self.router(queries, self.pool, excluded_experts=self.excluded_experts)

    def selection(self, queries: torch.Tensor, sample_ids: Sequence[str]) -> ComposeSelection:
        """Turn routed experts into the per-sample selection the forward pass reads."""
        if len(sample_ids) != queries.shape[0]:
            raise InferenceError(
                f"{len(sample_ids)} sample ids for {queries.shape[0]} queries"
            )
        result = self.forward(queries)
        states: Dict[str, str] = {}
        experts: Dict[str, List[int]] = {}
        for index, sample_id in enumerate(sample_ids):
            row = [int(value) for value in result.expert_ids[index].tolist()]
            sample_id = str(sample_id)
            experts[sample_id] = row
            states[sample_id] = (
                STATE_REUSE2 if len(row) > 1
                else (STATE_REUSE1 if row else STATE_BASE_ONLY)
            )
        return build_selection(sample_ids, states, experts, device=queries.device)

    def route_policy(self, queries: torch.Tensor) -> Dict[str, Any]:
        """The persisted inference policy: expert ids per row, and the winning keys.

        Note what is *absent*: no sample ids, no answers, nothing that came from
        the teacher.  This dict is the whole inference record.
        """
        result = self.forward(queries)
        rows = []
        for index in range(queries.shape[0]):
            rows.append({
                "expert_ids": [int(value) for value in result.expert_ids[index].tolist()],
                "scores": [float(value) for value in result.expert_scores[index].tolist()],
                "key_ids": [
                    (None if key_id is None else str(key_id))
                    for key_id in result.key_ids[index]
                ],
            })
        return {"rows": rows, "policy": "multi_key_cosine_topk"}


def validate_policy(pool: MultiKeyExpertPool, policy: Mapping[str, Any]) -> None:
    """A committed inference policy may only name experts that exist and are live."""
    live = set(pool.live_expert_ids())
    for index, row in enumerate(policy.get("rows", [])):
        for expert_id in row.get("expert_ids", []):
            if int(expert_id) not in live:
                raise InferenceError(
                    f"inference policy row {index} names expert {expert_id}, "
                    "which is not a live expert of this pool"
                )


__all__ = [
    "FORBIDDEN_INFERENCE_IDENTIFIERS",
    "FORBIDDEN_INFERENCE_STRINGS",
    "InferenceError",
    "V8InferenceRouter",
    "assert_inference_purity",
    "collect_code_identifiers",
    "validate_policy",
]
