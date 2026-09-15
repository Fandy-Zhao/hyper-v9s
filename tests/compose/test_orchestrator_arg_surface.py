"""The orchestrator may only read arguments its own parser declares.

``compose/experiments/v7_task_run.py`` builds its ``Namespace`` from the
``add_argument`` calls in that same file, so a reference to ``args.<name>``
with no matching declaration is not a type error the interpreter catches at
import time -- it is an ``AttributeError`` raised the first time control
reaches that line, i.e. part-way through a run whose earlier stages have
already been paid for.  It has happened twice in the V8 work (S1b's reuse-key
provenance key, then S3's screening artifact), both times from "the field
exists on the *subprocess* argument dataclass, so the orchestrator must have
it too".  The scanner below answers that question mechanically.
"""

import ast
import re
from pathlib import Path

ORCHESTRATOR = Path("compose/experiments/v7_task_run.py")


def _orchestrator_tree():
    return ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))


def _declared_arguments(tree):
    """Every destination the file's own ``add_argument`` calls can produce."""
    declared = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                declared.add(argument.value.lstrip("-").replace("-", "_"))
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                declared.add(keyword.value.value)
    return declared


def _read_arguments(tree):
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }


def test_orchestrator_reads_no_undeclared_argument():
    tree = _orchestrator_tree()
    undeclared = sorted(_read_arguments(tree) - _declared_arguments(tree))
    assert not undeclared, (
        "v7_task_run reads args.{} but never declares it; this raises "
        "AttributeError only once the run reaches that line. The value belongs "
        "either in the parser or in a deterministic path derived from the run "
        "root.".format(", args.".join(undeclared))
    )


def test_s3_is_handed_the_screening_at_its_run_root_path():
    """S3's screening input is addressed by path, never by command line.

    The artifact is written by S1b to a fixed location inside the task root and
    is deliberately absent from the run contract, so the orchestrator has to
    derive the address from ``root`` exactly as S2 does.  Handing S3 an empty
    ``--compose_v8_reusable_screening`` is not a harmless default: the trainer
    fails closed on it, because a launch that reaches S3 without the artifact
    is a broken stage order rather than a task with no reusable history.
    """
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "args.compose_v8_reusable_screening" not in source

    assert re.search(
        r"screening_path\s*=\s*root\s*/\s*SCREENING_DIR\s*/\s*SCREENING_NAME", source
    ), "the screening address must be derived from the task root"
    assert re.search(
        r'"--compose_v8_reusable_screening",\s*str\(screening_path\)', source
    ), "S3 must receive the derived screening path"


def test_the_screening_constants_match_the_hardcoded_s2_read():
    """S2 still spells the address out; the constants must not drift from it."""
    from compose.v8 import workflow

    assert workflow.SCREENING_DIR == "tasks"
    assert workflow.SCREENING_NAME == "v8_reusable_screening.json"
    assert "root / \"tasks\" / \"v8_reusable_screening.json\"" in (
        ORCHESTRATOR.read_text(encoding="utf-8")
    ).replace("'", '"')
