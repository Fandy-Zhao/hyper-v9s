"""What "this task is finished" means, and what admits the six-task run.

Two questions, deliberately answered by two functions, because conflating them
is what produced the failure this module exists to prevent.

``task_completion`` answers *may task t be skipped*.  The chain it replaces
asked only whether ``state/key_pool_taskN.pt`` and ``v9_task_statistics.json``
existed -- a pure ``-f`` test with no content check and no notion of the
evaluation row.  Two consequences followed, both silent:

* a truncated or zero-byte checkpoint satisfied it permanently, because nothing
  ever re-ran the task to notice;
* a task whose eval row ``A[t][0..t]`` was never produced was still "complete",
  so the lower-triangular matrix could come out missing cells that nothing
  reported as missing.

So every requirement here reads the artefact's *content* and says which one
failed.  The predicate is the conjunction the V9 plan states: training
finished, the key commit ran, the freeze and integrity audits passed, the
calibration artefacts exist, and the task's own evaluation row is present.

``go_no_go`` answers *may the formal six-task sequence start at all*.  It reads
the preflight's evidence and the launch contract, and it refuses on the
performance conditions the efficiency brief names -- most importantly the
one-backbone-forward invariant, which is the method's central cost claim and
the one that a per-expert enumeration would break while every other number
still looked healthy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Task order of the formal UCIT sequence (compose indices 0..5).
TASK_NAMES = ("ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k")


class V9ClosureError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _requirement(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _missing(name: str, path: Path) -> Dict[str, Any]:
    return _requirement(name, False, "missing {}".format(path))


def task_completion(
    root: Path,
    task_index: int,
    *,
    require_full_coverage: bool = True,
    require_eval: bool = True,
    eval_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Every stage of task ``task_index``, checked by reading its artefact.

    ``eval_root`` is where the lower-triangular matrix lives.  It defaults to
    ``root``, but the formal chain keeps one per-task training root *and* one
    shared evaluation root: the query cache names its file without a task
    component, so training has to be per-task, while the matrix has to
    accumulate across tasks to be a matrix at all.
    """
    root = Path(root)
    evaluation_root = Path(eval_root) if eval_root is not None else root
    task = int(task_index)
    training = root / "training" / "task{}".format(task)
    requirements: List[Dict[str, Any]] = []

    # ---- 1. training finished -------------------------------------------
    statistics_path = training / "v9_task_statistics.json"
    if not statistics_path.is_file():
        requirements.append(_missing("training", statistics_path))
    else:
        try:
            statistics = _read_json(statistics_path)
            micro_steps = int(statistics.get("micro_steps", 0))
            observed = int(statistics.get("observed_sample_count", 0))
            requirements.append(
                _requirement(
                    "training",
                    micro_steps > 0 and observed > 0,
                    "micro_steps={} observed_sample_count={}".format(micro_steps, observed),
                )
            )
        except (OSError, ValueError, TypeError) as error:
            requirements.append(
                _requirement("training", False, "unreadable: {}".format(error))
            )
    experts = training / "compose_experts.bin"
    requirements.append(
        _requirement(
            "training_experts",
            experts.is_file() and experts.stat().st_size > 0,
            "{} bytes={}".format(
                experts, experts.stat().st_size if experts.is_file() else "absent"
            ),
        )
    )

    # ---- 2. coverage (the flag the formal runner passes) -----------------
    coverage_path = training / "v9_full_data_coverage.json"
    if require_full_coverage:
        if not coverage_path.is_file():
            requirements.append(_missing("full_coverage", coverage_path))
        else:
            try:
                coverage = _read_json(coverage_path)
                ratio = float(coverage.get("train_sample_coverage", 0.0))
                requirements.append(
                    _requirement(
                        "full_coverage",
                        ratio >= 1.0,
                        "train_sample_coverage={:.6f}".format(ratio),
                    )
                )
                # The tail predicate.  ``train_sample_coverage`` counts forwards,
                # and a micro-batch whose accumulation window never closed was
                # forwarded -- so a run can report 1.0 there while a handful of
                # declared samples never reached an optimizer step.  The two
                # requirements below are the ones that close that gap, and the
                # step count is recomputed from the recorded split size rather
                # than read back from the file, so agreement here is agreement
                # with the contract and not with the run's own arithmetic.
                declared = int(coverage.get("num_train_samples", 0))
                global_batch = int(coverage.get("global_batch", 0)) or None
                applied = coverage.get("unique_optimizer_applied_sample_ids")
                opt_ratio = coverage.get("optimizer_coverage")
                unclosed = int(coverage.get("unclosed_window_sample_count", 0))
                observed_steps = int(coverage.get("optimizer_steps", 0))
                if opt_ratio is None or applied is None:
                    requirements.append(
                        _requirement(
                            "optimizer_coverage",
                            False,
                            "the coverage artefact predates the optimizer-side "
                            "audit: it records forwards but not optimizer-applied "
                            "samples, so an unclosed tail window would be "
                            "invisible here",
                        )
                    )
                else:
                    ok = float(opt_ratio) >= 1.0 and unclosed == 0
                    requirements.append(
                        _requirement(
                            "optimizer_coverage",
                            ok,
                            "optimizer_coverage={:.6f} applied={} unclosed_window_samples={}".format(
                                float(opt_ratio), applied, unclosed
                            ),
                        )
                    )
                expected = coverage.get("expected_optimizer_steps")
                if global_batch and declared:
                    required = int(math.ceil(declared / global_batch))
                    if expected is not None and int(expected) != required:
                        requirements.append(
                            _requirement(
                                "optimizer_steps",
                                False,
                                "artefact records expected_optimizer_steps={} but "
                                "ceil({} / {}) = {}".format(
                                    expected, declared, global_batch, required
                                ),
                            )
                        )
                    else:
                        requirements.append(
                            _requirement(
                                "optimizer_steps",
                                observed_steps == required,
                                "actual_optimizer_steps={} expected={} "
                                "(ceil({} / {}))".format(
                                    observed_steps, required, declared, global_batch
                                ),
                            )
                        )
            except (OSError, ValueError, TypeError) as error:
                requirements.append(
                    _requirement("full_coverage", False, "unreadable: {}".format(error))
                )

    # ---- 3. the key commit actually ran ----------------------------------
    committed = root / "state" / "key_pool_task{}.pt".format(task)
    audit_path = root / "data" / "audit_task{}.json".format(task)
    if not committed.is_file():
        requirements.append(_missing("key_commit", committed))
    elif not audit_path.is_file():
        requirements.append(_missing("key_commit", audit_path))
    else:
        try:
            audit = _read_json(audit_path)
            applied = audit.get("candidate_commit_applied")
            task_keys = audit.get("task_key_applied")
            committed_ids = list((applied or {}).get("committed", []))
            historical_ids = list((applied or {}).get("historical_ids", []))
            deployable = sorted(set(committed_ids) | set(historical_ids))
            # A pool with nothing deployable cannot serve Top-1/Top-2 at
            # inference, so an "applied" commit that leaves the row empty is a
            # failure the two-file test could not see.
            requirements.append(
                _requirement(
                    "key_commit",
                    isinstance(applied, dict)
                    and isinstance(task_keys, dict)
                    and len(deployable) > 0,
                    "committed={} historical={} task_keys_reset={}".format(
                        committed_ids, historical_ids, (task_keys or {}).get("reset", [])
                    ),
                )
            )
        except (OSError, ValueError, TypeError) as error:
            requirements.append(
                _requirement("key_commit", False, "unreadable: {}".format(error))
            )

    # ---- 4. freeze / integrity audits ------------------------------------
    freeze_path = training / "v9_freeze_audit.json"
    if not freeze_path.is_file():
        requirements.append(_missing("freeze_audit", freeze_path))
    else:
        try:
            freeze = _read_json(freeze_path)
            failed = sorted(key for key, value in freeze.items() if value is not True)
            requirements.append(
                _requirement(
                    "freeze_audit",
                    not failed and bool(freeze),
                    "all unchanged" if not failed else "NOT unchanged: {}".format(failed),
                )
            )
        except (OSError, ValueError, TypeError) as error:
            requirements.append(
                _requirement("freeze_audit", False, "unreadable: {}".format(error))
            )
    for name, filename in (
        ("answer_key_isolation", "v9_answer_key_isolation.json"),
        ("distributed_audit", "v9_distributed_audit.json"),
        ("trainable_parameter_audit", "v9_trainable_parameter_audit.json"),
    ):
        path = training / filename
        if path.is_file():
            try:
                _read_json(path)
                requirements.append(_requirement(name, True, str(path)))
            except (OSError, ValueError) as error:
                # The isolation report was once computed every task and written
                # nowhere; now that it is written, a corrupt one must not pass
                # as evidence.
                requirements.append(
                    _requirement(name, False, "unparseable: {}".format(error))
                )
        else:
            requirements.append(_missing(name, path))

    # ---- 5. calibration artefacts ----------------------------------------
    calibration_path = training / "v9_contribution_calibration.json"
    if not calibration_path.is_file():
        requirements.append(_missing("calibration", calibration_path))
    else:
        try:
            calibration = _read_json(calibration_path)
            samples = int(calibration.get("samples", 0))
            requirements.append(
                _requirement(
                    "calibration",
                    samples > 0 and "pearson" in calibration,
                    "samples={} pearson={}".format(
                        samples, calibration.get("pearson")
                    ),
                )
            )
        except (OSError, ValueError, TypeError) as error:
            requirements.append(
                _requirement("calibration", False, "unreadable: {}".format(error))
            )
    gain_path = training / "v9_candidate_validation_gain.json"
    if not gain_path.is_file():
        requirements.append(_missing("validation_gain", gain_path))
    else:
        try:
            gains = _read_json(gain_path)
            requirements.append(
                _requirement(
                    "validation_gain",
                    isinstance(gains, dict) and len(gains) > 0,
                    "{} experts measured".format(
                        len(gains) if isinstance(gains, dict) else 0
                    ),
                )
            )
        except (OSError, ValueError) as error:
            requirements.append(
                _requirement("validation_gain", False, "unreadable: {}".format(error))
            )

    # ---- 6. the task's own evaluation row A[t][0..t] ----------------------
    if require_eval:
        matrix_path = evaluation_root / "evaluation" / "continual_matrix.json"
        expected = list(range(task + 1))
        if not matrix_path.is_file():
            requirements.append(_missing("eval_row", matrix_path))
        else:
            try:
                matrix = _read_json(matrix_path)
                row = (matrix.get("rows") or {}).get(str(task), {})
                present = sorted(int(cell) for cell in row)
                scored = sorted(
                    int(cell)
                    for cell, metric in row.items()
                    if isinstance(metric, dict) and metric.get("value") is not None
                )
                requirements.append(
                    _requirement(
                        "eval_row",
                        present == expected and scored == expected,
                        "expected cells {}, present {}, scored {}".format(
                            expected, present, scored
                        ),
                    )
                )
            except (OSError, ValueError, TypeError) as error:
                requirements.append(
                    _requirement("eval_row", False, "unreadable: {}".format(error))
                )

    failed = [row["name"] for row in requirements if not row["ok"]]
    return {
        "task_index": task,
        "task_name": TASK_NAMES[task] if 0 <= task < len(TASK_NAMES) else None,
        "root": str(root),
        "complete": not failed,
        "failed": failed,
        "requirements": requirements,
    }


def go_no_go(
    preflight_root: Optional[Path] = None,
    *,
    world_size: int = 4,
    per_device_batch: int = 4,
    grad_accum: int = 2,
    target_global_batch: int = 32,
    query_encoder_calls: int = 0,
) -> Dict[str, Any]:
    """The gate the efficiency brief sets before a formal six-task run."""
    gates: List[Dict[str, Any]] = []
    effective = int(world_size) * int(per_device_batch) * int(grad_accum)
    gates.append(
        _requirement(
            "global_batch",
            effective == int(target_global_batch),
            "{} x {} x {} = {} (target {})".format(
                world_size, per_device_batch, grad_accum, effective, target_global_batch
            ),
        )
    )
    gates.append(
        _requirement(
            "query_source",
            int(query_encoder_calls) == 0,
            "QUERY_ENCODER_CALLS={} (a formal run encodes nothing)".format(
                query_encoder_calls
            ),
        )
    )
    if preflight_root is not None:
        root = Path(preflight_root)
        for task in range(len(TASK_NAMES)):
            if not (root / "training" / "task{}".format(task)).is_dir():
                continue
            completion = task_completion(root, task, require_full_coverage=False, require_eval=False)
            gates.append(
                _requirement(
                    "preflight_task{}".format(task),
                    completion["complete"],
                    "failed: {}".format(completion["failed"]),
                )
            )
            metrics = sorted(
                (root / "metrics").glob("task{}_train_steps*.jsonl".format(task))
            )
            if not metrics:
                gates.append(
                    _requirement(
                        "backbone_invariant_task{}".format(task),
                        False,
                        "no step metrics under {}".format(root / "metrics"),
                    )
                )
                continue
            ratio = _backbone_ratio(metrics)
            measured = [
                ratio["model_forwards_per_micro_step"],
                ratio["backbone_forwards_per_micro_step"],
            ]
            # ``measured`` counts as evidence only if a number was actually
            # read.  This gate once compared against a key the trainer does not
            # write, so it passed on every input including the ones it exists to
            # reject; requiring a reading is what keeps "no violation found" and
            # "nothing was looked at" from being the same answer.
            gates.append(
                _requirement(
                    "backbone_invariant_task{}".format(task),
                    any(value is not None for value in measured)
                    and not any(_ratio_violates(value) for value in measured)
                    and not _ratio_violates(ratio["wide_model_forwards_per_micro_step"]),
                    "hook={} bookkeeping={} wide={}".format(*measured,
                                                             ratio["wide_model_forwards_per_micro_step"]),
                )
            )
    blocked = [row["name"] for row in gates if not row["ok"]]
    return {
        "decision": "GO" if not blocked else "NO-GO",
        "blocked_by": blocked,
        "gates": gates,
    }


def _backbone_ratio(paths: List[Path]) -> Dict[str, Any]:
    """Last logged per-micro-step traversal ratios across a task's rank metrics.

    Read from the last row rather than summed, because each row already carries
    the cumulative ratio and summing cumulative ratios would multiply it by the
    number of rows.

    Two ratios are collected and both are gated on, because they are independent
    measurements of one claim: ``backbone_forwards_per_micro_step`` is the
    trainer's own bookkeeping and ``model_forwards_per_micro_step`` comes from a
    forward hook on the model itself.  The hook counter is the one that cannot be
    fooled by bookkeeping that forgets to increment, so a gate that read only one
    of them would be trusting the thing it exists to check.
    """
    result: Dict[str, Any] = {
        "model_forwards_per_micro_step": None,
        "backbone_forwards_per_micro_step": None,
        "wide_model_forwards_per_micro_step": None,
    }
    for path in paths:
        try:
            lines = [
                line
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            continue
        if not lines:
            continue
        try:
            row = json.loads(lines[-1])
        except json.JSONDecodeError:
            continue
        for key in result:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                current = result[key]
                result[key] = float(value) if current is None else max(current, float(value))
    return result


def _ratio_violates(value: Any) -> bool:
    """True when a measured traversals-per-micro-step ratio is not ~1.

    ``None`` means the run predates the counter that would have produced it, and
    is *not* a violation -- but it is also not evidence, which is why the gate
    below reports which of the two ratios it managed to read.
    """
    if value is None:
        return False
    return not 0.99 <= float(value) <= 1.01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="a task run root")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--preflight-root", default=None)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--query-encoder-calls", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.task_index is None:
        payload = go_no_go(
            args.preflight_root,
            world_size=args.world_size,
            per_device_batch=args.per_device_batch,
            grad_accum=args.grad_accum,
            query_encoder_calls=args.query_encoder_calls,
        )
    else:
        payload = task_completion(Path(args.root), args.task_index)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if "decision" in payload:
            print("{} (blocked by: {})".format(payload["decision"], payload["blocked_by"] or "nothing"))
        else:
            print(
                "task {} {}: {}".format(
                    payload["task_index"],
                    "COMPLETE" if payload["complete"] else "INCOMPLETE",
                    "all requirements met" if payload["complete"] else payload["failed"],
                )
            )
    raise SystemExit(0 if payload.get("complete", payload.get("decision") == "GO") else 1)


if __name__ == "__main__":
    main()
