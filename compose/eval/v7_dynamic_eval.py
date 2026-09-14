"""Reliable six-GPU dynamic evaluator for the formal V7/V8 UCIT matrix."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from compose.eval.formal_ucit_eval import _score_answers
from compose.eval.v7_formal_ucit_eval import (
    _ensure_cell_selections,
    _mirror_to_hyper_layout,
    _repo_root,
    _write_markdown,
    generation_command,
)


PRIORITY = (
    # The final Task5 model row is the reporting priority: occupy all six
    # evaluation GPUs with A[5][0..5] before backfilling earlier rows.
    (5, 2), (5, 5), (5, 0), (5, 3), (5, 4), (5, 1),
    (4, 2), (4, 0), (4, 3), (4, 1),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _ids(records):
    return [str(row.get("question_id", row.get("id"))) for row in records]


def _read_answers(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "invalid JSON at {}:{}: {}".format(path, line_number, error)
                )
    return rows


def _validate_answers(path, records, allow_partial):
    expected = _ids(records)
    if len(expected) != len(set(expected)):
        raise RuntimeError("test data has duplicate sample IDs: {}".format(path))
    rows = _read_answers(path)
    actual = [str(row.get("question_id")) for row in rows]
    if len(actual) != len(set(actual)):
        raise RuntimeError("answers contain duplicate sample IDs: {}".format(path))
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise RuntimeError("answers contain unexpected IDs: {}".format(sorted(unexpected)[:5]))
    if not allow_partial and (len(actual) != len(expected) or set(actual) != set(expected)):
        raise RuntimeError(
            "incomplete answers {}: {}/{}".format(path, len(actual), len(expected))
        )
    return len(actual)


def _annotation(formal, task):
    path = Path(formal["tasks"][task]["test_file"])
    coco = path.with_name("val_coco_type_3000.json")
    return coco if coco.is_file() else path


def _metric_path(root, stage, task):
    return root / "evaluation" / "scores" / "t{}".format(stage) / "task{}".format(task) / "metric.json"


def _answers_path(root, stage, task):
    return root / "evaluation" / "predictions" / "t{}".format(stage) / "task{}".format(task) / "answers.jsonl"


def _process_mentions(path):
    result = subprocess.run(
        ["pgrep", "-f", "--", str(path)], capture_output=True, text=True
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _free_gpus(allowed, reserved):
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    memory = {}
    for line in result.stdout.splitlines():
        index, used = [item.strip() for item in line.split(",", 1)]
        memory[index] = int(used)
    return [gpu for gpu in allowed if gpu not in reserved and memory.get(gpu, 999999) < 1000]


def _contract(root, formal, stage, task, selection, fast):
    complete = json.loads(
        (root / "task{}".format(stage) / "task_complete.json").read_text(encoding="utf-8")
    )
    question = Path(formal["tasks"][task]["test_file"])
    runtime = root / "task{}".format(stage) / "data" / "runtime_contract.json"
    return {
        "schema_version": 1,
        "stage": stage,
        "task": task,
        "checkpoint_hash": complete["checkpoint_hash"],
        "question_sha256": _sha256(question),
        "selection_sha256": _sha256(selection),
        "runtime_contract_sha256": _sha256(runtime),
        "generation": {
            "conv_mode": "vicuna_v1",
            "max_new_tokens": 128,
            "model_max_length": 2048,
            "do_sample": False,
            "num_beams": 1,
            "fast_selection_plan": bool(fast),
        },
    }


def _adopt_partial(path, contract, checkpoint):
    if not path.is_file():
        return
    sidecar = Path(str(path) + ".resume_contract.json")
    if sidecar.is_file():
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("resume contract mismatch for {}".format(path))
        return
    for row in _read_answers(path):
        metadata = row.get("metadata") or {}
        if metadata.get("checkpoint") != str(checkpoint):
            raise RuntimeError("cannot adopt answers from a different checkpoint")
        selection = metadata.get("selection") or {}
        if selection.get("selection_source") != "precomputed_global_top2_validation":
            raise RuntimeError("cannot adopt answers with a different routing source")
    _atomic_json(sidecar, contract)


def _score(root, formal, stage, task, answers):
    _validate_answers(
        answers,
        json.loads(Path(formal["tasks"][task]["test_file"]).read_text(encoding="utf-8")),
        allow_partial=False,
    )
    return _score_answers(
        root, stage, task, answers, annotation_file=str(_annotation(formal, task))
    )


def _rebuild_matrix(root):
    matrix = {"schema_version": 1, "rows": {}}
    latest = -1
    for stage in range(6):
        metrics = []
        for task in range(stage + 1):
            path = _metric_path(root, stage, task)
            if path.is_file():
                metrics.append(json.loads(path.read_text(encoding="utf-8")))
        if len(metrics) == stage + 1:
            metrics.sort(key=lambda value: int(value["task_id"]))
            matrix["rows"][str(stage)] = {
                str(value["task_id"]): value for value in metrics
            }
            _mirror_to_hyper_layout(root, stage, metrics)
            latest = stage
    matrix["final_row_task"] = latest
    _atomic_json(root / "evaluation" / "continual_matrix.json", matrix)
    _write_markdown(root, matrix)
    return sum(len(row) for row in matrix["rows"].values())


def _state_path(control, stage, task):
    return control / "cells" / "t{}_task{}.json".format(stage, task)


def _try_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    return handle


def _write_state(control, stage, task, **values):
    path = _state_path(control, stage, task)
    state = {}
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    state.update(values)
    state.update({"stage": stage, "task": task, "updated_at": time.time()})
    _atomic_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--formal-config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", default="1,3,4,5,6,7")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--fast-selection-plan", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    formal = yaml.safe_load(Path(args.formal_config).read_text(encoding="utf-8"))
    method = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if gpus != ["1", "3", "4", "5", "6", "7"]:
        raise ValueError("formal dynamic run is pinned to GPUs 1,3,4,5,6,7")
    control = root / "evaluation" / "control"
    control.mkdir(parents=True, exist_ok=True)
    controller_lock = (control / "controller.lock").open("w")
    try:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another dynamic evaluation controller is active")
    controller_lock.write(str(os.getpid()) + "\n")
    controller_lock.flush()

    all_cells = [(stage, task) for stage in range(6) for task in range(stage + 1)]
    priority = list(PRIORITY) + [cell for cell in all_cells if cell not in PRIORITY]
    running = {}
    scoring = {}
    attempts = {}
    full_seen = {}
    score_pool = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            for gpu, job in list(running.items()):
                code = job["process"].poll()
                if code is None:
                    continue
                job["log"].close()
                job["gpu_lock"].close()
                job["cell_lock"].close()
                stage, task = job["cell"]
                running.pop(gpu)
                if code != 0:
                    attempts[(stage, task)] = attempts.get((stage, task), 0) + 1
                    _write_state(
                        control, stage, task, status="failed", exit_code=code,
                        attempts=attempts[(stage, task)], gpu=gpu,
                    )
                    if attempts[(stage, task)] > args.max_retries:
                        raise RuntimeError("cell A[{}][{}] exhausted retries".format(stage, task))
                else:
                    _write_state(control, stage, task, status="generated", gpu=gpu)

            for cell, future in list(scoring.items()):
                if not future.done():
                    continue
                stage, task = cell
                scoring.pop(cell)
                metric = future.result()
                _write_state(control, stage, task, status="complete", metric=metric)
                _rebuild_matrix(root)

            completed = {
                cell for cell in all_cells if _metric_path(root, *cell).is_file()
            }
            # The final model row is a delivery barrier, not merely a soft
            # ordering preference.  Do not spend a released GPU or scorer slot
            # on an earlier stage until every Task5 cell has an official
            # metric.  This keeps the bottom-row result available as early as
            # possible even when one of its faster cells finishes first.
            task5_complete = all((5, task) in completed for task in range(6))
            eligible_priority = (
                priority if task5_complete
                else [cell for cell in priority if cell[0] == 5]
            )
            if len(completed) == 21:
                count = _rebuild_matrix(root)
                _atomic_json(
                    control / "complete.json",
                    {"completed_at": time.time(), "metric_count": count, "gpus": gpus},
                )
                print("completed all 21 formal matrix cells", flush=True)
                return

            active_cells = set(scoring) | {job["cell"] for job in running.values()}
            now = time.time()
            for stage, task in eligible_priority:
                cell = (stage, task)
                if cell in completed or cell in active_cells:
                    continue
                answers = _answers_path(root, stage, task)
                records = json.loads(
                    Path(formal["tasks"][task]["test_file"]).read_text(encoding="utf-8")
                )
                count = _validate_answers(answers, records, allow_partial=True)
                if count == len(records):
                    if _process_mentions(answers):
                        full_seen.pop(cell, None)
                        continue
                    first_seen = full_seen.setdefault(cell, now)
                    if now - first_seen >= 30 and len(scoring) < 2:
                        scoring[cell] = score_pool.submit(
                            _score, root, formal, stage, task, answers
                        )
                        _write_state(control, stage, task, status="scoring")
                    continue
                full_seen.pop(cell, None)

            free = _free_gpus(gpus, set(running))
            for gpu in free:
                selected = None
                for stage, task in eligible_priority:
                    cell = (stage, task)
                    if cell in completed or cell in scoring or cell in active_cells:
                        continue
                    answers = _answers_path(root, stage, task)
                    if _process_mentions(answers):
                        continue
                    records = json.loads(
                        Path(formal["tasks"][task]["test_file"]).read_text(encoding="utf-8")
                    )
                    if _validate_answers(answers, records, allow_partial=True) == len(records):
                        continue
                    selected = cell
                    break
                if selected is None:
                    break
                stage, task = selected
                active_cells.add(selected)
                gpu_lock = _try_lock(control / "locks" / "gpu{}.lock".format(gpu))
                cell_lock = _try_lock(
                    control / "locks" / "t{}_task{}.lock".format(stage, task)
                )
                if gpu_lock is None or cell_lock is None:
                    if gpu_lock is not None:
                        gpu_lock.close()
                    if cell_lock is not None:
                        cell_lock.close()
                    continue
                selection = _ensure_cell_selections(
                    root, formal, method, stage, task, gpu, args.python,
                    args.cache_manifest,
                )
                answers, command = generation_command(
                    root, formal, method, stage, task, args.python,
                    selection_manifest=selection,
                )
                answers.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = root / "task{}".format(stage) / "committed"
                contract = _contract(
                    root, formal, stage, task, selection, args.fast_selection_plan
                )
                contract_path = control / "contracts" / "t{}_task{}.json".format(stage, task)
                _atomic_json(contract_path, contract)
                _adopt_partial(answers, contract, checkpoint)
                command.extend(["--resume", "--resume-contract", str(contract_path)])
                if args.fast_selection_plan:
                    command.append("--fast-selection-plan")
                log_path = answers.with_name("generation.dynamic.log")
                log_handle = log_path.open("a", encoding="utf-8")
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    PYTHONPATH=_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", ""),
                )
                process = subprocess.Popen(
                    command, env=env, stdout=log_handle,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                running[gpu] = {
                    "cell": selected, "process": process, "log": log_handle,
                    "gpu_lock": gpu_lock, "cell_lock": cell_lock,
                }
                _write_state(
                    control, stage, task, status="running", gpu=gpu,
                    pid=process.pid, attempts=attempts.get(selected, 0),
                    contract_sha256=_sha256(contract_path),
                )
            _rebuild_matrix(root)
            time.sleep(args.poll_seconds)
    except BaseException:
        for job in running.values():
            process = job["process"]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            job["log"].close()
            job["gpu_lock"].close()
            job["cell_lock"].close()
        raise
    finally:
        score_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
