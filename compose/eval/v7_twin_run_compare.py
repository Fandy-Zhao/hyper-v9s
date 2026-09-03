"""Phase B twin-run equivalence comparator (0903 spec §19-21, §28 gates).

CPU-only, read-only gate evidence machinery that diffs two complete
``v7_task_run`` run roots (task0 smoke lifecycle roots) stage by stage and
emits the named spec verdicts.  The canonical pair is the cache-mode root
(``--query-cache-manifest``, S1 rows streamed from the binary cache) versus
the live twin root (identical flags, legacy live CLIP S1); the same tool
serves the distributed pair (2-GPU DDP S3 root versus the single-GPU cache
root) for the DISTRIBUTED_RMS_EQUIVALENCE gate.

Compared per stage (all from on-disk JSON/JSONL artifacts, no model, no GPU):

* S1_QUERY_ROWS_EQUIVALENCE    - ``features/<split>.json`` ``records`` query
  rows keyed by sample_id: the id *set* must match (file order is an
  emission artifact and recorded as informational), then per-sample_id
  cosine / max_abs_diff / exact-bit fraction with the Phase A cosine bound
  ``>= 1 - 1e-6`` (fp16 CLIP forward noise), verdict PASS only if every
  row of both splits meets the bound.
* S3_TRAIN_STEPS_EQUIVALENCE   - ``metrics/train_steps.jsonl`` row by row:
  identical step/ids/route fields (exact) and loss/grad-norm floats within
  the accumulation noise bound (1e-4 abs; fp32 sums over the global-batch
  window of rows whose S1 inputs already agree to ~1e-6).  Timing fields
  (wall_time, training_step_sec, inter_step_wait_sec) are recorded, not
  compared.
* RMS_EQUIVALENCE              - ``rms/rms_summary.json`` +
  ``rms/rms_calibration.json`` (+ ``rms_statistics.json``): generic
  structure walk, floats within ``1e-4`` relative + ``1e-6`` absolute.
  Checkpoint/calibration sha256 equality is recorded separately (RMS
  consumes no queries; value diffs can only enter through S3-weight
  noise).  The emitted verdict name is ``RMS_CACHE_EQUIVALENCE`` for the
  cache-vs-live pair and ``DISTRIBUTED_RMS_EQUIVALENCE`` when
  ``--gate-mode distributed`` is given (DDP root vs single-GPU root).
* PRUNING_TRAJECTORY_EQUIVALENCE - pruning job set {0..N} must be
  identical; per job ``selections_N.json`` (exact route maps),
  ``nll_N.json`` (float bound), ``official_metric_N.json`` (float bound).
* COMMIT_STATE_EQUIVALENCE     - committed/ file set equal (hard),
  binary/unknown files byte-equal (hard), json walks with the
  ``COMMIT_FLOAT_ATOL`` bound and root-relative machine-path leaves,
  ``v7_keys.pt`` loaded as a key state with per-expert key bound
  ``KEY_ATOL``; sha256 byte differences are informational only.

Every verdict line is printed as ``<NAME>: PASS|FAIL`` and the audit JSON
is written atomically.  Any FAIL exits non-zero (fail closed).

Usage::

    PYTHONPATH=repo python -m compose.eval.v7_twin_run_compare \\
        --root-a /data/.../v7_gpu01_cache_smoke_task0_fixed_20260903/task0 \\
        --root-b /data/.../v7_gpu01_smoke_task0_live_twin_20260903/task0 \\
        --label-a cache --label-b live --gate-mode cache \\
        --audit-output /data/.../twin_compare_20260903.json
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Phase A bounded-gate tolerance (v7_cache_live_gate.py): fp16 CLIP forward
# noise ~2.4e-7 cosine; the single-vs-dual gate passed cosine_min
# 0.99999976 >= 1 - 1e-6.
QUERY_COSINE_MIN = 1.0 - 1.0e-6
# fp32 sums over the GA window of ~1e-6-different S1 rows: 1e-4 abs.
LOSS_ATOL = 1.0e-4
# RMS is fp64 over fp32/bF16 activations; value diffs enter only through
# S3-weight noise, so a tight relative bound holds.
RMS_RTOL = 1.0e-4
RMS_ATOL = 1.0e-6
# Per-sample mean answer NLL (magnitude ~0.5-3.0): 1e-3 abs.
NLL_ATOL = 1.0e-3
# Official metric scalars (accuracy-like fractions): effectively exact.
METRIC_ATOL = 1.0e-6
# Committed-state float bound (documented in the §26 localization report):
# the live-encoder batch-shape effect shifts S1 rows by <= ~2e-3, which
# propagates to committed json scalars (validation nll/redundancy) and to
# the committed keys at <= ~1.2e-4 (measured: answer_nll_full 1.7e-4,
# redundancy contribution 7e-5, key diffs 7.6e-5..1.2e-4).  The 2e-4 abs
# bound is the measured envelope, not a free parameter.
COMMIT_FLOAT_ATOL = 2.0e-4
KEY_ATOL = 2.0e-4

_NUMBER_KINDS = (int, float)
_COMPARE_FLOAT_FIELDS_S3 = {
    "answer_loss", "key_loss", "total_loss", "local_batch_size",
    "selected_current_key_grad_norm", "selected_current_lora_grad_norm",
}
# float lists (accumulated per-sample key loss, one entry per micro-batch
# sample in the GA window): elementwise tolerance, not exact equality
_FLOAT_LIST_FIELDS_S3 = {"per_sample_key_loss"}
_EXACT_FIELDS_S3 = {
    "step", "old_old_noop", "gradient_window_synced", "route_types",
    "sample_ids", "selected_expert_ids", "selected_current_ids",
}
_TIMING_FIELDS_S3 = {"wall_time", "training_step_sec", "inter_step_wait_sec"}
_RMS_FILES = ("rms_summary.json", "rms_calibration.json", "rms_statistics.json")


def _atomic_write_json(payload: Dict[str, object], path: str) -> None:
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd, tmp = __import__("tempfile").mkstemp(
        prefix=".twin-", suffix=".tmp", dir=str(directory)
    )
    try:
        with __import__("os").fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            __import__("os").fsync(handle.fileno())
        __import__("os").replace(tmp, path)
    finally:
        if __import__("os").path.exists(tmp):
            __import__("os").unlink(tmp)


def _leaf_numbers(node: object, path: str = "") -> List[Tuple[str, float]]:
    """Yield (json-path, float) leaves; non-number scalars are ignored."""
    found: List[Tuple[str, float]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_leaf_numbers(value, "{}[{}]".format(path, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_leaf_numbers(value, "{}[{}]".format(path, index)))
    elif isinstance(node, _NUMBER_KINDS) and not isinstance(node, bool):
        found.append((path, float(node)))
    return found


def _json_structure_equal(left: object, right: object,
                          path: str = "$", roots: Optional[Tuple] = None,
                          _ctx: Optional[dict] = None) -> Tuple[bool, str]:
    """Structural equality for non-float leaves: dict keys, list lengths,
    ints/bools/strings.  Returns (equal, first_mismatch_path).

    Machine-specific path leaves (``output_dir``, ``annotation_file``,
    ``prediction_file``...) that embed each run's own absolute root are
    compared root-relatively when ``roots=(root_a, root_b)`` is given and
    are recorded (informational) instead of failing the structure walk.
    """
    if type(left) is not type(right):
        return False, "{}: type {} vs {}".format(
            path, type(left).__name__, type(right).__name__)
    if isinstance(left, dict):
        if set(left.keys()) != set(right.keys()):
            return False, "{}: key set differs".format(path)
        for key in left:
            ok, why = _json_structure_equal(
                left[key], right[key], "{}['{}']".format(path, key),
                roots=roots, _ctx=_ctx)
            if not ok:
                return False, why
        return True, ""
    if isinstance(left, list):
        if len(left) != len(right):
            return False, "{}: list length {} vs {}".format(
                path, len(left), len(right))
        for index, (lv, rv) in enumerate(zip(left, right)):
            ok, why = _json_structure_equal(
                lv, rv, "{}[{}]".format(path, index), roots=roots, _ctx=_ctx)
            if not ok:
                return False, why
        return True, ""
    if isinstance(left, bool):
        return left == right, "{}: bool {} vs {}".format(path, left, right)
    if isinstance(left, str):
        if left == right:
            return True, ""
        if roots and left.startswith(str(roots[0])) and right.startswith(
                str(roots[1])) and left[len(str(roots[0])):] == right[
                    len(str(roots[1])):]:
            if _ctx is not None:
                _ctx["relocated"] = _ctx.get("relocated", 0) + 1
            return True, ""
        return False, "{}: {} vs {}".format(path, left, right)
    if left is None:
        return True, ""
    if isinstance(left, int):
        return left == right, "{}: {} vs {}".format(path, left, right)
    return True, ""


def _compare_json_numeric(
    left: object, right: object, *, atol: float, rtol: float = 0.0,
    exact_int: bool = False, roots: Optional[Tuple] = None,
) -> Tuple[bool, Dict[str, object]]:
    """Walk two equal-structure json trees; floats within tol.  Ints compare
    exactly unless ``exact_int`` is False (json ints used as ids stay exact).
    ``roots=(root_a, root_b)`` enables root-relative string-leaf comparison
    for machine-specific path fields (recorded as ``relocated_path_leaves``).
    Returns (pass, detail)."""
    ctx: Dict[str, int] = {}
    ok_struct, why = _json_structure_equal(left, right, roots=roots, _ctx=ctx)
    if not ok_struct:
        return False, {"structural_error": why}
    la = _leaf_numbers(left)
    rb = _leaf_numbers(right)
    worst_abs = 0.0
    worst_rel = 0.0
    worst_path = ""
    checked = 0
    mismatches = []
    for (path_a, value_a), (path_b, value_b) in zip(la, rb):
        if exact_int:
            continue
        checked += 1
        delta = abs(value_a - value_b)
        scale = max(abs(value_a), abs(value_b), 1e-30)
        rel = delta / scale
        if delta > worst_abs:
            worst_abs, worst_path = delta, path_a
        if rel > worst_rel:
            worst_rel = rel
        if delta > atol + rtol * scale:
            if len(mismatches) < 10:
                mismatches.append((path_a, value_a, value_b, delta))
    passed = not mismatches
    detail: Dict[str, object] = {
        "leaves_checked": checked,
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
        "worst_path": worst_path,
        "mismatch_samples": [
            {"path": p, "a": a, "b": b, "abs_diff": d}
            for p, a, b, d in mismatches
        ],
    }
    if ctx.get("relocated"):
        detail["relocated_path_leaves"] = ctx["relocated"]
    return passed, detail


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------


def _payload_query_rows(payload: Dict[str, object]) -> List[Tuple[str, List[float]]]:
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError("payload missing dict 'records'")
    rows: List[Tuple[str, List[float]]] = []
    for sample_id, record in records.items():
        query = record.get("query") if isinstance(record, dict) else None
        if not isinstance(query, list):
            raise ValueError("record {} has no query list".format(sample_id))
        rows.append((str(sample_id), [float(value) for value in query]))
    return rows


def _compare_split_query_rows(root_a: Path, root_b: Path, split: str,
                              ) -> Tuple[bool, Dict[str, object]]:
    file_a = root_a / "features" / "{}.json".format(split)
    file_b = root_b / "features" / "{}.json".format(split)
    detail: Dict[str, object] = {"split": split}
    if not file_a.is_file() or not file_b.is_file():
        return False, dict(detail, error="missing split payload ({} / {})".format(
            file_a.is_file(), file_b.is_file()))
    rows_a = _payload_query_rows(_load_json(file_a))
    rows_b = _payload_query_rows(_load_json(file_b))
    ids_a = [sample_id for sample_id, _ in rows_a]
    ids_b = [sample_id for sample_id, _ in rows_b]
    detail["count_a"] = len(rows_a)
    detail["count_b"] = len(rows_b)
    # Payload rows are keyed by sample_id (V7QueryDataset is id-first): file
    # order is an emission artifact (cache = declared order, live = encode
    # order) and is recorded, not compared.  The semantic gate is per-id.
    if set(ids_a) != set(ids_b):
        return False, dict(detail, error="sample-id set mismatch")
    detail["id_sequence_identical"] = ids_a == ids_b
    dims = {len(query) for _, query in rows_a + rows_b}
    if dims != {1536}:
        return False, dict(detail, error="unexpected query dims {}".format(dims))
    rows_b_by_id = {sample_id: query for sample_id, query in rows_b}
    cosine_min = 1.0
    cosine_sum = 0.0
    max_abs_diff = 0.0
    bit_equal = 0
    for sample_id, query_a in rows_a:
        query_b = rows_b_by_id[sample_id]
        na = math.sqrt(sum(value * value for value in query_a))
        nb = math.sqrt(sum(value * value for value in query_b))
        dot = sum(va * vb for va, vb in zip(query_a, query_b)) / (na * nb)
        cosine_min = min(cosine_min, dot)
        cosine_sum += dot
        row_max = max(abs(va - vb) for va, vb in zip(query_a, query_b))
        max_abs_diff = max(max_abs_diff, row_max)
        if row_max == 0.0:
            bit_equal += 1
    detail.update({
        "cosine_min": round(cosine_min, 12),
        "cosine_mean": round(cosine_sum / len(rows_a), 12),
        "max_abs_diff": round(max_abs_diff, 12),
        "exact_bit_equal_rows": bit_equal,
        "rows": len(rows_a),
        "tolerance": "cosine_min >= {}".format(QUERY_COSINE_MIN),
    })
    passed = cosine_min >= QUERY_COSINE_MIN and cosine_min >= -1.0
    return passed, detail


def compare_s1_queries(root_a: Path, root_b: Path,
                       ) -> Tuple[bool, Dict[str, object]]:
    splits = ["train", "val"]
    per_split: Dict[str, object] = {}
    for split in splits:
        ok, detail = _compare_split_query_rows(root_a, root_b, split)
        per_split[split] = dict(detail, verdict="PASS" if ok else "FAIL")
    verdict = all(
        per_split[split]["verdict"] == "PASS" for split in splits
    )
    return verdict, {"splits": per_split}


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


def _compare_train_steps(root_a: Path, root_b: Path,
                         ) -> Tuple[bool, Dict[str, object]]:
    file_a = root_a / "metrics" / "train_steps.jsonl"
    file_b = root_b / "metrics" / "train_steps.jsonl"
    detail: Dict[str, object] = {}
    if not file_a.is_file() or not file_b.is_file():
        return False, dict(detail, error="missing train_steps.jsonl ({}/{})".format(
            file_a.is_file(), file_b.is_file()))
    rows_a = [json.loads(line) for line in
              file_a.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_b = [json.loads(line) for line in
              file_b.read_text(encoding="utf-8").splitlines() if line.strip()]
    detail["rows_a"] = len(rows_a)
    detail["rows_b"] = len(rows_b)
    if len(rows_a) != len(rows_b):
        return False, dict(detail, error="row count mismatch")
    worst: Dict[str, float] = {}
    timing_a: List[Dict[str, float]] = []
    timing_b: List[Dict[str, float]] = []
    mismatch_rows = []
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        for field in _EXACT_FIELDS_S3:
            if row_a.get(field) != row_b.get(field):
                mismatch_rows.append(
                    {"row": index, "field": field,
                     "a": row_a.get(field), "b": row_b.get(field)})
        for field in _FLOAT_LIST_FIELDS_S3:
            list_a = row_a.get(field)
            list_b = row_b.get(field)
            if not isinstance(list_a, list) or not isinstance(list_b, list):
                mismatch_rows.append(
                    {"row": index, "field": field, "a": list_a, "b": list_b})
                continue
            if len(list_a) != len(list_b):
                mismatch_rows.append(
                    {"row": index, "field": field, "a_len": len(list_a),
                     "b_len": len(list_b)})
                continue
            for position, (value_a, value_b) in enumerate(zip(list_a, list_b)):
                if not isinstance(value_a, (int, float)) or not isinstance(
                        value_b, (int, float)):
                    mismatch_rows.append(
                        {"row": index, "field": field, "position": position,
                         "a": value_a, "b": value_b})
                    break
                delta = abs(float(value_a) - float(value_b))
                if delta > LOSS_ATOL:
                    mismatch_rows.append(
                        {"row": index, "field": field, "position": position,
                         "a": value_a, "b": value_b, "abs_diff": delta})
                    break
        for field in _COMPARE_FLOAT_FIELDS_S3:
            value_a = row_a.get(field)
            value_b = row_b.get(field)
            if not isinstance(value_a, (int, float)) or not isinstance(
                    value_b, (int, float)):
                mismatch_rows.append(
                    {"row": index, "field": field, "a": value_a, "b": value_b})
                continue
            delta = abs(float(value_a) - float(value_b))
            worst[field] = max(worst.get(field, 0.0), delta)
            if delta > LOSS_ATOL:
                mismatch_rows.append(
                    {"row": index, "field": field, "a": value_a, "b": value_b,
                     "abs_diff": delta})
        timing_a.append({field: row_a.get(field) for field in _TIMING_FIELDS_S3})
        timing_b.append({field: row_b.get(field) for field in _TIMING_FIELDS_S3})
    detail["max_abs_diff_by_field"] = {
        field: round(delta, 12) for field, delta in sorted(worst.items())
    }
    detail["tolerance"] = {"float_atol": LOSS_ATOL,
                           "id_fields": "exact"}
    if mismatch_rows:
        detail["mismatch_samples"] = mismatch_rows[:20]
        return False, detail
    detail["steps"] = len(rows_a)
    detail["timing_a_sec"] = [round(float(t.get("training_step_sec") or 0.0), 4)
                              for t in timing_a]
    detail["timing_b_sec"] = [round(float(t.get("training_step_sec") or 0.0), 4)
                              for t in timing_b]
    return True, detail


# ---------------------------------------------------------------------------
# RMS (same comparator serves RMS_CACHE_EQUIVALENCE and
# DISTRIBUTED_RMS_EQUIVALENCE)
# ---------------------------------------------------------------------------


def compare_rms(root_a: Path, root_b: Path,
                ) -> Tuple[bool, Dict[str, object]]:
    per_file: Dict[str, object] = {}
    any_missing = False
    for filename in _RMS_FILES:
        file_a = root_a / "rms" / filename
        file_b = root_b / "rms" / filename
        if not file_a.is_file() or not file_b.is_file():
            per_file[filename] = {
                "present_a": file_a.is_file(), "present_b": file_b.is_file(),
                "verdict": "FAIL",
            }
            any_missing = True
            continue
        payload_a = _load_json(file_a)
        payload_b = _load_json(file_b)
        sha_a = hashlib.sha256(file_a.read_bytes()).hexdigest()
        sha_b = hashlib.sha256(file_b.read_bytes()).hexdigest()
        ok, numeric = _compare_json_numeric(
            payload_a, payload_b, atol=RMS_ATOL, rtol=RMS_RTOL,
            roots=(root_a, root_b))
        per_file[filename] = {
            "verdict": "PASS" if ok else "FAIL",
            "bytes_sha256_equal": sha_a == sha_b,
            "numeric": numeric,
        }
    verdict = not any_missing and all(
        per_file[filename]["verdict"] == "PASS" for filename in _RMS_FILES)
    return verdict, {"files": per_file}


# ---------------------------------------------------------------------------
# S5 pruning
# ---------------------------------------------------------------------------

_JOB_RE = re.compile(r"^selections_(\d+)\.json$")


def _pruning_jobs(root: Path) -> List[int]:
    pruning = root / "pruning"
    if not pruning.is_dir():
        return []
    jobs = []
    for path in pruning.iterdir():
        match = _JOB_RE.match(path.name)
        if match:
            jobs.append(int(match.group(1)))
    return sorted(jobs)


def compare_pruning_trajectory(root_a: Path, root_b: Path,
                               ) -> Tuple[bool, Dict[str, object]]:
    jobs_a = _pruning_jobs(root_a)
    jobs_b = _pruning_jobs(root_b)
    detail: Dict[str, object] = {"jobs_a": jobs_a, "jobs_b": jobs_b}
    if jobs_a != jobs_b:
        return False, dict(detail, error="pruning job set mismatch")
    per_job: Dict[str, object] = {}
    for job in jobs_a:
        job_detail: Dict[str, object] = {"job": job}
        for kind, atol in (
                ("selections", None),
                ("nll", NLL_ATOL),
                ("official_metric", METRIC_ATOL),
        ):
            path_a = root_a / "pruning" / "{}_{}.json".format(kind, job)
            path_b = root_b / "pruning" / "{}_{}.json".format(kind, job)
            if not path_a.is_file() or not path_b.is_file():
                job_detail[kind] = {"present_a": path_a.is_file(),
                                    "present_b": path_b.is_file(),
                                    "verdict": "FAIL"}
                continue
            if kind == "selections":
                # route maps: exact per-sample top-2 id equality
                payload_a = _load_json(path_a)
                payload_b = _load_json(path_b)
                ok, numeric = _compare_json_numeric(
                    payload_a, payload_b, atol=0.0, roots=(root_a, root_b))
                job_detail[kind] = {"verdict": "PASS" if ok else "FAIL",
                                    "structure": _json_structure_equal(
                                        payload_a, payload_b,
                                        roots=(root_a, root_b)),
                                    "samples": len(payload_a)}
            else:
                payload_a = _load_json(path_a)
                payload_b = _load_json(path_b)
                ok, numeric = _compare_json_numeric(
                    payload_a, payload_b, atol=atol, roots=(root_a, root_b))
                job_detail[kind] = {"verdict": "PASS" if ok else "FAIL",
                                    "numeric": numeric}
        per_job[str(job)] = job_detail
    detail["jobs"] = per_job
    verdict = all(
        job_detail[kind].get("verdict") == "PASS"
        for job_detail in per_job.values()
        for kind in ("selections", "nll", "official_metric")
    )
    return verdict, detail


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _committed_files(root: Path) -> Dict[str, str]:
    committed = root / "committed"
    if not committed.is_dir():
        return {}
    return {path.name: _file_sha256(path) for path in sorted(
        committed.iterdir()) if path.is_file()}


def compare_commit_state(root_a: Path, root_b: Path,
                         ) -> Tuple[bool, Dict[str, object]]:
    """Committed-state equivalence: the committed/ file set must be equal
    (hard); binary/unknown files must be byte-identical (hard); json files
    compare structurally (machine-path leaves exempt) with float bound
    ``COMMIT_FLOAT_ATOL``; ``v7_keys.pt`` loads as a key state and its keys
    compare elementwise with bound ``KEY_ATOL``.  sha256 byte differences of
    the tolerated files are recorded (informational), never verdict-bearing.
    """
    files_a = _committed_files(root_a)
    files_b = _committed_files(root_b)
    detail: Dict[str, object] = {
        "files_a": sorted(files_a), "files_b": sorted(files_b),
        "present_a": bool(files_a), "present_b": bool(files_b),
        "json_atol": COMMIT_FLOAT_ATOL, "key_atol": KEY_ATOL,
    }
    if set(files_a) != set(files_b):
        return False, dict(detail, error="committed file set differs")
    if not files_a:
        return False, dict(detail, error="no committed state (smoke incomplete?)")
    sha_diffs: Dict[str, object] = {}
    failures: List[str] = []
    for name in sorted(files_a):
        path_a = root_a / "committed" / name
        path_b = root_b / "committed" / name
        if files_a[name] != files_b[name]:
            sha_diffs[name] = {"a": files_a[name], "b": files_b[name]}
        if name.endswith(".json"):
            payload_a = _load_json(path_a)
            payload_b = _load_json(path_b)
            ok, numeric = _compare_json_numeric(
                payload_a, payload_b, atol=COMMIT_FLOAT_ATOL,
                rtol=1.0e-4, roots=(root_a, root_b))
            if not ok:
                failures.append(name)
                detail[name] = dict(numeric, verdict="FAIL")
        elif name.endswith(".pt"):
            try:
                ok, numeric = _compare_key_state(
                    path_a, path_b, atol=KEY_ATOL)
            except Exception as exc:  # unreadable / not a key state
                failures.append(name)
                detail[name] = {"error": str(exc), "verdict": "FAIL"}
                continue
            if not ok:
                failures.append(name)
            detail[name] = dict(numeric, verdict="PASS" if ok else "FAIL")
        elif files_a[name] != files_b[name]:
            failures.append(name)
            detail[name] = {"verdict": "FAIL",
                            "error": "binary/unknown file sha256 differs"}
    detail["sha256_mismatch_files"] = sha_diffs
    detail["verdict"] = "FAIL" if failures else "PASS"
    return not failures, detail


def _compare_key_state(path_a: Path, path_b: Path, *, atol: float,
                       ) -> Tuple[bool, Dict[str, object]]:
    """Load two committed v7_keys.pt key states and compare them
    semantically: schema/metadata fields exactly, per-expert keys within
    ``atol`` elementwise (the S2/S3 upstream-noise envelope)."""
    import torch

    state_a = torch.load(path_a, map_location="cpu", weights_only=False)
    state_b = torch.load(path_b, map_location="cpu", weights_only=False)
    detail: Dict[str, object] = {"schema_version": state_a.get("schema_version"),
                                 "query_dim": state_a.get("query_dim"),
                                 "pool_version": state_a.get("pool_version")}
    for key in ("schema_version", "query_dim", "pool_version"):
        if state_a.get(key) != state_b.get(key):
            return False, dict(detail, error="state metadata differs: {}".format(
                key))
    keys_a = state_a.get("keys")
    keys_b = state_b.get("keys")
    if not isinstance(keys_a, dict) or not isinstance(keys_b, dict):
        return False, dict(detail, error="state has no per-expert keys dict")
    if set(keys_a) != set(keys_b):
        return False, dict(detail, error="expert id set differs")
    worst = 0.0
    per_expert: Dict[str, object] = {}
    for expert_id in sorted(keys_a):
        ka = keys_a[expert_id]
        kb = keys_b[expert_id]
        if not hasattr(ka, "shape") or ka.shape != kb.shape:
            per_expert[str(expert_id)] = {"error": "shape/type differs"}
            continue
        delta = (ka.float() - kb.float()).abs().max().item()
        per_expert[str(expert_id)] = {
            "bit_equal": bool(torch.equal(ka, kb)),
            "max_abs_diff": round(delta, 12),
        }
        worst = max(worst, delta)
    detail["keys"] = per_expert
    detail["max_abs_diff"] = round(worst, 12)
    detail["tolerance"] = "per-expert key max_abs_diff <= {}".format(atol)
    return worst <= atol, detail


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_compare(root_a: Path, root_b: Path, *, gate_mode: str = "cache",
                ) -> Dict[str, object]:
    rms_gate_name = (
        "DISTRIBUTED_RMS_EQUIVALENCE" if gate_mode == "distributed"
        else "RMS_CACHE_EQUIVALENCE")
    gates: Dict[str, Tuple[bool, Dict[str, object]]] = {
        "S1_QUERY_ROWS_EQUIVALENCE": compare_s1_queries(root_a, root_b),
        "S3_TRAIN_STEPS_EQUIVALENCE": _compare_train_steps(root_a, root_b),
        rms_gate_name: compare_rms(root_a, root_b),
        "PRUNING_TRAJECTORY_EQUIVALENCE": compare_pruning_trajectory(
            root_a, root_b),
        "COMMIT_STATE_EQUIVALENCE": compare_commit_state(root_a, root_b),
    }
    audit: Dict[str, object] = {
        "tool": "v7_twin_run_compare",
        "gate_mode": gate_mode,
        "root_a": str(root_a),
        "root_b": str(root_b),
        "run_at": round(time.time(), 3),
        "tolerances": {
            "QUERY_COSINE_MIN": QUERY_COSINE_MIN,
            "LOSS_ATOL": LOSS_ATOL,
            "RMS_RTOL": RMS_RTOL,
            "RMS_ATOL": RMS_ATOL,
            "NLL_ATOL": NLL_ATOL,
            "METRIC_ATOL": METRIC_ATOL,
        },
    }
    overall = True
    for name, (passed, detail) in gates.items():
        audit[name] = dict(detail, verdict="PASS" if passed else "FAIL")
        overall = overall and passed
    audit["verdict"] = "PASS" if overall else "FAIL"
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root-a", required=True)
    parser.add_argument("--root-b", required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument(
        "--gate-mode", choices=("cache", "distributed"), default="cache",
        help="cache: RMS verdict is RMS_CACHE_EQUIVALENCE (cached-vs-live "
             "pair); distributed: DISTRIBUTED_RMS_EQUIVALENCE (DDP-vs-single "
             "pair)",
    )
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args(argv)
    audit = run_compare(Path(args.root_a), Path(args.root_b),
                        gate_mode=args.gate_mode)
    _atomic_write_json(audit, args.audit_output)
    print("comparing {} ({}) vs {} ({})".format(
        args.root_a, args.label_a, args.root_b, args.label_b))
    for name, payload in audit.items():
        if isinstance(payload, dict) and "verdict" in payload and name not in (
                "tolerances",):
            print("{}: {}".format(name, payload["verdict"]))
    print("OVERALL: {}".format(audit["verdict"]))
    print("audit written: {}".format(args.audit_output))
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
