"""Resumable V9-S lower-triangular UCIT evaluation driver.

Row ``A[t][0..t]`` is produced with the task-``t`` committed key pool: the same
pool serves every cell of the row, which is what makes the row a measurement of
one model rather than of ``t+1`` different ones.

A row may be built in more than one pass: ``--tasks`` narrows the work to the
cells asked for and the matrix merges rather than replaces, so the per-task pass
can write the diagonal cell ``A[t][t]`` beside the training that produced it and
a later sweep can add the cross-task cells beside it.  Both passes make the same
measurement, with the same pool and the same fixed queries; what differs is
only when the GPU time is spent.

Three properties are load-bearing and each is enforced rather than assumed:

**Record order.**  ``llava.eval.eval_caption`` maps the *i*-th answer line to
COCO ``image_id = i+1``, so a reordered or short answer file does not fail -- it
silently scores the right captions against the wrong images.  Generation is
sharded across GPUs for speed, and the shards are merged back by sample id and
re-emitted in the original record order, which both restores the order and
proves nothing was dropped.

**The original scorer.**  Scoring is delegated to the Hyper-LLaVA scorers
(``_score_answers``), never re-implemented, so the V9 numbers are comparable
with the V7/V8 matrix by construction.

**Resumability.**  A cell whose metric already exists is skipped, so a run that
dies on cell 4 of row 3 does not regenerate cells 0-3.  ``--dry-run`` validates
every cell input and prints the work plan without loading a model.

``--cells-json`` is produced by ``build_cells`` / ``--build-cells`` from the
formal test files and the run's query manifest; no cell path is hand-written.
"""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .evaluation import build_selection_manifest

#: The formal UCIT order, compose index 0..5.
TASK_NAMES = ("ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k")


class V9FormalEvaluationError(RuntimeError): pass


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def _sample_id(record) -> str:
    value = record.get("question_id", record.get("id"))
    if value is None:
        raise V9FormalEvaluationError("a test record has neither question_id nor id")
    return str(value)


def load_cells(path):
    cells=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cells, list): raise V9FormalEvaluationError("cells JSON must be a list")
    required={"task_index","question_file"}
    for cell in cells:
        if not required <= set(cell): raise V9FormalEvaluationError("each cell needs task_index, question_file")
        if not (cell.get("query_cache") or cell.get("query_cache_manifest")):
            raise V9FormalEvaluationError(
                "cell for task {} declares no query source: committed-pool routing "
                "needs the precomputed query cache (query_cache or "
                "query_cache_manifest)".format(cell.get("task_index"))
            )
    return sorted(cells, key=lambda item:int(item["task_index"]))


def build_cells(
    instructions_root: str | Path,
    query_cache_manifest: Optional[str] = None,
    query_cache_root: Optional[str] = None,
    legacy_query_cache_root: Optional[str] = None,
    tasks: Sequence[int] = range(len(TASK_NAMES)),
    annotation_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """The cells of a row, derived from the formal test files themselves.

    Nothing about a cell is typed by hand: the question file is located under
    the UCIT instructions root for the task's own name, and the annotation for
    a caption task is taken from the ``val_coco_type_3000.json`` sitting beside
    it, which is what ``_score_answers`` would otherwise have to guess.
    """
    root = Path(instructions_root).expanduser()
    cells: List[Dict[str, Any]] = []
    for task in tasks:
        task = int(task)
        if not 0 <= task < len(TASK_NAMES):
            raise V9FormalEvaluationError("task index must be 0..5, got {}".format(task))
        name = TASK_NAMES[task]
        question_file = root / name / "test_3000.json"
        if not question_file.is_file():
            raise V9FormalEvaluationError("missing formal test file {}".format(question_file))
        cell: Dict[str, Any] = {"task_index": task, "question_file": str(question_file)}
        if query_cache_manifest:
            cell["query_cache_manifest"] = str(query_cache_manifest)
            cell["split"] = "test"
            if query_cache_root:
                cell["query_cache_root"] = str(query_cache_root)
        else:
            if not legacy_query_cache_root:
                raise V9FormalEvaluationError(
                    "no query source: pass a query cache manifest or a legacy "
                    "query cache root"
                )
            cache = Path(legacy_query_cache_root).expanduser() / name / "test.json"
            if not cache.is_file():
                raise V9FormalEvaluationError("missing query cache {}".format(cache))
            cell["query_cache"] = str(cache)
        annotation = (Path(annotation_root).expanduser() / name / "val_coco_type_3000.json"
                      if annotation_root else question_file.with_name("val_coco_type_3000.json"))
        if annotation.is_file():
            cell["annotation_file"] = str(annotation)
        cells.append(cell)
    return cells


def plan_cells(root, stage, cells, key_state):
    """Where each requested cell's artefacts live.

    A cell ``A[stage][task]`` may only test against a task the stage has already
    been trained on, so every requested task must lie in ``0..stage``.  What is
    *not* required is that all of them are requested at once: the per-task pass
    asks for the diagonal cell alone and the final sweep asks for whichever
    cross-task cells are still missing, and both are the same measurement made
    at different times.
    """
    allowed=set(range(int(stage)+1))
    actual=[int(cell["task_index"]) for cell in cells]
    if not actual:
        raise V9FormalEvaluationError(f"stage {stage} was given no cells to evaluate")
    if len(set(actual)) != len(actual):
        raise V9FormalEvaluationError(f"stage {stage} repeats a cell: {actual}")
    outside=sorted(set(actual) - allowed)
    if outside:
        raise V9FormalEvaluationError(
            f"stage {stage} has not been trained on tasks {outside}; it may only "
            f"be evaluated on {sorted(allowed)}"
        )
    plan=[]
    for cell in cells:
        task=int(cell["task_index"]); selection=root/"evaluation"/"selections"/f"t{stage}"/f"task{task}"/"selections.json"
        answers=root/"evaluation"/"predictions"/f"t{stage}"/f"task{task}"/"answers.jsonl"
        metric=root/"evaluation"/"scores"/f"t{stage}"/f"task{task}"/"metric.json"
        plan.append({"task_index":task,"question_file":str(cell["question_file"]),"query_cache":cell.get("query_cache"),
                     "query_cache_manifest":cell.get("query_cache_manifest"),"query_cache_root":cell.get("query_cache_root"),
                     "split":cell.get("split","test"),
                     "annotation_file":cell.get("annotation_file"),"selection":str(selection),"answers":str(answers),
                     "metric":str(metric),"key_state":str(key_state)})
    return plan


def write_selection(item):
    target=Path(item["selection"]); target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file(): return
    payload=build_selection_manifest(
        item["key_state"], item.get("query_cache"),
        query_cache_manifest=item.get("query_cache_manifest"),
        task_index=int(item["task_index"]),
        split=str(item.get("split") or "test"),
        query_cache_root=item.get("query_cache_root"),
    )
    temporary=target.with_suffix(target.suffix+".tmp")
    temporary.write_text(json.dumps(payload["rows"],indent=2,sort_keys=True)+"\n",encoding="utf-8"); temporary.replace(target)


def merge_shards(parts: Sequence[Path], records: Sequence[Dict[str, Any]], output: Path) -> Dict[str, Any]:
    """Merge shard files by sample id and re-emit them in record order.

    Re-emitting in ``records`` order is the whole point: the caption scorer
    reads row *i* as ``image_id = i+1``, so the merge has to be a positional
    reconstruction, not a concatenation that happens to look right.  The
    id-keyed step is also what proves completeness -- a shard that silently
    produced fewer rows would otherwise shorten the answer file and mis-score
    every cell after the gap.
    """
    expected = [_sample_id(record) for record in records]
    if len(set(expected)) != len(expected):
        raise V9FormalEvaluationError("test records have duplicate ids; a positional merge would be ambiguous")
    expected_set = set(expected)
    by_id: Dict[str, Any] = {}
    for part in parts:
        if not part.is_file():
            raise V9FormalEvaluationError("missing shard output {}".format(part))
        with part.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample = str(row.get("question_id", row.get("id")))
                if sample not in expected_set:
                    raise V9FormalEvaluationError("{}:{} unexpected question_id {}".format(part, line_number, sample))
                if sample in by_id:
                    raise V9FormalEvaluationError("duplicate question_id {} across shards".format(sample))
                by_id[sample] = row
    missing = [sample for sample in expected if sample not in by_id]
    if missing or len(by_id) != len(expected):
        raise V9FormalEvaluationError(
            "incomplete shard merge: got {}, expected {}, missing first {}".format(len(by_id), len(expected), missing[:5])
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".merging")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in expected:
            handle.write(json.dumps(by_id[sample], ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    return {"rows": len(expected), "shards": [str(part) for part in parts]}


def _generation_command(a, item) -> List[str]:
    return [a.python, "-m", "compose.eval.eval_task", "--adapter-kind", "compose",
            "--model-path", a.model_path, "--checkpoint-dir", a.checkpoint_dir,
            "--projector-path", a.projector_path, "--vision-tower", a.vision_tower,
            "--question-file", item["question_file"], "--image-folder", a.image_folder,
            "--selection-manifest", item["selection"],
            "--load-only-manifest-experts", "--fast-selection-plan",
            "--cardinality-scale", "none", "--device", "cuda:0"]


def generate_answers(a, item, gpus: Sequence[str], records) -> Dict[str, Any]:
    """One cell's answers: one process per GPU, merged in record order."""
    answers = Path(item["answers"])
    answers.parent.mkdir(parents=True, exist_ok=True)
    gpus = [str(gpu).strip() for gpu in gpus if str(gpu).strip()]
    if len(set(gpus)) != len(gpus):
        raise V9FormalEvaluationError("duplicate GPU ids in {}".format(gpus))
    command = _generation_command(a, item)
    # A shard per GPU unless there are fewer records than GPUs, in which case
    # extra shards would be empty and ``eval_task`` refuses a chunk with no
    # records.
    shards = max(1, min(len(gpus), len(records)))
    if shards == 1:
        command += ["--answers-file", str(answers),
                    "--run-summary-file", str(answers.with_name("run_summary.json"))]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if completed.returncode:
            raise V9FormalEvaluationError(
                "generation failed for task {}:\n{}".format(item["task_index"], completed.stdout[-4000:])
            )
        return {"rows": len(records), "shards": [str(answers)], "mode": "single"}
    processes, parts, log_lines = [], [], []
    log_path = answers.with_name("generation.sharded.log")
    for index in range(shards):
        gpu = gpus[index]
        part = answers.with_name("answers.shard{}_of{}.jsonl".format(index, shards))
        summary = answers.with_name("run_summary.shard{}_of{}.json".format(index, shards))
        parts.append(part)
        shard_command = list(command)
        shard_command += ["--answers-file", str(part), "--run-summary-file", str(summary),
                          "--num-chunks", str(shards), "--chunk-idx", str(index)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   PYTHONPATH=_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", ""))
        processes.append((index, gpu, subprocess.Popen(shard_command, env=env, stdout=subprocess.PIPE,
                                                       stderr=subprocess.STDOUT, text=True)))
    errors = []
    for index, gpu, proc in processes:
        captured, _ = proc.communicate()
        log_lines.append("\n===== shard {}/{} GPU{} =====\n{}".format(index, shards, gpu, captured))
        if proc.returncode:
            errors.append("shard {}/{} GPU{} exit {}".format(index, shards, gpu, proc.returncode))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("".join(log_lines))
    if errors:
        raise V9FormalEvaluationError("; ".join(errors) + "; see {}".format(log_path))
    merged = merge_shards(parts, records, answers)
    merged["mode"] = "sharded"
    merged["gpus"] = gpus[:shards]
    return merged


def cell_lock(root: Path, stage: int, task: int):
    """An ``flock`` on ``locks/eval_tX_taskY.lock``, held while a cell is built.

    Two passes can legitimately touch one cell -- the per-task pass writes the
    diagonal, the sweep adds the cross-task cells -- and a cell that was scored
    in between is skipped rather than repeated.  The lock covers the window the
    file test cannot: two workers that both look before either writes would
    otherwise generate the same 3000 answers twice and race on the merge.
    """
    directory = Path(root) / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "eval_t{}_task{}.lock".format(int(stage), int(task))).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise V9FormalEvaluationError(
            "another worker is building cell A[{}][{}] right now; it must not be "
            "generated twice".format(int(stage), int(task))
        )
    return handle


def evaluate_row(a, root: Path, stage: int, cells, gpus: Sequence[str]) -> Dict[str, Any]:
    """Produce every requested cell of a row, skipping the ones already scored."""
    from compose.eval.formal_ucit_eval import _mirror_to_hyper_layout, _score_answers, _update_matrix

    plan = plan_cells(root, stage, cells, a.key_state)
    metrics: List[Dict[str, Any]] = []
    produced: List[Dict[str, Any]] = []
    for item in plan:
        task = int(item["task_index"])
        metric_path = Path(item["metric"])
        if metric_path.is_file():
            metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
            continue
        handle = cell_lock(root, stage, task)
        try:
            if metric_path.is_file():
                metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
                continue
            write_selection(item)
            records = json.loads(Path(item["question_file"]).read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                raise V9FormalEvaluationError("empty or non-list test file {}".format(item["question_file"]))
            generation = generate_answers(a, item, gpus, records)
            metric = _score_answers(root, stage, task, Path(item["answers"]),
                                    annotation_file=item.get("annotation_file"))
            metric["generation"] = generation
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            metrics.append(metric)
            produced.append({"task_index": task, "value": metric["value"], "generation": generation})
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    metrics.sort(key=lambda value: int(value["task_id"]))
    _update_matrix(root, stage, metrics)
    _mirror_to_hyper_layout(root, stage, metrics)
    marker = root / "evaluation" / "t{}".format(stage) / "row_complete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    # What the stage directory now holds, which after an incremental fill is not
    # the same as what this invocation produced: writing only ``metrics`` here
    # would make a sweep that added one cross-task cell look like it had
    # replaced the row.
    written = sorted(
        int(path.parent.name[4:])
        for path in (root / "evaluation" / "scores" / "t{}".format(stage)).glob("task*/metric.json")
        if path.parent.name.startswith("task")
    )
    marker.write_text(json.dumps({
        "stage": int(stage), "cells": written,
        "values": {str(int(m["task_id"])): m["value"] for m in metrics},
        "produced_now": produced,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"stage": int(stage), "cells": [int(m["task_id"]) for m in metrics],
            "produced_now": produced,
            "values": {str(int(m["task_id"])): m["value"] for m in metrics}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True); p.add_argument("--stage",type=int,required=True)
    p.add_argument("--cells-json",required=True); p.add_argument("--key-state",required=True)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--build-cells",action="store_true",
                   help="write --cells-json from the formal test files instead of reading it")
    p.add_argument("--instructions-root")
    p.add_argument("--query-cache-manifest")
    p.add_argument("--query-cache-root")
    p.add_argument("--legacy-query-cache-root")
    p.add_argument("--tasks",default=None,
                   help="comma-separated evaluated-task indices to build cells for; "
                        "defaults to 0..stage.  Lets the per-task pass ask for its "
                        "diagonal cell and the final sweep for the cells still missing")
    p.add_argument("--gpus",default="0",
                   help="comma-separated GPU ids for sharded generation, e.g. 0,1,2,3")
    p.add_argument("--python",default="python"); p.add_argument("--checkpoint-dir"); p.add_argument("--model-path"); p.add_argument("--projector-path"); p.add_argument("--vision-tower"); p.add_argument("--image-folder"); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(); root=Path(a.root)
    if a.build_cells:
        if not a.instructions_root:
            raise V9FormalEvaluationError("--build-cells needs --instructions-root")
        # A cell ``A[stage][task]`` with ``task > stage`` is not a measurement
        # that exists -- the stage has not been trained on that task -- so the
        # builder stops at the stage unless the caller names a narrower set.
        # Emitting all six tasks here would make every row below the last one
        # fail its own integrity check at evaluation time, hours into the chain.
        wanted = (sorted(int(item) for item in a.tasks.split(",") if item.strip())
                  if a.tasks else list(range(int(a.stage) + 1)))
        cells = build_cells(a.instructions_root, a.query_cache_manifest, a.query_cache_root,
                            a.legacy_query_cache_root, tasks=wanted)
        target = Path(a.cells_json); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(cells, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"cells": len(cells), "written": str(target)}, indent=2))
        return
    cells = load_cells(a.cells_json)
    if a.dry_run:
        print(json.dumps({"stage":a.stage,"cells":plan_cells(root,a.stage,cells,a.key_state),"gpu_work_started":False},indent=2)); return
    required=(a.checkpoint_dir,a.model_path,a.projector_path,a.vision_tower,a.image_folder)
    if not all(required): raise V9FormalEvaluationError("GPU execution requires checkpoint/model/projector/vision/image paths")
    gpus = [item.strip() for item in a.gpus.split(",") if item.strip()]
    if not gpus:
        raise V9FormalEvaluationError("--gpus must name at least one device")
    result = evaluate_row(a, root, a.stage, cells, gpus)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__=="__main__": main()
