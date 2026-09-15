"""Resumable V9-S lower-triangular UCIT evaluation driver.

GPU execution uses only a committed key pool, fixed test queries and the common
UCIT generation/scoring harness.  ``--dry-run`` validates all cell inputs and
prints the exact work plan without loading a model or invoking generation.
"""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
from .evaluation import build_selection_manifest

class V9FormalEvaluationError(RuntimeError): pass

def load_cells(path):
    cells=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cells, list): raise V9FormalEvaluationError("cells JSON must be a list")
    required={"task_index","question_file","query_cache"}
    for cell in cells:
        if not required <= set(cell): raise V9FormalEvaluationError("each cell needs task_index, question_file, query_cache")
    return sorted(cells, key=lambda item:int(item["task_index"]))

def plan_cells(root, stage, cells, key_state):
    expected=list(range(int(stage)+1))
    actual=[int(cell["task_index"]) for cell in cells]
    if actual != expected: raise V9FormalEvaluationError(f"stage {stage} requires cells {expected}, got {actual}")
    plan=[]
    for cell in cells:
        task=int(cell["task_index"]); selection=root/"evaluation"/"selections"/f"t{stage}"/f"task{task}"/"selections.json"
        answers=root/"evaluation"/"predictions"/f"t{stage}"/f"task{task}"/"answers.jsonl"
        plan.append({"task_index":task,"question_file":str(cell["question_file"]),"query_cache":str(cell["query_cache"]),"annotation_file":cell.get("annotation_file"),"selection":str(selection),"answers":str(answers),"key_state":str(key_state)})
    return plan

def write_selection(item):
    target=Path(item["selection"]); target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file(): return
    payload=build_selection_manifest(item["key_state"],item["query_cache"])
    temporary=target.with_suffix(target.suffix+".tmp")
    temporary.write_text(json.dumps(payload["rows"],indent=2,sort_keys=True)+"\n",encoding="utf-8"); temporary.replace(target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True); p.add_argument("--stage",type=int,required=True)
    p.add_argument("--cells-json",required=True); p.add_argument("--key-state",required=True)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--python",default="python"); p.add_argument("--checkpoint-dir"); p.add_argument("--model-path"); p.add_argument("--projector-path"); p.add_argument("--vision-tower"); p.add_argument("--image-folder"); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(); root=Path(a.root); plan=plan_cells(root,a.stage,load_cells(a.cells_json),a.key_state)
    if a.dry_run:
        print(json.dumps({"stage":a.stage,"cells":plan,"gpu_work_started":False},indent=2)); return
    required=(a.checkpoint_dir,a.model_path,a.projector_path,a.vision_tower,a.image_folder)
    if not all(required): raise V9FormalEvaluationError("GPU execution requires checkpoint/model/projector/vision/image paths")
    from compose.eval.formal_ucit_eval import _score_answers, _update_matrix
    metrics=[]
    for item in plan:
        write_selection(item); answers=Path(item["answers"]); answers.parent.mkdir(parents=True,exist_ok=True)
        command=[a.python,"-m","compose.eval.eval_task","--adapter-kind","compose","--model-path",a.model_path,"--checkpoint-dir",a.checkpoint_dir,"--projector-path",a.projector_path,"--vision-tower",a.vision_tower,"--question-file",item["question_file"],"--image-folder",a.image_folder,"--answers-file",str(answers),"--run-summary-file",str(answers.with_name("run_summary.json")),"--selection-manifest",item["selection"],"--load-only-manifest-experts","--fast-selection-plan","--device",a.device]
        if subprocess.run(command).returncode: raise V9FormalEvaluationError(f"generation failed for task {item['task_index']}")
        metrics.append(_score_answers(root,a.stage,item["task_index"],answers,annotation_file=item.get("annotation_file")))
    _update_matrix(root,a.stage,metrics)
if __name__=="__main__": main()
