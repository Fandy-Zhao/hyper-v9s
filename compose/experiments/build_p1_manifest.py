"""Build the deterministic 12-job Compose P1 manifest."""

import argparse
import json
from pathlib import Path


ROOT = Path("/home/zhaozhuofan/Hyper-LlaVA")
MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
CKPT_ROOT = "/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1"
DATA = ROOT / "experiments/data/controlled_format_v1_training/instructions"
IMAGES = ROOT / "experiments/data/controlled_format_v1"
PAIRS = {
    "a_independent_b": ("A_plus_B", "0,1", "assembled/seed{seed}/a_independent_b"),
    "a_residual_b": ("A_plus_B", "0,1", "seed{seed}/residual_b"),
    "independent_b_c": ("B_plus_C", "1,2", "assembled/seed{seed}/independent_b_c"),
    "residual_b_c": ("B_plus_C", "1,2", "assembled/seed{seed}/residual_b_c"),
}
SEEDS = {0: 42, 1: 43, 2: 44}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    manifest = output_root / "configs/tasks.jsonl"
    resolved = output_root / "configs/p1_resolved_configs"
    resolved.mkdir(parents=True, exist_ok=True)
    tasks = []
    for analysis_seed, checkpoint_seed in SEEDS.items():
        for pair_name, (dataset, expert_ids, checkpoint_template) in PAIRS.items():
            checkpoint = "{}/{}".format(CKPT_ROOT, checkpoint_template.format(seed=checkpoint_seed))
            output_dir = output_root / "predictions/p1/{}_seed{}".format(pair_name, analysis_seed)
            task_id = "p1_{}_seed{}".format(pair_name, analysis_seed)
            command = (
                "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m compose.eval.compose_p1 "
                "--model-path {model} --vision-tower {vision} --checkpoint {checkpoint} "
                "--calibration-questions {validation} --test-questions {test} --images {images} "
                "--expert-ids {expert_ids} --pair-name {pair_name} --checkpoint-seed {checkpoint_seed} "
                "--analysis-seed {analysis_seed} --output-dir {output_dir} --batch-size {{batch_size}}"
            ).format(
                model=MODEL,
                vision=VISION,
                checkpoint=checkpoint,
                validation=DATA / dataset / "val_eval.json",
                test=DATA / dataset / "test_eval.json",
                images=IMAGES,
                expert_ids=expert_ids,
                pair_name=pair_name,
                checkpoint_seed=checkpoint_seed,
                analysis_seed=analysis_seed,
                output_dir=output_dir,
            )
            task = {
                "id": task_id,
                "stage": "p1",
                "config": "c0_c1_c2_c3",
                "seed": analysis_seed,
                "checkpoint_seed": checkpoint_seed,
                "gpu_requirement": 1,
                "command": command,
                "output_dir": str(output_dir),
                "checkpoint": checkpoint,
                "dependencies": [],
                "status": "PENDING",
                "batch_size": 8,
                "oom_batch_sizes": [8, 4, 2, 1],
                "completion_files": ["summary.json", "per_sample.jsonl", "rms_statistics.json"],
            }
            tasks.append(task)
            (resolved / "{}.json".format(task_id)).write_text(
                json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "tasks": len(tasks)}, sort_keys=True))


if __name__ == "__main__":
    main()
