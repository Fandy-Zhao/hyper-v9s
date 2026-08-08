#!/usr/bin/env python3
"""Build the Compose P1-Real task queue (configs/compose_p1_real_tasks.jsonl).

Frozen training protocol (identical to synthetic P1):
  rank 8, alpha 16, dropout 0, target q/k/v/o/gate/up/down, epoch 1,
  global batch 64, AdamW lr 2e-4, cosine, warmup 0.03, bf16,
  model_max_length 2048, gradient checkpointing, seeds 0/1/2.
"""

import argparse
import json
from pathlib import Path

ROOT = Path("/home/zhaozhuofan/Hyper-LlaVA")
OUTPUT_ROOT = ROOT / "outputs" / "compose_p1_real_20260803T120000Z"
DATA_ROOT = OUTPUT_ROOT / "data"
CKPT_ROOT = Path("/data/ckpt/zhaozhuofan/compose/p1_real")
MODEL_PATH = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
DEEPSPEED = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/deepspeed"

EXPECTED_RANK8 = 19988480
SEEDS = (0, 1, 2)


def train_task(seed, expert, data_name, expert_id, origin, tags, master_port):
    task_id = "train_{}_seed{}".format(expert, seed)
    checkpoint = CKPT_ROOT / "seed{}".format(seed) / expert
    command = (
        # NOTE: the deepspeed launcher overrides CUDA_VISIBLE_DEVICES and would
        # place workers on physical GPU 0 (occupied); torch.distributed.run
        # respects CUDA_VISIBLE_DEVICES, which the scheduler sets per task.
        "{py} -m torch.distributed.run --nproc_per_node 1 --master_port {port} "
        "-m compose.train.train_compose --deepspeed {root}/scripts/zero2.json "
        "--model_name_or_path {model} --pretrain_mm_mlp_adapter {model}/mm_projector.bin "
        "--version v1 --data_path {data} --image_folder {img} "
        "--vision_tower {vt} "
        "--compose_rank 8 --compose_alpha 16 --compose_dropout 0 "
        "--compose_expert_ids {eid} --compose_trainable_expert_ids {eid} "
        "--compose_expert_name {expert} --compose_origin_task_id {origin} --compose_expert_tags {tags} "
        "--compose_gates 1 --compose_gate_normalization none "
        "--expected_adapter_parameters {params} "
        "--mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 "
        "--mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad "
        "--group_by_modality_length True --bf16 True --output_dir {ckpt} "
        "--num_train_epochs 1 --save_strategy epoch "
        "--per_device_train_batch_size 8 --per_device_eval_batch_size 16 "
        "--gradient_accumulation_steps 8 --evaluation_strategy no "
        "--learning_rate 2e-4 --weight_decay 0 --warmup_ratio 0.03 "
        "--lr_scheduler_type cosine --logging_steps 1 --tf32 True "
        "--model_max_length 2048 --gradient_checkpointing True "
        "--dataloader_num_workers 4 --report_to none --seed {seed}"
    ).format(
        py=PYTHON, port=master_port, root=ROOT, model=MODEL_PATH, data=data_name,
        img=DATA_ROOT, vt=VISION_TOWER, eid=expert_id, expert=expert,
        origin=origin, tags=tags, params=EXPECTED_RANK8, ckpt=checkpoint, seed=seed,
    )
    return {
        "id": task_id,
        "stage": "p1_real_train",
        "config": "frozen_p1_protocol_v1",
        "seed": seed,
        "gpu_requirement": 1,
        "command": command,
        # output_dir must be the checkpoint dir: the scheduler's completion
        # check (completion_files) validates against output_dir
        "output_dir": str(checkpoint),
        "checkpoint": str(checkpoint),
        "dependencies": [],
        "status": "PENDING",
        "batch_size": 8,
        "oom_batch_sizes": [8, 4, 2, 1],
        "completion_files": ["compose_experts.bin", "compose_experts.json"],
    }


def assemble_task(seed, sources, master_port):
    task_id = "assemble_seed{}".format(seed)
    output = CKPT_ROOT / "assembled" / "seed{}".format(seed) / "independent_b_c"
    parts = []
    for expert, expert_id in sources:
        parts.append("--source {} {}".format(CKPT_ROOT / "seed{}".format(seed) / expert, expert_id))
    command = "{} -m compose.experts.assemble {} --output-dir {}".format(
        PYTHON, " ".join(parts), output)
    return {
        "id": task_id,
        "stage": "p1_real_assemble",
        "config": "independent_b_c",
        "seed": seed,
        "gpu_requirement": 1,
        "command": command,
        "output_dir": str(output),
        "checkpoint": str(output),
        "dependencies": ["train_expert_b_seed{}".format(seed), "train_expert_c_seed{}".format(seed)],
        "status": "PENDING",
        # non-empty: the scheduler iterates batch sizes as attempts; assembly
        # ignores batch_size (CPU-only, no {batch_size} placeholder)
        "oom_batch_sizes": [8],
        "completion_files": ["compose_experts.bin", "compose_experts.json"],
    }


def eval_task(seed, test_name, calib_name, extra, master_port, mode_filter="",
              skip_c3=False, samples="", images_root=None):
    task_id = "eval_{}_seed{}{}".format(test_name, seed, extra)
    checkpoint = CKPT_ROOT / "assembled" / "seed{}".format(seed) / "independent_b_c"
    output_dir = OUTPUT_ROOT / "predictions" / "p1_real" / "seed{}".format(seed) / test_name
    images = images_root or DATA_ROOT
    command = (
        "{py} -m compose.eval.compose_p1_real "
        "--model-path {model} --vision-tower {vt} --checkpoint {ckpt} "
        "--calibration-questions {calib} --test-questions {test} --images {img} "
        "--expert-ids 1,2 --pair-name independent_b_c "
        "--checkpoint-seed {seed} --analysis-seed {aseed} --output-dir {out} "
        "--batch-size {{batch_size}} --device cuda:0 --max-new-tokens 32 "
        "{mode} {skip} {samples}"
    ).format(
        py=PYTHON, model=MODEL_PATH, vt=VISION_TOWER, ckpt=checkpoint,
        calib=DATA_ROOT / "records" / "{}.json".format(calib_name),
        test=DATA_ROOT / "records" / "{}.json".format(test_name),
        img=images, seed=seed, aseed=seed % 3, out=output_dir,
        mode="--mode-filter {}".format(mode_filter) if mode_filter else "",
        skip="--skip-c3" if skip_c3 else "",
        samples="--test-samples {}".format(samples) if samples else "",
    )
    return {
        "id": task_id,
        "stage": "p1_real_eval",
        "config": "c0_c1_c2_c3",
        "seed": seed,
        "gpu_requirement": 1,
        "command": command,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "dependencies": ["assemble_seed{}".format(seed)],
        "status": "PENDING",
        "batch_size": 8,
        "oom_batch_sizes": [8, 4, 2, 1],
        "completion_files": ["summary.json", "per_sample.jsonl", "rms_statistics.json"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    tasks = []
    for seed in SEEDS:
        tasks.append(train_task(seed, "expert_b", str(DATA_ROOT / "instructions" / "B_train.json"),
                                1, "RealTextVQA/B_only", "real,textvqa,read", 29000 + seed))
        tasks.append(train_task(seed, "expert_c", str(DATA_ROOT / "instructions" / "C_train_full.json"),
                                2, "RealTextVQA/C_only", "real,textvqa,count", 29100 + seed))
        tasks.append(assemble_task(seed, [("expert_b", 1), ("expert_c", 2)], 29200 + seed))
        # main B+C evaluation (all modes)
        tasks.append(eval_task(seed, "BC_test", "BC_calib", "", 29300 + seed))
        # expert validity evals (B on B val, C on C val)
        tasks.append(eval_task(seed, "B_val", "B_val", "_single", 29400 + seed,
                               mode_filter="base,single_left", skip_c3=True))
        tasks.append(eval_task(seed, "C_val_full", "C_val_full", "_single", 29500 + seed,
                               mode_filter="base,single_right", skip_c3=True))
        # external cross-dataset evals (domain-shift evidence); the image
        # roots are the external datasets' own roots because the records
        # reference their official relative paths
        tasks.append(eval_task(seed, "external_ocrvqa_Bonly_test", "BC_calib", "_ext",
                               29600 + seed, mode_filter="base,single_left,c1", skip_c3=True,
                               samples="800",
                               images_root=Path("/data/ckpt/zhangyanqin/project/ModalPrompt/datasets")))
        tasks.append(eval_task(seed, "external_vqav2_count_test", "BC_calib", "_ext",
                               29700 + seed, mode_filter="base,single_right,c1", skip_c3=True,
                               samples="800",
                               images_root=Path("/data/dataset/zhaozhuofan/LiLoRA_datasets")))
    # layer-group diagnostics (single seed 0, after ablation data exists)
    task_file = output_root / "configs" / "compose_p1_real_tasks.jsonl"
    task_file.parent.mkdir(parents=True, exist_ok=True)
    task_file.write_text(
        "\n".join(json.dumps(task, sort_keys=True) for task in tasks) + "\n",
        encoding="utf-8")
    print("wrote {} tasks to {}".format(len(tasks), task_file))
    print("C4/C6 (joint rank-16 / task rank-8) NOT scheduled: natural B+C pool "
          "has 0 training samples after reserving calibration and test "
          "(data-limited, recorded in the report).")


if __name__ == "__main__":
    main()
