"""Task0 fixed-composition capacity chain driver.

This module is intentionally additive.  It owns only the independent
``task0_capacity_chain_seed42`` run root and never mutates the existing
multi-r8 run.  The four trainable points use the same full raw Task0 train
file, with fixed all-sample activation and no router/clustering.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR = BASE_MODEL + "/mm_projector.bin"
TRAIN_FILE = "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/train.json"
TEST_FILE = "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json"
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
HYPER_CHECKPOINT = "/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/06_18/Task1"
BASE_POOL = str(REPO / "experiments/runs/compose_ucit_v62_formal_seed42/task0/base_only")
GPUS = (4, 5, 6, 7)
SEED = 42
VALIDATION_SAMPLES = 256

CONFIGS = {
    "single_r8": {"rank": 8, "alpha": 16.0, "expert_ids": [0], "seeds": {0: 4201}, "gpu": 4},
    "two_r8": {"rank": 8, "alpha": 16.0, "expert_ids": [0, 1], "seeds": {0: 4201, 1: 4202}, "gpu": 5},
    "four_r8": {"rank": 8, "alpha": 16.0, "expert_ids": [0, 1, 2, 3], "seeds": {0: 4201, 1: 4202, 2: 4203, 3: 4204}, "gpu": 6},
    "single_r16": {"rank": 16, "alpha": 32.0, "expert_ids": [0], "seeds": {0: 4201}, "gpu": 7},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def root_from(value):
    return Path(value).resolve()


def command_for(root, name, smoke=False):
    spec = CONFIGS[name]
    output = root / name / ("smoke_output" if smoke else "output")
    data = root / "data" / ("smoke_train.json" if smoke else "train.json")
    evaluation = root / "data" / ("smoke_val.json" if smoke else "validation_audit.json")
    ids = ",".join(str(x) for x in spec["expert_ids"])
    gates = ",".join("1" for _ in spec["expert_ids"])
    seeds = ",".join("{}={}".format(k, v) for k, v in sorted(spec["seeds"].items()))
    command = [
        PYTHON, "-m", "compose.train.train_compose",
        "--model_name_or_path", BASE_MODEL,
        "--vision_tower", VISION_TOWER,
        "--pretrain_mm_mlp_adapter", PROJECTOR,
        "--version", "v1",
        "--data_path", str(data), "--eval_data_path", str(evaluation),
        "--image_folder", IMAGE_FOLDER, "--image_aspect_ratio", "pad",
        "--compose_mode", "fixed", "--compose_rank", str(spec["rank"]),
        "--compose_alpha", str(spec["alpha"]), "--compose_dropout", "0.05",
        "--compose_expert_ids", ids, "--compose_trainable_expert_ids", ids,
        "--compose_gates", gates, "--compose_gate_normalization", "none",
        "--compose_expert_seeds", seeds,
        "--compose_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--mm_vision_select_layer", "-2", "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False", "--mm_projector_type", "mlp2x_gelu",
        "--mm_projector_lr", "2e-5", "--tune_mm_mlp_adapter", "False",
        "--output_dir", str(output), "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "16",
        "--gradient_accumulation_steps", "64", "--learning_rate", "2e-4",
        "--weight_decay", "0.", "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine", "--evaluation_strategy", "epoch",
        "--save_strategy", "no", "--logging_steps", "1", "--bf16", "True",
        "--tf32", "True", "--model_max_length", "2048",
        "--gradient_checkpointing", "True", "--dataloader_num_workers", "4",
        "--group_by_modality_length", "True", "--lazy_preprocess", "True",
        "--freeze_backbone", "True", "--seed", str(SEED), "--do_eval", "True",
        "--report_to", "none",
    ]
    if smoke:
        command += ["--max_steps", "3"]
    return command


def prepare(root):
    root = root_from(root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("refusing non-empty capacity-chain root: {}".format(root))
    root.mkdir(parents=True, exist_ok=True)
    for name in list(CONFIGS) + ["base", "hyperllava_task0"]:
        (root / name).mkdir()
    (root / "data").mkdir()
    (root / "summary").mkdir()
    train = read_json(TRAIN_FILE)
    test = read_json(TEST_FILE)
    if len(train) != 23998 or len(test) != 3000:
        raise RuntimeError("unexpected ImageNet-R counts: train={} test={}".format(len(train), len(test)))
    train_link = root / "data" / "train.json"
    test_link = root / "data" / "test.json"
    train_link.symlink_to(TRAIN_FILE)
    test_link.symlink_to(TEST_FILE)
    indices = random.Random(SEED).sample(range(len(train)), VALIDATION_SAMPLES)
    validation = [train[i] for i in indices]
    smoke_train = train[:8]
    smoke_val = validation[:8]
    write_json(root / "data" / "validation_audit.json", validation)
    write_json(root / "data" / "smoke_train.json", smoke_train)
    write_json(root / "data" / "smoke_val.json", smoke_val)
    write_json(root / "data" / "dataset_manifest.json", {
        "train_path": TRAIN_FILE, "train_samples": len(train),
        "validation_audit_path": str(root / "data" / "validation_audit.json"),
        "validation_audit_samples": len(validation),
        "validation_is_overlapping_audit": True,
        "test_path": TEST_FILE, "test_samples": len(test),
        "seed": SEED,
        "train_ids_sha256": hashlib.sha256("\n".join(str(x["id"]) for x in train).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(str(x["question_id"]) for x in test).encode()).hexdigest(),
    })

    adapter_config = read_json(Path(HYPER_CHECKPOINT) / "adapter_config.json")
    training_args = read_json(Path(HYPER_CHECKPOINT) / "training_args.json")
    trainer_state = read_json(Path(HYPER_CHECKPOINT) / "trainer_state.json")
    hyper_state = torch.load(Path(HYPER_CHECKPOINT) / "adapter_model.bin", map_location="cpu", weights_only=False)
    hyper_trainable_params = sum(int(value.numel()) for value in hyper_state.values())
    reference = {
        "schema_version": 1,
        "chain": ["P_base", "P_single_r8", "P_2xr8", "P_4xr8", "P_r16", "P_Hyper-LLaVA_task_LoRA"],
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "git_commit": git_commit(),
        "physical_gpus": list(GPUS), "global_seed": SEED,
        "base_model": BASE_MODEL, "vision_tower": VISION_TOWER,
        "projector": PROJECTOR, "image_preprocessing": "pad",
        "prompt_template": "v1 / vicuna_v1", "answer_format": "case-insensitive exact match",
        "max_length": 2048, "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128},
        "dataset": {
            "train": TRAIN_FILE, "train_samples": len(train),
            "validation_audit": str(root / "data" / "validation_audit.json"),
            "validation_audit_samples": len(validation),
            "validation_is_overlapping_audit": True,
            "test": TEST_FILE, "test_samples": len(test),
            "note": "Original Hyper command has evaluation_strategy=no and uses all 23,998 records; the fixed 256-row audit is recorded for val-loss only and does not remove samples from training.",
        },
        "hyperllava_checkpoint": HYPER_CHECKPOINT,
        "hyperllava_adapter_config": {key: adapter_config.get(key) for key in ["peft_type", "task_type", "r", "lora_alpha", "lora_dropout", "target_modules", "expert_num", "cur_task", "task_embedding_dim"]},
        "hyperllava_trainable_parameter_count": hyper_trainable_params,
        "hyperllava_training_observed": {
            "command": training_args.get("command"),
            "global_step": trainer_state.get("global_step"),
            "epoch": trainer_state.get("epoch"),
            "train_loss": trainer_state.get("log_history", [])[-1].get("train_loss"),
            "effective_global_batch": 64,
            "per_device_batch": 8, "gradient_accumulation": 2, "world_size": 4,
            "learning_rate": 2e-4, "optimizer": "adamw_torch/deepspeed ZeRO-2 (command uses deepspeed)",
            "weight_decay": 0.0, "scheduler": "cosine", "warmup_ratio": 0.03,
            "epochs": 1, "bf16": True, "fp16": False, "tf32": True,
            "vision_frozen": True, "projector_frozen": True, "backbone_frozen": True,
            "evaluation_strategy": "no", "seed": "not serialized in parsed_args; command audit required",
        },
        "capacity_chain_recipe": {
            "optimizer": "adamw_torch", "learning_rate": 2e-4, "weight_decay": 0.0,
            "mm_projector_lr": 2e-5,
            "scheduler": "cosine", "warmup_ratio": 0.03, "epochs": 1,
            "effective_global_batch": 64, "per_device_batch": 1, "gradient_accumulation": 64,
            "bf16": True, "fp16": False, "tf32": True, "dropout": 0.05,
            "target_modules": adapter_config.get("target_modules"),
            "alpha_over_rank": 2.0,
            "fixed_selection": True, "router": False, "clustering": False,
            "composition": {"two_r8": "(E1+E2)/sqrt(2)", "four_r8": "(E1+E2+E3+E4)/sqrt(4)"},
            "expert_seeds": {"E1": 4201, "E2": 4202, "E3": 4203, "E4": 4204},
        },
        "discrepancies": [
            "The original Hyper checkpoint uses rank 48, six task slots, per-device batch 8 on four GPUs, and no validation pass; the capacity chain uses independent fixed Compose LoRA experts with the same effective batch 64, target boundary, data, preprocessing, optimizer family, schedule, and one epoch.",
            "The chain uses microbatch 1 plus accumulation 64 on one physical GPU to preserve effective global batch 64 and fit 24 GiB cards.",
            "Alpha is scaled as 2*rank (r8=16, r16=32) so alpha/r remains the observed Hyper value 2; this is recorded as a rank-derived scaling invariant.",
        ],
    }
    write_json(root / "reference_hyperllava_config.json", reference)
    for name in CONFIGS:
        spec = dict(CONFIGS[name])
        spec["config_name"] = name
        spec["global_seed"] = SEED
        spec["train_samples"] = len(train)
        spec["effective_global_batch"] = 64
        spec["data_path"] = str(train_link)
        spec["validation_path"] = str(root / "data" / "validation_audit.json")
        spec["command"] = command_for(root, name)
        write_json(root / name / "config.json", spec)
        (root / name / "command.txt").write_text(" ".join(spec["command"]) + "\n", encoding="utf-8")
        (root / name / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    write_json(root / "base" / "config.json", {"name": "base", "checkpoint": BASE_POOL, "test_pipeline": "same Compose eval"})
    write_json(root / "hyperllava_task0" / "config.json", {"name": "hyperllava_task0", "checkpoint": HYPER_CHECKPOINT, "forced_task_id": 0, "test_pipeline": "same prompt/image/generation/scorer"})
    print("prepared {} with train={} val_audit={} test={} commit={}".format(root, len(train), len(validation), len(test), git_commit()), flush=True)


def train(args):
    root = root_from(args.root)
    name = args.config
    if name not in CONFIGS:
        raise ValueError("unknown capacity config: {}".format(name))
    gpu = int(args.gpu)
    if gpu not in GPUS:
        raise ValueError("only physical GPUs 4-7 are allowed")
    spec = CONFIGS[name]
    output = root / name / ("smoke_output" if args.smoke else "output")
    marker = root / name / ("SMOKE_DONE" if args.smoke else "DONE")
    failed = root / name / ("SMOKE_FAILED" if args.smoke else "FAILED")
    if marker.exists():
        print("skip {} (already marked {})".format(name, marker), flush=True)
        return
    command = command_for(root, name, smoke=args.smoke)
    (root / name / ("smoke_command.txt" if args.smoke else "command.txt")).write_text(" ".join(command) + "\n", encoding="utf-8")
    log = root / name / ("smoke.log" if args.smoke else "train.log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
    env["TOKENIZERS_PARALLELISM"] = "false"
    started = time.time()
    try:
        with log.open("a", encoding="utf-8") as handle:
            handle.write("START_UTC={} GPU={} COMMAND={}\n".format(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpu, " ".join(command)))
            handle.flush()
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
            peak_memory = 0
            while process.poll() is None:
                try:
                    query = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                        text=True, stderr=subprocess.DEVNULL,
                    )
                    peak_memory = max(peak_memory, int(query.strip().splitlines()[0]))
                except (OSError, ValueError, subprocess.CalledProcessError):
                    pass
                time.sleep(5)
            result_code = process.wait()
        if result_code != 0:
            raise RuntimeError("training exit {}".format(result_code))
        write_json(root / name / ("smoke_metrics.json" if args.smoke else "runtime.json"), {
            "gpu": gpu, "duration_seconds": time.time() - started,
            "peak_memory_mib": peak_memory,
            "examples_seen": 8 if args.smoke else 23998,
            "effective_global_batch": 64,
        })
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
        print("{} complete".format(name), flush=True)
    except BaseException as exc:
        failed.write_text("{}\n".format(exc), encoding="utf-8")
        print("{} FAILED: {}".format(name, exc), flush=True)
        raise


def assemble(args):
    root = root_from(args.root)
    for name, spec in CONFIGS.items():
        if not (root / name / "DONE").exists():
            print("skip assemble {} (training not DONE)".format(name), flush=True)
            continue
        pool = root / name / "pool"
        if (pool / "compose_experts.json").exists():
            continue
        previous = None
        for expert_id in spec["expert_ids"]:
            state = root / name / "output" / "expert_{:04d}.pt".format(expert_id)
            if not state.exists():
                raise FileNotFoundError(state)
            command = [PYTHON, "-m", "compose.eval.assemble_expert",
                "--model-path", BASE_MODEL, "--vision-tower", VISION_TOWER,
                "--projector-path", PROJECTOR, "--expert-state-dict", str(state),
                "--expert-id", str(expert_id), "--rank", str(spec["rank"]),
                "--alpha", str(spec["alpha"]), "--output-dir", str(pool)]
            if previous:
                command += ["--old-expert-checkpoint", str(previous)]
            subprocess.run(command, cwd=REPO, check=True)
            previous = pool
        (root / name / "ASSEMBLED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
    print("assembly complete", flush=True)


def rms(args):
    root = root_from(args.root)
    output = {"definition": "LoRA delta=B@A*(alpha/r); raw_sum=sum(delta); normalized_sum=raw_sum/sqrt(N)", "configs": {}}
    for name, spec in CONFIGS.items():
        if not (root / name / "DONE").exists():
            continue
        expert_deltas = {}
        for expert_id in spec["expert_ids"]:
            payload = torch.load(root / name / "output" / "expert_{:04d}.pt".format(expert_id), map_location="cpu", weights_only=False)
            state = payload["state_dict"]
            layers = {}
            for key, value in state.items():
                if key.endswith(".lora_B.weight"):
                    layer = key[:-len(".lora_B.weight")]
                    layers[layer] = value.float() @ state[layer + ".lora_A.weight"].float() * (float(spec["alpha"]) / float(spec["rank"]))
            expert_deltas[expert_id] = layers
        records = {}
        for layer in sorted(next(iter(expert_deltas.values()))):
            single = {str(e): float(expert_deltas[e][layer].square().mean().sqrt()) for e in expert_deltas}
            raw = sum(expert_deltas[e][layer] for e in expert_deltas)
            normalized = raw / math.sqrt(len(expert_deltas))
            records[layer] = {"expert_rms": single, "raw_sum_rms": float(raw.square().mean().sqrt()), "normalized_sum_rms": float(normalized.square().mean().sqrt())}
        output["configs"][name] = {"experts": len(expert_deltas), "representative_layers": {k: records[k] for k in sorted(records)[:8]}, "layers": records}
    write_json(root / "summary" / "rms_composition.json", output)
    print("RMS diagnostics complete", flush=True)


def _merge_and_score(root, name, chunks, annotation):
    from compose.experiments.task0_multi_r8_eval import _score, read_jsonl, write_jsonl
    rows = []
    for chunk in chunks:
        rows.extend(read_jsonl(chunk))
    answers = root / name / "evaluation" / "answers.jsonl"
    write_jsonl(answers, rows)
    metric = _score(Path(annotation), answers, root / name / "evaluation" / "score")
    write_json(root / name / "evaluation" / "metric.json", metric)
    return metric


def evaluate(args):
    root = root_from(args.root)
    annotation = TEST_FILE
    for name, spec in [("base", None)] + list(CONFIGS.items()):
        checkpoint = BASE_POOL if name == "base" else str(root / name / "pool")
        if not Path(checkpoint).is_dir():
            write_json(root / name / "evaluation.json", {"status": "SKIPPED", "reason": "checkpoint missing", "checkpoint": checkpoint})
            continue
        eval_dir = root / name / "evaluation"
        chunks = []
        processes = []
        for index, gpu in enumerate(GPUS):
            chunk = eval_dir / "chunks" / "{}_{}.jsonl".format(len(GPUS), index)
            chunks.append(chunk)
            command = [PYTHON, "-m", "compose.eval.eval_task", "--adapter-kind", "compose",
                "--model-path", BASE_MODEL, "--checkpoint-dir", checkpoint,
                "--projector-path", PROJECTOR, "--vision-tower", VISION_TOWER,
                "--question-file", TEST_FILE, "--image-folder", IMAGE_FOLDER,
                "--answers-file", str(chunk), "--run-summary-file", str(chunk.with_suffix(".summary.json")),
                "--num-chunks", str(len(GPUS)), "--chunk-idx", str(index),
                "--conv-mode", "vicuna_v1", "--max-new-tokens", "128", "--model-max-length", "2048"]
            if name == "base":
                command += ["--expert-ids", "", "--gates", ""]
            else:
                command += ["--expert-ids", ",".join(str(x) for x in spec["expert_ids"]),
                    "--gates", ",".join("1" for _ in spec["expert_ids"]), "--normalization", "none"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
            log = eval_dir / "chunk_{}.log".format(index)
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            processes.append((handle, subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        failures = []
        for handle, process in processes:
            code = process.wait(); handle.close()
            if code: failures.append(code)
        if failures:
            write_json(root / name / "evaluation.json", {"status": "FAILED", "failures": failures})
            print("evaluation failed {}".format(name), flush=True)
            continue
        metric = _merge_and_score(root, name, chunks, annotation)
        write_json(root / name / "evaluation.json", {"status": "DONE", "metric": metric})
        print("{} accuracy={}".format(name, metric.get("value")), flush=True)


def hyper_worker(args):
    import types
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from compose.eval.eval_task import _chunk, _prompt
    from compose.data.records import question_text

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        HYPER_CHECKPOINT, BASE_MODEL, "llava-lora", text_tower=VISION_TOWER,
        device=device, eval_modality_routing_mode="task")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    while not hasattr(base, "get_vision_tower") and hasattr(base, "model"):
        base = base.model
    if not hasattr(base, "compute_modality_route"):
        raise RuntimeError("cannot locate Hyper-LLaVA multimodal routing object")

    def force_task(self, image_guide_features, text_guide_features, valid_task_ids, routing_mode=None, use_hyperbolic=True):
        if 0 not in valid_task_ids:
            raise RuntimeError("Task0 is absent from Hyper checkpoint valid task ids")
        p = torch.zeros(image_guide_features.shape[0], len(valid_task_ids), device=image_guide_features.device, dtype=torch.float32)
        p[:, list(valid_task_ids).index(0)] = 1.0
        return {"p_v": p, "p_s": p, "p_d": p, "alpha": None, "prior_img_per_task": torch.ones(len(valid_task_ids), device=p.device)}

    base.compute_modality_route = types.MethodType(force_task, base)
    records = read_json(TEST_FILE)
    records = _chunk(records, args.num_chunks, args.chunk_idx)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = _prompt(record, model.config, "vicuna_v1")
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            with Image.open(os.path.join(IMAGE_FOLDER, str(record["image"]))) as image_handle:
                image = image_handle.convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                output_ids = model.generate(input_ids=ids, images=image_tensor, do_sample=False, num_beams=1, max_new_tokens=128, use_cache=True)
            text = tokenizer.batch_decode(output_ids[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            handle.write(json.dumps({"question_id": str(record.get("question_id", record.get("id"))), "prompt": question_text(record), "text": text}, ensure_ascii=False) + "\n")
    print("hyper worker complete {}".format(out), flush=True)


def hyper_eval(args):
    root = root_from(args.root)
    out_dir = root / "hyperllava_task0" / "evaluation"
    processes = []; chunks = []
    for index, gpu in enumerate(GPUS):
        output = out_dir / "chunks" / "{}_{}.jsonl".format(len(GPUS), index); chunks.append(output)
        log = out_dir / "chunk_{}.log".format(index); log.parent.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, "-m", "compose.experiments.task0_capacity_chain", "hyper-worker", "--output", str(output), "--num-chunks", str(len(GPUS)), "--chunk-idx", str(index)]
        handle = log.open("a", encoding="utf-8")
        processes.append((handle, subprocess.Popen(command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO)), stdout=handle, stderr=subprocess.STDOUT)))
    failures=[]
    for handle, process in processes:
        code=process.wait(); handle.close()
        if code: failures.append(code)
    if failures:
        write_json(root / "hyperllava_task0" / "evaluation.json", {"status": "FAILED", "failures": failures}); return
    metric = _merge_and_score(root, "hyperllava_task0", chunks, TEST_FILE)
    write_json(root / "hyperllava_task0" / "evaluation.json", {"status": "DONE", "metric": metric, "forced_task_id": 0})
    print("Hyper Task0 accuracy={}".format(metric.get("value")), flush=True)


def report(args):
    root = root_from(args.root)
    rows = []
    base_eval = read_json(root / "base" / "evaluation.json") if (root / "base" / "evaluation.json").exists() else {}
    hyper_eval_doc = read_json(root / "hyperllava_task0" / "evaluation.json") if (root / "hyperllava_task0" / "evaluation.json").exists() else {}
    base_score = base_eval.get("metric", {}).get("value")
    hyper_score = hyper_eval_doc.get("metric", {}).get("value")
    def train_stats(name, spec):
        state_path = root / name / "output" / "trainer_state.json"
        state = read_json(state_path) if state_path.exists() else {}
        history = state.get("log_history", [])
        eval_rows = [x for x in history if "eval_loss" in x]
        last = history[-1] if history else {}
        params = 0
        for eid in spec["expert_ids"]:
            payload_path = root / name / "output" / "expert_{:04d}.pt".format(eid)
            if payload_path.exists():
                payload = torch.load(payload_path, map_location="cpu", weights_only=False)
                params += sum(int(x.numel()) for x in payload["state_dict"].values())
        evaluation = read_json(root / name / "evaluation.json") if (root / name / "evaluation.json").exists() else {}
        reference = read_json(root / "reference_hyperllava_config.json")
        return {
            "config": name, "rank_structure": "{}x r{}".format(len(spec["expert_ids"]), spec["rank"]),
            "trainable_params": params, "train_loss": last.get("train_loss"),
            "val_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
            "task0_score": evaluation.get("metric", {}).get("value"),
            "gpu": spec["gpu"], "effective_global_batch": 64,
            "total_steps": state.get("global_step"), "samples_seen": 23998,
            "train_runtime_seconds": last.get("train_runtime"),
        }
    rows.append({"config": "Base", "rank_structure": "none", "trainable_params": 0, "train_loss": None, "val_loss": None, "task0_score": base_score, "effective_global_batch": None})
    for name, spec in CONFIGS.items(): rows.append(train_stats(name, spec))
    reference = read_json(root / "reference_hyperllava_config.json")
    rows.append({"config": "Hyper-LLaVA Task0 LoRA", "rank_structure": "6x r48 (Task0 direct)", "trainable_params": reference.get("hyperllava_trainable_parameter_count"), "train_loss": reference.get("hyperllava_training_observed", {}).get("train_loss"), "val_loss": None, "task0_score": hyper_score, "effective_global_batch": 64})
    by = {row["config"]: row.get("task0_score") for row in rows}
    base_value = by.get("Base"); hyper_value = by.get("Hyper-LLaVA Task0 LoRA")
    deltas = {}
    if base_value is not None:
        for key in ("Single-r8", "2xr8", "4xr8", "Single-r16"):
            canonical = {"Single-r8": "single_r8", "2xr8": "two_r8", "4xr8": "four_r8", "Single-r16": "single_r16"}[key]
            score = by.get(canonical)
            if score is not None: deltas[key + "_vs_base"] = score - base_value
    for a, b in [("2xr8", "Single-r8"), ("4xr8", "2xr8"), ("2xr8", "Single-r16"), ("4xr8", "Single-r16")]:
        aa = by.get({"2xr8": "two_r8", "4xr8": "four_r8", "Single-r8": "single_r8", "Single-r16": "single_r16"}[a])
        bb = by.get({"2xr8": "two_r8", "4xr8": "four_r8", "Single-r8": "single_r8", "Single-r16": "single_r16"}[b])
        if aa is not None and bb is not None: deltas[a + "_minus_" + b] = aa - bb
    if hyper_value is not None:
        for name in ("two_r8", "four_r8", "single_r16"):
            if by.get(name) is not None: deltas[name + "_gap_to_hyper"] = hyper_value - by[name]
    payload = {"chain": rows, "scores": by, "deltas": deltas, "hyper_checkpoint": HYPER_CHECKPOINT, "git_commit": git_commit()}
    write_json(root / "summary" / "performance_chain.json", payload)
    lines = ["# Task0 capacity chain", "", "| Config | Rank structure | Trainable Params | Train Loss | Val Loss | Task0 Score | Δ vs Base | Gap to Hyper-LLaVA |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        score = row.get("task0_score"); delta = None if score is None or base_value is None else score - base_value; gap = None if score is None or hyper_value is None else hyper_value - score
        lines.append("| {config} | {rank_structure} | {trainable_params} | {train_loss} | {val_loss} | {score} | {delta} | {gap} |".format(config=row["config"], rank_structure=row["rank_structure"], trainable_params=row.get("trainable_params", "-"), train_loss=row.get("train_loss", "-"), val_loss=row.get("val_loss", "-"), score=score if score is not None else "-", delta=delta if delta is not None else "-", gap=gap if gap is not None else "-"))
    lines += ["", "## Chain", "", "P_base → P_single_r8 → P_2×r8 → P_4×r8 → P_r16 → P_Hyper-LLaVA_task_LoRA", "", "The diagnosis is intentionally left as pending until all completed metrics are present. See `reference_hyperllava_config.json` for the recipe discrepancies.", ""]
    (root / "summary" / "performance_chain.md").write_text("\n".join(lines), encoding="utf-8")
    print("performance report written", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "assemble", "rms", "evaluate", "hyper-eval", "report"):
        p = sub.add_parser(command); p.add_argument("--root", required=True)
    p = sub.add_parser("train"); p.add_argument("--root", required=True); p.add_argument("--config", required=True); p.add_argument("--gpu", required=True, type=int); p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("hyper-worker"); p.add_argument("--output", required=True); p.add_argument("--num-chunks", required=True, type=int); p.add_argument("--chunk-idx", required=True, type=int)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.root)
    elif args.command == "train": train(args)
    elif args.command == "assemble": assemble(args)
    elif args.command == "rms": rms(args)
    elif args.command == "evaluate": evaluate(args)
    elif args.command == "hyper-eval": hyper_eval(args)
    elif args.command == "hyper-worker": hyper_worker(args)
    elif args.command == "report": report(args)


if __name__ == "__main__": main()
