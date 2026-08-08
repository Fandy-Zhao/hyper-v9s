#!/usr/bin/env python3
"""D2: nonlinear interaction localization.

D2.1 per-layer interaction norms:
  I_h^l = h_BC^l - h_B^l - h_C^l + h_0^l
  rho^l = ||I|| / (||h_B-h_0|| + ||h_C-h_0|| + eps)
Pooled hidden states are recorded at the LAST token and the mean over all
tokens (never full tensors). Final-logit interactions are computed for the
gold answer, the generated answer, the strongest wrong answer, the top-1
margin and the vocabulary-wide norm.

D2.2 module-level ablations (single seed):
  only-attention / only-MLP / only-qv / only-o / only-gate-up-down /
  B-full+C-attn / B-full+C-MLP / C-full+B-attn / C-full+B-MLP
Each reports official accuracy (greedy generation), marginal NLL and
conditional gains.

Outputs metrics/d2_layer_interaction.csv, metrics/d2_module_interaction.csv.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from compose.data.real_p1.official_metric import normalize, vqa_accuracy
from compose.eval.compose_p1_real import _collate_real, generate_rows, selection
from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import AdapterBridge
from compose.oracle.evaluator import _prepare_multimodal_batch


def layer_under(bridge, mask_kind):
    """Which bridge layers take the composition delta under a module mask."""
    if mask_kind == "all":
        return {name for name, _ in bridge.named_layers}
    if mask_kind == "attention":
        return {name for name, _ in bridge.named_layers if ".self_attn." in name}
    if mask_kind == "mlp":
        return {name for name, _ in bridge.named_layers if ".mlp." in name}
    if mask_kind == "qv":
        return {name for name, _ in bridge.named_layers if name.endswith(("q_proj", "v_proj"))}
    if mask_kind == "o":
        return {name for name, _ in bridge.named_layers if name.endswith("o_proj")}
    if mask_kind == "gate_up_down":
        return {name for name, _ in bridge.named_layers if name.endswith(("gate_proj", "up_proj", "down_proj"))}
    raise ValueError(mask_kind)


def pooled(hidden):
    """(last_token, mean_over_tokens) as detached float vectors."""
    last = hidden[0, -1].detach().float()
    mean = hidden[0].mean(dim=0).detach().float()
    return last, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-questions", required=True)
    parser.add_argument("--calibration-questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--checkpoint-seed", type=int, default=0)
    parser.add_argument("--analysis-seed", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--skip-layer", action="store_true")
    parser.add_argument("--skip-module", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower,
        projector_path=str(Path(args.model_path) / "mm_projector.bin"),
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    manager = bundle.expert_pool.manager
    ids = (1, 2)

    test = json.loads(Path(args.test_questions).read_text(encoding="utf-8"))
    if args.test_samples > 0:
        test = test[: args.test_samples]

    def module_kind(layer_name):
        if ".self_attn." in layer_name:
            if layer_name.endswith("q_proj") or layer_name.endswith("v_proj"):
                return "qv"
            return "attn_other"
        return "mlp"

    # ---------------- D2.1 layer interaction ----------------
    if not args.skip_layer:
        layer_rows = []
        for record in test:
            layer_rows.extend(_collect_layer_interaction(
                bundle, bridge, registry, ids, record, args))
        fields = sorted({key for row in layer_rows for key in row})
        with (output_root / "metrics" / "d2_layer_interaction.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(layer_rows)
        print("wrote d2_layer_interaction.csv ({} rows)".format(len(layer_rows)))

    # ---------------- D2.2 module ablations ----------------
    if not args.skip_module:
        module_rows = []
        configs = [
            ("all", "all"), ("attention_only", "attention"), ("mlp_only", "mlp"),
            ("qv_only", "qv"), ("o_only", "o"), ("gate_up_down_only", "gate_up_down"),
        ]
        # pair configurations with per-expert masks
        pair_configs = [
            ("B_full_C_attn", "all", "attention"),
            ("B_full_C_mlp", "all", "mlp"),
            ("C_full_B_attn", "attention", "all"),
            ("C_full_B_mlp", "mlp", "all"),
        ]
        for name, mask in configs:
            active = layer_under(bridge, mask)
            rows = _evaluate_masked(bundle, bridge, registry, ids, test, args, active, active, name)
            module_rows.extend(rows)
        for name, mask_b, mask_c in pair_configs:
            active_b = layer_under(bridge, mask_b)
            active_c = layer_under(bridge, mask_c)
            rows = _evaluate_masked(bundle, bridge, registry, ids, test, args, active_b, active_c, name)
            module_rows.extend(rows)
        with (output_root / "metrics" / "d2_module_interaction.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(module_rows[0].keys()))
            writer.writeheader()
            writer.writerows(module_rows)
        print("wrote d2_module_interaction.csv ({} rows)".format(len(module_rows)))


def _collect_layer_interaction(bundle, bridge, registry, ids, record, args):
    """Per-layer hidden collection for the four configurations."""
    from compose.eval.compose_p1_real import selection
    rows = []
    layer_names = [name for name, _ in bridge.named_layers]
    configs = ("base", "B", "C", "BC")
    hidden = {c: {} for c in configs}
    z = {}
    for config_name, mode in (("base", "base"), ("B", "single_left"),
                              ("C", "single_right"), ("BC", "c1")):
        states = []
        hooks = []
        for module in (module for _, module in bridge.named_layers):
            state = {}
            states.append(state)
            def hook(current_module, inputs, base_output, state=state):
                state["last"], state["mean"] = pooled(base_output)
            hooks.append(module.register_forward_hook(hook))
        with torch.inference_mode():
            with selection(bundle, registry, bridge, None, ids, mode):
                prepared = _prepare_multimodal_batch(
                    bundle, _collate_real(bundle, [record], args.images, args.device))
                logits = bundle.model(**prepared).logits
        for hook in hooks:
            hook.remove()
        hidden[config_name] = dict(zip(layer_names, states))
        z[config_name] = logits[0].float()
    logit_interaction = z["BC"] - z["B"] - z["C"] + z["base"]
    gold = bundle.tokenizer.encode(str(record["answer"]), add_special_tokens=False)[0]
    for layer_name in layer_names:
        h0, hB, hC, hBC = (hidden[c][layer_name] for c in configs)
        for tag, pick in (("last", lambda st: st["last"]), ("mean", lambda st: st["mean"])):
            i = pick(hBC) - pick(hB) - pick(hC) + pick(h0)
            denom = (pick(hB) - pick(h0)).norm() + (pick(hC) - pick(h0)).norm() + 1e-8
            rows.append({
                "sample_id": str(record["question_id"]),
                "layer_name": layer_name,
                "module": "attn" if ".self_attn." in layer_name else "mlp",
                "pool": tag,
                "interaction_norm": float(i.norm().item()),
                "rho": float((i.norm() / denom).item()),
                "b_delta_norm": float((pick(hB) - pick(h0)).norm().item()),
                "c_delta_norm": float((pick(hC) - pick(h0)).norm().item()),
            })
    last_pos = -1  # last token position predicts the first answer token
    rows.append({
        "sample_id": str(record["question_id"]),
        "layer_name": "logits",
        "module": "logits",
        "pool": "vocab",
        "interaction_norm": float(logit_interaction.norm().item()),
        "rho": 0.0,
        "b_delta_norm": 0.0,
        "c_delta_norm": 0.0,
        "gold_logit_interaction": float(logit_interaction[last_pos, gold].item()),
        "vocab_interaction_norm": float(logit_interaction.norm().item()),
        "top1_margin_interaction": float(
            (logit_interaction[last_pos, z["BC"][last_pos].argmax()]
             - logit_interaction[last_pos, z["BC"][last_pos].topk(2).indices[1]]).item()),
    })
    return rows


def _evaluate_masked(bundle, bridge, registry, ids, test, args, active_b, active_c, name):
    """Evaluate a masked composition: delta applied only on active layers."""
    from compose.eval.compose_p1_real import generate_rows
    rows = []
    manager = bundle.expert_pool.manager
    for record in test:
        def make_hook(active_set, expert_id, gate):
            def hook(current_module, inputs, base_output, active_set=active_set,
                     expert_id=expert_id, gate=gate):
                if not inputs:
                    return base_output
                output = base_output
                delta = bridge.compute_expert_delta(current_module, expert_id, inputs[0])
                output = output + delta.to(output.dtype) * gate
                return output
            return hook
        hooks = []
        for layer_name, module in bridge.named_layers:
            if layer_name in active_b:
                hooks.append(module.register_forward_hook(make_hook(active_b, 1, 1.0)))
            if layer_name in active_c:
                hooks.append(module.register_forward_hook(make_hook(active_c, 2, 1.0)))
        manager.clear_default_selection()
        try:
            prediction = generate_rows(bundle, [record], args.images, args.device, 32)[0]
        finally:
            for hook in hooks:
                hook.remove()
        answers = [str(a) for a in record["answers"]]
        score = vqa_accuracy(prediction, answers)
        rows.append({
            "sample_id": str(record["question_id"]),
            "configuration": name,
            "vqa_score": score,
            "accuracy": int(score >= 2.0 / 3.0),
            "prediction": prediction,
        })
    return rows


if __name__ == "__main__":
    main()
