import os

import torch
import transformers

from llava import conversation as conversation_lib
from llava.train.llava_trainer import LLaVATraine

from compose.adapters import decoder_projection_names
from compose.model import ComposeLlavaForCausalLM, load_compose_config

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .data import make_supervised_data_module


class PeftBaselineTrainer(LLaVATrainer):
    """Trainer with the same LoRA-B finite-gradient audit as Compose."""

    def __init__(self, *args, **kwargs):
        self.trainable_lora_b_count = 0
        self.max_finite_gradient_lora_b_count = 0
        self._finite_gradient_lora_b_names = set()
        self._gradient_hook_handles = []
        super().__init__(*args, **kwargs)
        for name, parameter in self.model.named_parameters():
            if "lora_B" in name and parameter.requires_grad:
                self.trainable_lora_b_count += 1
                self._gradient_hook_handles.append(
                    parameter.register_hook(
                        lambda gradient, parameter_name=name: self._record_gradient(
                            parameter_name, gradient
                        )
                    )
                )

    def _record_gradient(self, name, gradient):
        if bool(torch.isfinite(gradient).all()):
            self._finite_gradient_lora_b_names.add(name)
            self.max_finite_gradient_lora_b_count = max(
                self.max_finite_gradient_lora_b_count,
                len(self._finite_gradient_lora_b_names),
            )
        return gradient

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        lora_b_parameters = [
            paramete
            for name, parameter in model.named_parameters()
            if "lora_B" in name and parameter.requires_grad
        ]
        finite_count = sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in lora_b_parameters
        )
        self.max_finite_gradient_lora_b_count = max(
            self.max_finite_gradient_lora_b_count, finite_count
        )
        return loss

    def _save(self, output_dir=None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_di
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


def train() -> None:
    from peft import LoraConfig, get_peft_model

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.vision_tower is None:
        raise ValueError("PEFT parity baseline requires --vision_tower")

    config = load_compose_config(
        model_args.model_name_or_path, cache_dir=training_args.cache_di
    )
    config.mm_vision_tower = model_args.vision_towe
    config.mm_vision_select_layer = model_args.mm_vision_select_laye
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.mm_projector_type = model_args.mm_projector_type
    model = ComposeLlavaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.config.use_cache = False
    model.get_model().initialize_vision_modules(model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    vision_tower.to(device=training_args.device, dtype=vision_dtype)

    model.requires_grad_(False)
    target_modules = decoder_projection_names(model)
    if len(target_modules) != 224:
        raise ValueError(
            "PEFT parity baseline expected 224 decoder projections, found {}".format(
                len(target_modules)
            )
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=model_args.compose_rank,
            lora_alpha=model_args.compose_alpha,
            lora_dropout=model_args.compose_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if (
        model_args.expected_adapter_parameters is not None
        and trainable_parameter_count != model_args.expected_adapter_parameters
    ):
        raise ValueError(
            "trainable parameter count mismatch; expected={}, actual={}".format(
                model_args.expected_adapter_parameters, trainable_parameter_count
            )
        )
    lora_layer_count = sum("lora_A.default.weight" in name for name, _ in model.named_parameters())
    if lora_layer_count != len(target_modules):
        raise ValueError(
            "PEFT injection count mismatch; targets={}, injected={}".format(
                len(target_modules), lora_layer_count
            )
        )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        model_args.version, conversation_lib.conv_templates["vicuna_v1"]
    )
    data_args.image_processor = vision_tower.image_processo
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.mm_projector_lr = training_args.mm_projector_l
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.base_model.model.initialize_vision_tokenizer(model_args, tokenizer)

    if training_args.local_rank in (-1, 0):
        print(
            "PEFT parity config: {}".format(
                {
                    "rank": model_args.compose_rank,
                    "alpha": model_args.compose_alpha,
                    "dropout": model_args.compose_dropout,
                    "target_count": len(target_modules),
                    "target_modules": target_modules,
                    "trainable_parameter_count": trainable_parameter_count,
                }
            )
        )

    data_module = make_supervised_data_module(tokenizer, data_args)
    trainer = PeftBaselineTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    checkpoints = (
        [name for name in os.listdir(training_args.output_dir) if name.startswith("checkpoint-")]
        if os.path.isdir(training_args.output_dir)
        else []
    )
    if checkpoints:
        raise ValueError(
            "output_dir contains checkpoint-* entries; parity runs require a fresh output: {}".format(
                sorted(checkpoints)
            )
        )
    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    if training_args.should_save:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    if training_args.local_rank in (-1, 0):
        supervision_summary = data_module["data_collator"].supervision_summary()
        if training_args.dataloader_num_workers:
            supervision_summary["scope"] = "main-process-only"
            supervision_summary["note"] = (
                "worker collators enforce zero-supervision errors but do not share counters; "
                "use compose.train.audit_supervision for dataset-wide statistics"
            )
        print("PEFT supervision summary: {}".format(supervision_summary))
        print("Trainable LoRA-B count: {}".format(trainer.trainable_lora_b_count))
        print(
            "Finite-gradient LoRA-B count: {}".format(
                trainer.max_finite_gradient_lora_b_count
            )
        )


if __name__ == "__main__":
    train()
