import os

import torch

from llava.train.llava_trainer import LLaVATrainer

from compose.experts.checkpoint import save_expert_checkpoint


class ComposeTrainer(LLaVATrainer):
    """LLaVA Trainer that stores adapter-only Compose checkpoints."""

    def __init__(self, *args, expert_pool=None, **kwargs) -> None:
        if expert_pool is None:
            raise ValueError("expert_pool is required")
        self.expert_pool = expert_pool
        self.trainable_lora_b_count = 0
        self.max_finite_gradient_lora_b_count = 0
        self._finite_gradient_lora_b_names = set()
        self._gradient_hook_handles = []
        super().__init__(*args, **kwargs)
        for name, parameter in self.model.named_parameters():
            if (
                ".experts." in name
                and name.endswith(".lora_B.weight")
                and parameter.requires_grad
            ):
                self.trainable_lora_b_count += 1
                self._gradient_hook_handles.append(
                    parameter.register_hook(
                        lambda gradient, parameter_name=name: self._record_lora_b_gradient(
                            parameter_name, gradient
                        )
                    )
                )

    def _record_lora_b_gradient(self, name, gradient):
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
            parameter
            for name, parameter in model.named_parameters()
            if ".experts." in name
            and name.endswith(".lora_B.weight")
            and parameter.requires_grad
        ]
        self.trainable_lora_b_count = max(
            self.trainable_lora_b_count, len(lora_b_parameters)
        )
        finite_count = sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in lora_b_parameters
        )
        self.max_finite_gradient_lora_b_count = max(
            self.max_finite_gradient_lora_b_count, finite_count
        )
        return loss

    def _save(self, output_dir=None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        if not self.args.should_save:
            return
        self.expert_pool.sync_training_step(self.state.global_step)
        os.makedirs(output_dir, exist_ok=True)
        self.model.config.save_pretrained(output_dir)
        save_expert_checkpoint(self.expert_pool, output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
