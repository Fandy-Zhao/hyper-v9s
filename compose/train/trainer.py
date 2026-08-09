import os
from contextlib import nullcontext

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
        # Gradient-checkpoint recomputation runs inside loss.backward(), but
        # the selection context (a ContextVar) exits when the model forward
        # returns.  Keep the per-batch selection active across the whole
        # step so recomputation applies the same LoRA routing and the saved
        # tensor count matches (non-reentrant checkpoint contract).
        from compose.adapters.runtime import use_selection
        from compose.adapters.types import ComposeSelection

        selection = None
        raw = inputs.get("compose_selections")
        if raw is not None:
            # Per-sample (ids, gates) tuples from ComposeSelectionCollator;
            # every row is already padded to MAX_ACTIVE_EXPERTS slots.
            expert_ids = torch.tensor(
                [list(ids) for ids, _ in raw], dtype=torch.long
            )
            gates = torch.tensor(
                [list(gates) for _, gates in raw], dtype=torch.float32
            )
            selection = ComposeSelection(expert_ids, gates)
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if distributed:
            # 4-GPU (torchrun): gradient-exact DDP step (spec §4/§5).
            loss = self._ddp_training_step(model, inputs, selection)
        elif selection is not None:
            with use_selection(selection):
                loss = super().training_step(model, inputs)
        else:
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

    def _ddp_training_step(self, model, inputs, selection):
        """4-GPU (torchrun) training step — gradient-exact DDP (spec §4/§5).

        Verbatim body of the pinned transformers 4.33.3
        ``Trainer.training_step`` (LLaVATrainer does not override it), with
        two additions:

        1. The per-sample selection ContextVar stays active across the
           whole step, so activation-checkpoint recomputation during
           ``backward()`` sees the same LoRA routing (same contract as the
           single-GPU path; see ``_prepare_cluster_expert_backward``).
        2. World-size loss scaling: 4.33 backpropagates the undivided
           micro-batch loss and accumulates gradients by SUM over
           ``gradient_accumulation_steps``, while DDP all-reduces
           (averages) each micro-step gradient over the ranks. With the
           global batch held constant (``per_device x grad_accum x
           world_size``), the accumulated DDP gradient would be
           1/world_size of the single-GPU gradient for the same samples.
           Scaling the loss by ``world_size`` makes the accumulated
           gradient EXACTLY equal to the single-GPU protocol's — a
           gradient-equivalence compensation, not a learning-rate change.
           The returned (logging) loss is unscaled.
        """
        from compose.adapters.runtime import use_selection

        world_size = int(self.args.world_size)
        model.train()
        inputs = self._prepare_inputs(inputs)
        context = use_selection(selection) if selection is not None else nullcontext()
        with context:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.do_grad_scaling:
                self.scaler.scale(loss * world_size).backward()
            elif self.use_apex:
                from apex import amp

                with amp.scale_loss(loss * world_size, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss * world_size)
        return loss.detach() / self.args.gradient_accumulation_steps

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
