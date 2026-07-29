from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection

from .multimodal_arch import ComposeLlavaMetaForCausalLM, ComposeLlavaMetaModel


class ComposeLlavaConfig(LlamaConfig):
    model_type = "compose_llava"


class ComposeLlavaModel(ComposeLlavaMetaModel, LlamaModel):
    config_class = ComposeLlavaConfig

    def __init__(self, config: ComposeLlavaConfig) -> None:
        super().__init__(config)


class ComposeLlavaForCausalLM(LlamaForCausalLM, ComposeLlavaMetaForCausalLM):
    """LLaVA causal LM with no Hyper-LLaVA state or routing dependencies."""

    config_class = ComposeLlavaConfig

    def __init__(self, config: ComposeLlavaConfig) -> None:
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ComposeLlavaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self) -> ComposeLlavaModel:
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        compose_selection: Optional[ComposeSelection] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        context = use_selection(compose_selection) if compose_selection is not None else nullcontext()
        with context:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        selection = kwargs.pop("compose_selection", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if selection is not None:
            inputs["compose_selection"] = selection
        return inputs


AutoConfig.register(ComposeLlavaConfig.model_type, ComposeLlavaConfig)
AutoModelForCausalLM.register(ComposeLlavaConfig, ComposeLlavaForCausalLM)
