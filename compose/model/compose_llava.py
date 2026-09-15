from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection

from .multimodal_arch import ComposeLlavaMetaForCausalLM, ComposeLlavaMetaModel


class ComposeLlavaConfig(LlamaConfig):
    model_type = "compose_llava"


_CORE_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
)


def load_compose_config(model_name_or_path: str, cache_dir: str = None) -> ComposeLlavaConfig:
    """Convert a raw LLaVA/LLaMA config without invoking model-type dispatch."""

    source, _ = PretrainedConfig.get_config_dict(
        model_name_or_path, cache_dir=cache_dir
    )
    source_type = source.pop("model_type", None)
    if source_type not in ("llava", "llama", "compose_llava"):
        raise ValueError(
            "expected a LLaVA/LLaMA config, found model_type={!r}".format(source_type)
        )
    source.pop("architectures", None)
    config = ComposeLlavaConfig(**source)
    config.architectures = ["ComposeLlavaForCausalLM"]
    mismatches = {
        field: (source[field], getattr(config, field))
        for field in _CORE_CONFIG_FIELDS
        if field in source and source[field] != getattr(config, field)
    }
    if mismatches:
        raise ValueError("Compose config conversion changed core dimensions: {}".format(mismatches))
    return config


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
        # The V7 trainer asks for a per-sample answer loss on every step.  Until
        # this flag was read, ``**kwargs`` swallowed it and the class had no
        # per-sample path at all -- it inherits LlamaForCausalLM, not
        # LlavaLlamaForCausalLM, so the branch there never ran for a compose run.
        # The answer term stayed a token-weighted batch mean while the key term
        # was a batch sum, which made the effective loss weight track the
        # micro-batch width.  See ``sum_of_per_sample_token_means``.
        v7_sum_per_sample_loss = bool(kwargs.pop("v7_sum_per_sample_loss", False))
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
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                # See LlavaLlamaForCausalLM: one CE primitive produces both
                # the scalar answer loss and the transient per-sample NLL.
                labels=None if v7_sum_per_sample_loss else labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if v7_sum_per_sample_loss and labels is not None:
            if return_dict is False:
                raise ValueError("V7 per-sample loss requires return_dict output")
            # Imported here rather than at module scope: the compose package is
            # deliberately importable without pulling in the llava language-model
            # module, and `tests/compose/test_imports.py` enforces that.  The
            # helper is still the *same* helper -- sharing it is the point, since
            # ComposeLlavaForCausalLM is a sibling of LlavaLlamaForCausalLM and
            # not a subclass of it.
            from llava.model.language_model.llava_llama import (
                per_sample_token_mean_nll,
            )

            nll = per_sample_token_mean_nll(outputs.logits, labels)
            outputs.v7_per_sample_answer_nll = nll
            # Item assignment, not ``outputs.loss = ...``.  ``ModelOutput``
            # forwards an attribute write into the underlying mapping only when
            # the field is *already* present.  The parent ran with
            # ``labels=None``, so it dropped ``loss`` as an empty field and an
            # attribute write would leave the mapping without a "loss" key --
            # the step then dies in the HF trainer with "The model did not
            # return a loss from the inputs, only the following keys: logits",
            # naming every input but never the missing field.
            outputs["loss"] = nll.sum()
        return outputs

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
