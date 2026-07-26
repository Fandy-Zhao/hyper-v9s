#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..routing import InstanceModalityRouter


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.training = False
        self.cur_task = 0
        self.expert_num = 6

        # Initialize anchors
        self.image_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, 768)) for _ in range(10)]
        )

        self.text_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, 768)) for _ in range(10)]
        )

        self.image_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(10)]
            )
        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(10)]
            )
        # 高斯统计：mean、var、count（按 expert 维度存）
        hidden_dim = 768
        self.image_mean = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        self.image_var = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        # count 是标量，用 float（而不是 bfloat16 更稳）
        self.image_count = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.expert_num)
        ])

        # ---- text mean/var/count ----
        self.text_mean = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        self.text_var = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        self.text_count = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.expert_num)
        ])


        self.expert_weight = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.instance_router = None
        self.router_aux_loss = None
        self.router_replay_loss = None
        self.router_log_dict = {}
        self.initialize_instance_router()

    def initialize_instance_router(self):
        routing_mode = getattr(self.config, "modality_routing_mode", "task")
        if routing_mode == "sample" and self.instance_router is None:
            self.instance_router = InstanceModalityRouter(
                input_dim=8,
                hidden_dim=getattr(self.config, "router_hidden_dim", 32),
                dropout=getattr(self.config, "router_dropout", 0.0),
                residual_scale=getattr(self.config, "router_residual_scale", 1.0),
            )
        elif routing_mode != "sample":
            self.instance_router = None

    def set_cur_task(self, cur_task, expert_num):
        self.cur_task = cur_task
        self.expert_num = expert_num

        for name, param in self.image_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True

    def set_boundary_for_save(self):
        for name, param in self.image_boundary.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.image_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.image_mean.named_parameters():
            param.requires_grad = True

        for name, param in self.image_var.named_parameters():
            param.requires_grad = True

        for name, param in self.image_count.named_parameters():
            param.requires_grad = True

        for name, param in self.text_mean.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_var.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_count.named_parameters():
            param.requires_grad = True

        if self.instance_router is not None:
            for param in self.instance_router.parameters():
                param.requires_grad = True

    def get_model(self):
        return self.model

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        routing_mode = getattr(self.config, "modality_routing_mode", "task")
        if self.training and routing_mode == "sample":
            router_loss = getattr(self, "router_aux_loss", None)
            replay_loss = getattr(self, "router_replay_loss", None)
            if router_loss is not None or replay_loss is not None:
                if return_dict is False:
                    loss = outputs[0]
                    rest = outputs[1:]
                    if router_loss is not None:
                        loss = loss + getattr(self.config, "router_loss_weight", 0.1) * router_loss
                    if replay_loss is not None:
                        loss = loss + getattr(self.config, "router_replay_weight", 0.0) * replay_loss
                    return (loss,) + rest
                loss = outputs.loss
                if router_loss is not None:
                    loss = loss + getattr(self.config, "router_loss_weight", 0.1) * router_loss
                if replay_loss is not None:
                    loss = loss + getattr(self.config, "router_replay_weight", 0.0) * replay_loss
                outputs.loss = loss
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
