"""作用：构建和封装视觉编码器，把图像输入转换为 LLaVA 可使用的视觉特征。"""

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTextConfig


class CLIPVisionTower(nn.Module):
    """作用：CLIPVisionTower 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, vision_tower, args, delay_load=False):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        """作用：加载外部资源、模型权重、数据或配置，并转换为后续流程需要的结构。"""
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """作用：执行 feature_select 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            clip_image_features = image_forward_outs.image_embeds.to(images.dtype)

        return clip_image_features, image_features

    @property
    def dummy_feature(self):
        """作用：执行 dummy_feature 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """作用：执行 dtype 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return self.vision_tower.dtype

    @property
    def device(self):
        """作用：执行 device 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return self.vision_tower.device

    @property
    def config(self):
        """作用：执行 config 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """作用：执行 hidden_size 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return self.config.hidden_size

    @property
    def num_patches(self):
        """作用：执行 num_patches 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPTextTower(nn.Module):
    """作用：CLIPTextTower 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, text_tower, args, delay_load=False):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__()
        
        self.is_loaded = False

        self.text_tower_name = text_tower
        self.select_layer = args.mm_text_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_tower_name)

    def load_model(self):
        """作用：加载外部资源、模型权重、数据或配置，并转换为后续流程需要的结构。"""
        self.text_tower = CLIPTextModel.from_pretrained(self.text_tower_name)
        self.text_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, text_inputs, return_hidden_states=False):
        # if type(texts_input_ids) is list:
        #     text_features = []
        #     for text_input_ids in texts_input_ids:
        #         text_forward_out = self.text_tower(text_input_ids.to(self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
        #         text_feature = self.feature_select(text_forward_out).to(text_input_ids.dtype)
        #         text_features.append(text_feature)
        # else:
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        text_forward_outs = self.text_tower(**(text_inputs.to(self.device)))
        if return_hidden_states:
            text_hidden_features = text_forward_outs.last_hidden_state.to(self.dtype)
            text_features = text_forward_outs.pooler_output.to(self.dtype)

            return [text_hidden_features, text_features]
        else:
            text_features = text_forward_outs.pooler_output.to(self.dtype)
            return text_features

    @property
    def dummy_feature(self):
        """作用：执行 dummy_feature 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """作用：执行 dtype 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return self.text_tower.dtype

    @property
    def device(self):
        """作用：执行 device 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return self.text_tower.device

    @property
    def config(self):
        """作用：执行 config 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        if self.is_loaded:
            return self.text_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """作用：执行 hidden_size 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return self.config.hidden_size