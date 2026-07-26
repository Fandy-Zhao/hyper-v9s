"""作用：构建多模态投影器，把视觉特征映射到语言模型隐藏空间。"""

import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    """作用：IdentityMap 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__()

    def forward(self, x, *args, **kwargs):
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        return x

    @property
    def config(self):
        """作用：执行 config 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    """作用：SimpleResBlock 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, channels):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    """作用：根据配置构建模型组件、数据结构或运行时对象。"""
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
