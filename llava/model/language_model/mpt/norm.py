"""作用：提供 MPT 语言模型兼容实现，包括配置、注意力、模块结构、初始化和 HuggingFace 适配逻辑。"""

import torch

def _cast_if_autocast_enabled(tensor):
    """作用：执行 _cast_if_autocast_enabled 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class LPLayerNorm(torch.nn.LayerNorm):

    """作用：LPLayerNorm 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x):
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

def rms_norm(x, weight=None, eps=1e-05):
    """作用：执行 rms_norm 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output

class RMSNorm(torch.nn.Module):

    """作用：RMSNorm 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)

class LPRMSNorm(RMSNorm):

    """作用：LPRMSNorm 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x):
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype)
NORM_CLASS_REGISTRY = {'layernorm': torch.nn.LayerNorm, 'low_precision_layernorm': LPLayerNorm, 'rmsnorm': RMSNorm, 'low_precision_rmsnorm': LPRMSNorm}