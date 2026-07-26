"""作用：提供 MPT 语言模型兼容实现，包括配置、注意力、模块结构、初始化和 HuggingFace 适配逻辑。"""

from contextlib import contextmanager
import torch
import torch.nn as nn

@contextmanager
def init_empty_weights(include_buffers: bool=False):
    """Meta initialization context manager.

    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(torch.device('meta'), include_buffers=include_buffers) as f:
        yield f

@contextmanager
def init_on_device(device: torch.device, include_buffers: bool=False):
    """Device initialization context manager.

    A context manager under which models are initialized with all parameters
    on the specified device.

    Args:
        device (`torch.device`): Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        """作用：执行 register_empty_parameter 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer):
        """作用：执行 register_empty_buffer 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
    if include_buffers:
        tensor_constructors_to_patch = {torch_function_name: getattr(torch, torch_function_name) for torch_function_name in ['empty', 'zeros', 'ones', 'full']}
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):

        """作用：执行 patch_tensor_constructor 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
        def wrapper(*args, **kwargs):
            """作用：执行 wrapper 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
            kwargs['device'] = device
            return fn(*args, **kwargs)
        return wrapper
    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for (torch_function_name, old_torch_function) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)