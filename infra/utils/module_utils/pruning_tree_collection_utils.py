import math
from torch import nn
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from typing import Type

from config.config_protocol import ConfigProtocol
from infra.utils.model_utils import ModelUtils

def is_transform_type(cfg: ConfigProtocol, module_type: Type):
    is_transform = False
    for transform_type in cfg.BASE_TRANSFORM_TYPES:
        if issubclass(module_type, transform_type):
            is_transform = True
            break
    return is_transform

def is_attention_type(cfg: ConfigProtocol, module_type: Type):
    is_attention = False
    for transform_type in cfg.BASE_ATTENTION_TYPES:
        if issubclass(module_type, transform_type):
            is_attention = True
            break
    return is_attention

def get_num_conv_dims(module: Module):
    if isinstance(module, nn.Conv1d):
        return 1
    elif isinstance(module, nn.Conv2d):
        return 2
    elif isinstance(module, nn.Conv3d):
        return 3
    else:
        raise TypeError(f"Type {type(module)} not supported.")

def get_conv_params_per_dim(module, num_dims):
    padding_per_dim = None
    if isinstance(module.padding, int):
        padding_per_dim = (module.padding,) * num_dims
    elif isinstance(module.padding, tuple):
        padding_per_dim = module.padding

    dilation_per_dim = None
    if isinstance(module.dilation, int):
        dilation_per_dim = (module.dilation,) * num_dims
    elif isinstance(module.dilation, tuple):
        dilation_per_dim = module.dilation

    kernel_size_per_dim = None
    if isinstance(module.kernel_size, int):
        kernel_size_per_dim = (module.kernel_size,) * num_dims
    elif isinstance(module.dilation, tuple):
        kernel_size_per_dim = module.kernel_size

    stride_per_dim = None
    if isinstance(module.stride, int):
        stride_per_dim = (module.stride,) * num_dims
    elif isinstance(module.stride, tuple):
        stride_per_dim = module.stride

    return padding_per_dim, dilation_per_dim, kernel_size_per_dim, stride_per_dim

def changes_feature_map_dims(num_dims, padding, dilation, kernel_size, stride): # TODO: type hints
    DUMMY_INPUT_SIZE = 64
    assert num_dims == len(padding)
    assert len(padding) == len(dilation)
    assert len(dilation) == len(kernel_size)
    assert len(kernel_size) == len(stride)

    preserved = True
    for dim_idx in range(num_dims):
        numerator = DUMMY_INPUT_SIZE + (2 * padding[dim_idx]) - dilation[dim_idx] \
            * (kernel_size[dim_idx] - 1) - 1
        output_size = math.floor((numerator / stride[dim_idx]) + 1)
        current_preserved = (DUMMY_INPUT_SIZE == output_size)
        if not current_preserved:
            preserved = False
            break

    return not preserved

def contains_excluded_keyword(cfg: ConfigProtocol, module_name: str):
    contains_excluded = any(kw in module_name for kw in cfg.TRANSFORM_EXCLUSION_KEYWORDS)
    if contains_excluded:
        print(f"Excluding {module_name} - contains excluded keyword")
    
    return contains_excluded

def is_attention_child(cfg: ConfigProtocol, module_name: str):
    return any(kw in module_name for kw in cfg.ATTENTION_CHILD_KEYWORDS)

def is_feature_map_transforming_conv(module: Module, module_name: str):
    if issubclass(type(module), _ConvNd):
        num_dims = get_num_conv_dims(module)
        padding, dilation, kernel, stride = get_conv_params_per_dim(module, num_dims)
        is_feature_map_transforming = changes_feature_map_dims(
            module=module,
            num_dims=num_dims,
            padding=padding,
            dilation=dilation,
            kernel_size=kernel,
            stride=stride
        )

        if is_feature_map_transforming:
            print(f"Excluding {module_name} - does not preserve feature map dimensions")

        return is_feature_map_transforming
    else:
        return False

def meets_exclusion_criteria(cfg: ConfigProtocol, module: Module, module_name: str):
    # TODO: maybe write an abstraction for this that defines an interface for these
    # exclusion checking methods and allows me to "register" them with a checker
    # that does what this function does
    exclusion_criteria = []
    exclusion_criteria.append(contains_excluded_keyword(cfg, module_name))
    exclusion_criteria.append(is_attention_child(cfg, module_name))
    exclusion_criteria.append(is_feature_map_transforming_conv(module, module_name))
    return any(crit for crit in exclusion_criteria)
