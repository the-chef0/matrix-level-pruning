import math
from torch import nn
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from typing import Type

from config.config_protocol import ConfigProtocol

def is_transform_type(cfg: ConfigProtocol, module_type: Type) -> bool:
    """Checks whether the given type belongs among the transform types, as defined in the config,

    Args:
        cfg (ConfigProtocol): See class docstring.
        module_type (Type): The class of the module.
    Returns:
        bool: True if the type belongs among the transform types, false otherwise.
    """
    is_transform = False
    for transform_type in cfg.BASE_TRANSFORM_TYPES:
        if issubclass(module_type, transform_type):
            is_transform = True
            break
    return is_transform

def is_attention_type(cfg: ConfigProtocol, module_type: Type) -> bool:
    """Checks whether the given type belongs among the attention types, as defined in the config,

    Args:
        cfg (ConfigProtocol): See class docstring.
        module_type (Type): The class of the module.
    Returns:
        bool: True if the type belongs among the attention types, false otherwise.
    """
    is_attention = False
    for transform_type in cfg.BASE_ATTENTION_TYPES:
        if issubclass(module_type, transform_type):
            is_attention = True
            break
    return is_attention

def get_num_conv_dims(conv_module: Module) -> int:
    """Given an assumed conv module, returns an integer indicating how many dimensions the
    module operates on.

    Args:
        conv_module (Module): The conv module.
    Returns:
        int: The number of dimensions.
    """
    if isinstance(conv_module, nn.Conv1d):
        return 1
    elif isinstance(conv_module, nn.Conv2d):
        return 2
    elif isinstance(conv_module, nn.Conv3d):
        return 3
    else:
        raise TypeError(f"Type {type(conv_module)} not supported.")

def get_conv_params_per_dim(conv_module: Module, num_dims: int) \
    -> tuple[tuple, tuple, tuple, tuple]:
    """Returns 4 tuples, each of length num_dims, containing the per-dimension padding, dilation,
    kernel size and stride quantites of the given conv module.

    Args:
        conv_module (Module): The conv module.
        num_dims (int): The number of dimensions in the conv module.
    Returns:
        tuple[tuple, tuple, tuple, tuple]: Tuples containing per-dimension padding, dilation, kernel
        size and stride quantities.
    """
    padding_per_dim = None
    if isinstance(conv_module.padding, int):
        padding_per_dim = (conv_module.padding,) * num_dims
    elif isinstance(conv_module.padding, tuple):
        padding_per_dim = conv_module.padding

    dilation_per_dim = None
    if isinstance(conv_module.dilation, int):
        dilation_per_dim = (conv_module.dilation,) * num_dims
    elif isinstance(conv_module.dilation, tuple):
        dilation_per_dim = conv_module.dilation

    kernel_size_per_dim = None
    if isinstance(conv_module.kernel_size, int):
        kernel_size_per_dim = (conv_module.kernel_size,) * num_dims
    elif isinstance(conv_module.dilation, tuple):
        kernel_size_per_dim = conv_module.kernel_size

    stride_per_dim = None
    if isinstance(conv_module.stride, int):
        stride_per_dim = (conv_module.stride,) * num_dims
    elif isinstance(conv_module.stride, tuple):
        stride_per_dim = conv_module.stride

    return padding_per_dim, dilation_per_dim, kernel_size_per_dim, stride_per_dim

def changes_feature_map_dims(num_dims: int, padding: tuple, dilation: tuple, kernel_size: tuple, \
    stride: tuple) -> bool:
    """Given the number of dimensions in a conv module and per-dimension padding, dilation,
    kernel size and stride values, determines using a dummy feature map size whether the module
    changes its feature map size.

    Args:
        num_dims (int): The number of dimensions in the conv module.
        padding (tuple): Per-dimension padding values.
        dilation (tuple): Per-dimension dilation values.
        kernel_size (tuple): Per-dimension kernel size values.
        stride (tuple): Per-dimension stride values.
    Returns:
        bool: True if the args were to lead to a feature map dimension change under the represented
            conv layer, false otherwise.
    """
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
    return any(kw in module_name for kw in cfg.MHA_PROJECTION_NAME_MAPPING.values())

def is_feature_map_transforming_conv(module: Module, module_name: str):
    if issubclass(type(module), _ConvNd):
        num_dims = get_num_conv_dims(module)
        padding, dilation, kernel, stride = get_conv_params_per_dim(module, num_dims)
        is_feature_map_transforming = changes_feature_map_dims(
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
