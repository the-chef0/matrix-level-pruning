from torch import nn
from torch.nn import modules
from transformers.models.llama.modeling_llama import LlamaRMSNorm

BASE_TRANSFORM_TYPES = (
    modules.Linear,
    modules.conv._ConvNd
)

CRITICAL_LAYER_KEYWORDS = (
    'self_attn',
    'lm_head',
    'fc'
)

BASE_OPERATION_TYPES = (
    nn.SiLU,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    LlamaRMSNorm
)
