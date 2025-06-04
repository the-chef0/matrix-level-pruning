from torch import nn
from torch.nn import modules
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm

BASE_TRANSFORM_TYPES = (
    modules.Linear,
    modules.conv._ConvNd
)

TRANSFORM_EXCLUSION_KEYWORDS = (
    'lm_head',
    'fc',
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj'
)

BASE_ATTENTION_TYPES = (
    LlamaAttention,
)

BASE_OPERATION_TYPES = (
    nn.SiLU,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    LlamaRMSNorm
)
