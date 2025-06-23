from torch import nn
from torch.nn import modules
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm

BASE_TRANSFORM_TYPES = set([
    modules.Linear,
    modules.conv._ConvNd
])

TRANSFORM_EXCLUSION_KEYWORDS = set([
    'lm_head',
    'fc',
    'classifier'
])

BASE_ATTENTION_TYPES = set([
    LlamaAttention,
])

ATTENTION_CHILD_KEYWORDS = set([
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj'
])

BASE_OPERATION_TYPES = set([
    nn.SiLU,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.ReLU6,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    LlamaRMSNorm
])
