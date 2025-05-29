from torch.nn import modules, SiLU
from transformers.models.llama.modeling_llama import LlamaRMSNorm

BASE_TRANSFORM_TYPES = (
    modules.Linear,
    modules.conv._ConvNd
)

FORBIDDEN_TRANSFORM_KEYWORDS = (
    'self_attn',
    'lm_head'
)

BASE_OPERATION_TYPES = (
    SiLU,
    LlamaRMSNorm
)
