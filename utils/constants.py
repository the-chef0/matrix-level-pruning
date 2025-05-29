from torch.nn import modules

BASE_TRANSFORM_TYPES = (
    modules.Linear,
    modules.conv._ConvNd
)

FORBIDDEN_TRANSFORM_KEYWORDS = (
    'self_attn',
    'lm_head'
)
