from dataclasses import dataclass

from torch import nn
from torch.nn import modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm

from config.config_protocol import ConfigProtocol, MHAProjection
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner, RMSNormPruner

@dataclass
class Config(ConfigProtocol):
    DEVICE = 'cuda'
    MODEL = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
    TOKENIZER = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    DUMMY_INPUT = MODEL.dummy_inputs['input_ids']
    IMPORTANCES_SAVE_PATH = './importances.csv'
    TARGET_SPARSITY = 0.1
    PRUNED_MODEL_SAVE_DIR = '/home/michal/hf-models/llama-pruned'
    EVALUATE = True
    EVAL_RESULTS_PATH = './eval-results.json'
    DEP_GRAPH_ARGS = {
        'output_transform': lambda output: output.logits,
        'customized_pruners': {
            LlamaRMSNorm: RMSNormPruner(),
            nn.SiLU: OperationPruner(),
        }
    }
    TRANSFORM_EXCLUSION_KEYWORDS = set([
        'lm_head',
        'fc',
        'classifier'
    ])
    BASE_ATTENTION_TYPES = set([
        LlamaAttention,
    ])
    MHA_PROJECTION_NAME_MAPPING = {
        MHAProjection.Q: 'q_proj',
        MHAProjection.K: 'k_proj',
        MHAProjection.V: 'v_proj',
        MHAProjection.O: 'o_proj'
    }
    BASE_TRANSFORM_TYPES = set([
        modules.Linear,
        modules.conv._ConvNd
    ])
    BASE_ACT_TYPES = set([
        nn.ELU,
        nn.Hardshrink,
        nn.Hardsigmoid,
        nn.Hardtanh,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.PReLU,
        nn.ReLU,
        nn.ReLU6,
        nn.RReLU,
        nn.SELU,
        nn.CELU,
        nn.GELU,
        nn.Sigmoid,
        nn.SiLU,
        nn.Mish,
        nn.Softplus,
        nn.Softshrink,
        nn.Softsign,
        nn.Tanh,
        nn.Tanhshrink,
        nn.Threshold,
        nn.GLU,
        nn.Softmin,
        nn.Softmax,
        nn.Softmax2d,
        nn.LogSoftmax,
        nn.AdaptiveLogSoftmaxWithLoss,
    ])
    BASE_NORM_TYPES = set([
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LazyBatchNorm1d,
        nn.LazyBatchNorm2d,
        nn.LazyBatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LazyInstanceNorm1d,
        nn.LazyInstanceNorm2d,
        nn.LazyInstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
        nn.RMSNorm,
        LlamaRMSNorm
    ])
