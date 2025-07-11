from dataclasses import dataclass

from torch import nn
from torch.nn import modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm

from config.config_protocol import ConfigProtocol, MHAProjection
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner, RMSNormPruner

@dataclass
class Config(ConfigProtocol):
    MODEL = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B').to('cuda')
    TOKENIZER = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    DUMMY_INPUT = MODEL.dummy_inputs['input_ids'].to('cuda')
    IMPORTANCES_SAVE_PATH = './importances.csv'
    PRUNING_ITERATIONS = 1
    PRUNED_MODEL_SAVE_DIR = '/home/michal/hf-models/pruned'
    EVALUATE = True
    EVAL_RESULTS_PATH = './eval-results.json'
    DEP_GRAPH_ARGS = {
        'example_inputs': DUMMY_INPUT,
        'output_transform': lambda output: output.logits,
        'customized_pruners': {
            LlamaRMSNorm: RMSNormPruner(),
            nn.SiLU: OperationPruner(),
        }
    }
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
    MHA_PROJECTION_NAME_MAPPING = {
        MHAProjection.Q: 'q_proj',
        MHAProjection.K: 'k_proj',
        MHAProjection.V: 'v_proj',
        MHAProjection.O: 'o_proj'
    }
    BASE_OPERATION_TYPES = set([
        nn.SiLU,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.ReLU6,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        LlamaRMSNorm
    ])
