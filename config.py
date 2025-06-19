from torch import nn
from torch.nn import SiLU
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, SelectiveMultiply

from utils.customized_pruners import OperationPruner, RMSNormPruner

MODEL = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B').to('cuda')
TOKENIZER = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
IMPORTANCES_SAVE_PATH = './importances.csv'
PRUNING_ITERATIONS = 1
PRUNED_MODEL_SAVE_DIR = '/home/michal/hf-models/pruned'
EVALUATE = True
EVAL_RESULTS_PATH = './eval-results.json'
DEP_GRAPH_ARGS = {
    'example_inputs': MODEL.dummy_inputs['input_ids'].to('cuda'),
    'output_transform': lambda output: output.logits,
    'customized_pruners': {
        LlamaRMSNorm: RMSNormPruner(),
        nn.SiLU: OperationPruner(),
        SelectiveMultiply: OperationPruner()
    }
}
