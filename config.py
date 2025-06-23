from torch import nn
from torch.nn import SiLU
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, SelectiveMultiply

from utils.customized_pruners import OperationPruner, RMSNormPruner

class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.to('cuda')
        self.dummy_inputs = self.model.dummy_inputs

    def forward(self, input_ids):
        output = self.model(input_ids=input_ids, use_cache=False)
        return output.logits

MODEL = LogitsOnlyWrapper(
    AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
)
TOKENIZER = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
DUMMY_INPUT = MODEL.dummy_inputs['input_ids'].to('cuda')
IMPORTANCES_SAVE_PATH = './importances.csv'
PRUNING_ITERATIONS = 1
PRUNED_MODEL_SAVE_DIR = '/home/michal/hf-models/pruned'
EVALUATE = True
EVAL_RESULTS_PATH = './eval-results.json'
DEP_GRAPH_ARGS = {
    'example_inputs': DUMMY_INPUT,
    #'output_transform': lambda output: output.logits,
    'customized_pruners': {
        LlamaRMSNorm: RMSNormPruner(),
        nn.SiLU: OperationPruner(),
        SelectiveMultiply: OperationPruner()
    }
}
