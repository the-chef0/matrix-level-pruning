from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, SelectiveMultiply

from importances_and_groups import collect_groups
from utils.binary_operation_patcher import BinaryOperationPatcher
from utils.customized_pruners import OperationPruner, RMSNormPruner
from utils.model_utils import ModelUtils

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

model_utils = ModelUtils(
    model=MODEL,
    tokenizer=TOKENIZER,
    dep_graph_args=DEP_GRAPH_ARGS
)
print("Model loaded")

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_groups = collect_groups(
    model_utils,
    iteration=0,
    save_path=IMPORTANCES_SAVE_PATH
)

_, g = importances_and_groups[130]
print(g)
g.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_groups = collect_groups(
    model_utils,
    iteration=1,
    save_path=IMPORTANCES_SAVE_PATH
)

_, g = importances_and_groups[129]
print(g)
g.prune()

model_utils.build_dependency_graph()
BinaryOperationPatcher(model_utils).patch()
