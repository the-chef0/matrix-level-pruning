import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from eval_pruned import evaluate_pruned
from importances_and_groups import collect_groups
from utils.identity_patcher import IdentityPatcher, IdentityWithGrad
from utils.customized_pruners import OperationPruner, RMSNormPruner
from utils.model_utils import ModelUtils

MODEL = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B').to('cuda')
TOKENIZER = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
DUMMY_INPUT = MODEL.dummy_inputs['input_ids'].to('cuda')
IMPORTANCES_SAVE_PATH = './importances.csv'
PRUNING_ITERATIONS = 1
PRUNED_MODEL_SAVE_DIR = '/home/michal/hf-models/pruned'
EVALUATE = False
EVAL_RESULTS_PATH = './eval-results.json'
DEP_GRAPH_ARGS = {
    'example_inputs': DUMMY_INPUT,
    'output_transform': lambda output: output.logits,
    'customized_pruners': {
        LlamaRMSNorm: RMSNormPruner(),
        nn.SiLU: OperationPruner(),
        IdentityWithGrad: OperationPruner()
    }
}

model_utils = ModelUtils(
    model=MODEL,
    dummy_input=DUMMY_INPUT,
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

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

IdentityPatcher(model_utils).patch()
