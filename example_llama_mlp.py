import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from infra.evaluator import evaluate_pruned
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.passes.identity_patching import IdentityPatcher
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner, RMSNormPruner
from infra.utils.model_utils import ModelUtils

class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.to('cuda')
        self.dummy_inputs = self.model.dummy_inputs

    def forward(self, input_ids):
        output = self.model(input_ids=input_ids, use_cache=False)
        return output.logits

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

importances_and_groups = collect_pruning_trees(
    model_utils,
    iteration=0,
    save_path=IMPORTANCES_SAVE_PATH
)

_, g = importances_and_groups[130]
print(g)
g.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_groups = collect_pruning_trees(
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
print(model_utils.model)

# binary_operation_patcher = IRBinOpPatcher(model_utils)
# binary_operation_patcher.patch()

# if EVALUATE:
#     assert PRUNED_MODEL_SAVE_DIR is not None
#     assert EVAL_RESULTS_PATH is not None
    
#     evaluate_pruned(
#         model_utils=model_utils,
#         pruned_model_save_dir=PRUNED_MODEL_SAVE_DIR,
#         eval_result_path=EVAL_RESULTS_PATH
#     )

