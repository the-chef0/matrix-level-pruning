import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from config.config import Config as cfg
from infra.evaluator import evaluate_pruned
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.passes.identity_patching import IdentityPatcher
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner, RMSNormPruner
from infra.utils.model_utils import ModelUtils

model_utils = ModelUtils(cfg)
print("Model loaded")

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_trees = collect_pruning_trees(
    cfg,
    model_utils,
    iteration=0,
)

_, tree_to_prune = importances_and_trees[130]
print(tree_to_prune)
tree_to_prune.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_trees = collect_pruning_trees(
    cfg,
    model_utils,
    iteration=1,
)

_, tree_to_prune = importances_and_trees[129]
print(tree_to_prune)
tree_to_prune.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

IdentityPatcher(model_utils).patch()
print(model_utils.model)

if cfg.EVALUATE:
    assert cfg.PRUNED_MODEL_SAVE_DIR is not None
    assert cfg.EVAL_RESULTS_PATH is not None
    
    evaluate_pruned(
        model_utils=model_utils,
        pruned_model_save_dir=cfg.PRUNED_MODEL_SAVE_DIR,
        eval_result_path=cfg.EVAL_RESULTS_PATH
    )
