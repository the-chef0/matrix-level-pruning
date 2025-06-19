import os

import torch

import config
from eval_pruned import evaluate_pruned # TODO: Make relative imports work
from importances_and_groups import collect_groups
from utils.binary_operation_patcher import BinaryOperationPatcher
from utils.model_utils import ModelUtils

model_utils = ModelUtils(
    model=config.MODEL,
    tokenizer=config.TOKENIZER,
    dep_graph_args=config.DEP_GRAPH_ARGS
)
print("Model loaded")

assert config.PRUNING_ITERATIONS >= 0
for i in range(config.PRUNING_ITERATIONS):
    print(f"Iteration {i + 1}")
    # TODO: Make this more efficient
    # Every iteration after the first re-collects groups for the entire model,
    # but we know which modules changed in each pruning iteration.
    # It should suffice to just re-collect groups for the modules affected by pruning.

    print("(Re)building module - name mappings")
    model_utils.build_module_name_mappings()
    print("(Re)building dependency graph")
    model_utils.build_dependency_graph()

    importances_and_groups = collect_groups(
        model_utils,
        iteration=i,
        save_path=config.IMPORTANCES_SAVE_PATH
    )

    _, group_to_prune = importances_and_groups.pop(0)
    print(f"Pruning group {group_to_prune}")
    group_to_prune.prune()

model_utils.build_dependency_graph()
BinaryOperationPatcher(model_utils).patch()

pruned_model_utils = model_utils

if config.PRUNED_MODEL_SAVE_DIR:
    pruned_model_utils.tokenizer.save_pretrained(config.PRUNED_MODEL_SAVE_DIR)
    torch.save(pruned_model_utils.model, os.path.join(config.PRUNED_MODEL_SAVE_DIR, "model.pth"))
    print(f"Saved pruned model to {config.PRUNED_MODEL_SAVE_DIR}")

if config.EVALUATE:
    assert config.PRUNED_MODEL_SAVE_DIR is not None
    assert config.EVAL_RESULTS_PATH is not None
    
    evaluate_pruned(
        model_utils=pruned_model_utils,
        pruned_model_save_dir=config.PRUNED_MODEL_SAVE_DIR,
        eval_result_path=config.EVAL_RESULTS_PATH
    )
