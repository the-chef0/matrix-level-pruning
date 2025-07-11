import os

import torch

from config.config import Config as cfg
from config.config_protocol import ConfigProtocol
from infra.evaluator import evaluate_pruned
from infra.passes.identity_patching import IdentityPatcher
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.utils.model_utils import ModelUtils

model_utils = ModelUtils(cfg)
print("Model loaded")

assert cfg.PRUNING_ITERATIONS >= 0
for i in range(cfg.PRUNING_ITERATIONS):
    print(f"Iteration {i + 1}")
    # TODO: Make this more efficient
    # Every iteration after the first re-collects groups for the entire model,
    # but we know which modules changed in each pruning iteration.
    # It should suffice to just re-collect groups for the modules affected by pruning.

    print("(Re)building module - name mappings")
    model_utils.build_module_name_mappings()
    print("(Re)building dependency graph")
    model_utils.build_dependency_graph()

    importances_and_trees = collect_pruning_trees(
        cfg, model_utils,
        iteration=i,
    )

    _, tree_to_prune = importances_and_trees.pop(0)
    print(f"Pruning group {tree_to_prune}")
    tree_to_prune.prune()

model_utils.build_dependency_graph()
IdentityPatcher(model_utils).patch()

pruned_model_utils = model_utils

if cfg.PRUNED_MODEL_SAVE_DIR:
    pruned_model_utils.tokenizer.save_pretrained(cfg.PRUNED_MODEL_SAVE_DIR)
    torch.save(pruned_model_utils.model, os.path.join(cfg.PRUNED_MODEL_SAVE_DIR, "model.pth"))
    print(f"Saved pruned model to {cfg.PRUNED_MODEL_SAVE_DIR}")

if cfg.EVALUATE:
    assert cfg.PRUNED_MODEL_SAVE_DIR is not None
    assert cfg.EVAL_RESULTS_PATH is not None
    
    evaluate_pruned(
        model_utils=pruned_model_utils,
        pruned_model_save_dir=cfg.PRUNED_MODEL_SAVE_DIR,
        eval_result_path=cfg.EVAL_RESULTS_PATH
    )
