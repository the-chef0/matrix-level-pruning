import os

import torch

from config.config import Config as cfg
from infra.evaluator import evaluate_pruned
from infra.passes.identity_patching import IdentityPatcher
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.pruning_tree_types.transform_pruning_tree import TransformPruningTree
from infra.utils.model_utils import ModelUtils

model_utils = ModelUtils(cfg)

importances_and_trees = collect_pruning_trees(
    cfg,
    model_utils,
    iteration=0,
)

for imp_and_tree in importances_and_trees:
    _, tree = imp_and_tree
    
    if isinstance(tree, TransformPruningTree):
        root_module = tree.get_root_module()
        if model_utils.module_to_name[root_module] == 'model.layers.6.mlp.up_proj':
            tree.prune()
            break

importances_and_trees = collect_pruning_trees(
    cfg,
    model_utils,
    iteration=1,
)

for imp_and_tree in importances_and_trees:
    _, tree = imp_and_tree
    
    if isinstance(tree, TransformPruningTree):
        root_module = tree.get_root_module()
        if model_utils.module_to_name[root_module] == 'model.layers.6.mlp.gate_proj':
            tree.prune()
            break

IdentityPatcher(cfg, model_utils).patch()
print(model_utils.model)

pruned_model_utils = model_utils

if cfg.PRUNED_MODEL_SAVE_DIR:
    pruned_model_utils.tokenizer.save_pretrained(cfg.PRUNED_MODEL_SAVE_DIR)
    torch.save(pruned_model_utils.model, os.path.join(cfg.PRUNED_MODEL_SAVE_DIR, "model.pth"))
    print(f"Saved pruned model to {cfg.PRUNED_MODEL_SAVE_DIR}")

if cfg.EVALUATE:
    assert cfg.PRUNED_MODEL_SAVE_DIR is not None
    assert cfg.EVAL_RESULTS_PATH is not None
    
    evaluate_pruned(
        cfg=cfg,
        model_utils=pruned_model_utils
    )
