import os

import torch

from config.config import Config as cfg
from infra.evaluator import evaluate_pruned
from infra.passes.identity_patching import IdentityPatcher
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.utils.model_utils import ModelUtils

model_utils = ModelUtils(cfg)

importances_and_trees = collect_pruning_trees(
    cfg,
    model_utils,
    iteration=0,
)

_, tree_to_prune = importances_and_trees[130]
print(tree_to_prune)
tree_to_prune.prune()

importances_and_trees = collect_pruning_trees(
    cfg,
    model_utils,
    iteration=1,
)

_, tree_to_prune = importances_and_trees[129]
print(tree_to_prune)
tree_to_prune.prune()

IdentityPatcher(cfg, model_utils).patch()

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
