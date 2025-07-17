from itertools import count
import os

import torch

from config.config import Config as cfg
from infra.evaluator import evaluate_pruned
from infra.passes.identity_patching import IdentityPatcher
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.utils.model_utils import ModelUtils

model_utils = ModelUtils(cfg)
print("Model loaded")

curr_sparsity = 0
iter = count()
assert cfg.TARGET_SPARSITY > 0 and cfg.TARGET_SPARSITY < 1
while curr_sparsity < cfg.TARGET_SPARSITY:
    i = next(iter)
    print(f"Iteration {i + 1}")
    # TODO: Make this more efficient
    # Every iteration after the first re-collects groups for the entire model,
    # but we know which modules changed in each pruning iteration.
    # It should suffice to just re-collect groups for the modules affected by pruning.

    importances_and_trees = collect_pruning_trees(
        cfg, model_utils,
        iteration=i,
    )

    _, tree_to_prune = importances_and_trees.pop(0)
    tree_to_prune.prune()
    curr_sparsity = model_utils.get_sparsity()
    print(f"Current sparsity: {curr_sparsity}, target sparsity {cfg.TARGET_SPARSITY}")

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
