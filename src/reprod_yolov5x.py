from datetime import datetime
from copy import deepcopy

import torch

from config.config_yolov5x import Config as cfg
from infra.passes.identity_patching import IdentityPatcher
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.utils.model_utils_yolo import ModelUtils


model_utils = ModelUtils(cfg)
print("Model loaded")
print(model_utils.model)

# Mapping from % parameter reduction to number of iterations
FIGURE_POINTS = {
    "0%": 0,
    "5%": 7,
    "11%": 10,
    "16%": 11,
    "20%": 13,
    "25%": 14,
    "29%": 15,
    "33%": 17,
    "38%": 18,
    "42%": 19,
    "46%": 20,
    "52%": 24,
}

for i in range(FIGURE_POINTS["0%"]):
    importances_and_trees = collect_pruning_trees(
        cfg, model_utils,
        iteration=i,
    )
    importances_and_trees[0][1].prune()


IdentityPatcher(cfg, model_utils).patch()
model_utils.model(cfg.DUMMY_INPUT)

# Extract the underlying DetectionModel from DetectMultiBackend for saving
# model_utils.model is DetectMultiBackend, model_utils.model.model is the DetectionModel
underlying_model = model_utils.model.model

# # Save in the format expected by train.py
ckpt = {
    "epoch": -1,
    "best_fitness": None,
    "model": deepcopy(underlying_model),
    "ema": None,
    "updates": None,
    "optimizer": None,
    "opt": None,
    "git": None,
    "date": datetime.now().isoformat(),
}

torch.save(ckpt, 'yolov5x-pruned.pt')
print("Pruned model saved to test.pt")
print(model_utils.model)
print(f"Sparsity: {model_utils.get_sparsity()}")