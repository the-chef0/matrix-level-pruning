import json

from config.config_llama_13b import Config as cfg
from infra.evaluator import evaluate_pruned
from infra.passes.identity_patching import IdentityPatcher
from infra.passes.pruning_tree_collection import collect_pruning_trees
from infra.utils.model_utils import ModelUtils

from infra.pruning_tree_types.transform_pruning_tree import TransformPruningTree
from infra.pruning_tree_types.attention_pruning_tree import AttentionPruningTreeGenerator

model_utils = None
if cfg.MODEL is not None:
    model_utils = ModelUtils(cfg)
    print("Model loaded")

with open('./angular-13b-coarse.json') as file:
    importances = json.load(file)

# Mapping from % parameter reduction to number of iterations
FIGURE_POINTS = {
    "0%": 0,
    "7%": 3,
    "12%": 5,
    "19%": 8,
    "27%": 11,
    "34%": 14,
    "41%": 17,
    "49%": 20,
    "55%": 23,
}

for i in range(FIGURE_POINTS["0%"]):
    layer_idx = int(importances[i][1])
    layer = model_utils.model.model.layers[layer_idx]

    tree = TransformPruningTree(cfg, model_utils, root_module=layer.mlp.up_proj)
    tree.prune()
    tree = TransformPruningTree(cfg, model_utils, root_module=layer.mlp.gate_proj)
    tree.prune()
    tree = TransformPruningTree(cfg, model_utils, root_module=layer.mlp.down_proj)
    tree.prune()

    attn_generator = AttentionPruningTreeGenerator(cfg, layer.self_attn)
    for attn_tree in attn_generator.get_trees(model_utils):
        attn_tree.prune()

IdentityPatcher(cfg, model_utils).patch()

print(model_utils.model(model_utils.model.dummy_inputs['input_ids'].to('cuda')))
print(model_utils.model)
print(f"SPARSITY {model_utils.get_sparsity()}")

pruned_model_utils = model_utils

if cfg.EVALUATE:
    assert cfg.PRUNED_MODEL_SAVE_DIR is not None
    assert cfg.EVAL_RESULTS_PATH is not None
    
    evaluate_pruned(
        cfg=cfg,
        model_utils=pruned_model_utils
    )
