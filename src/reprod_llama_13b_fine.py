import json

from config.config_llama_7b import Config as cfg
from infra.evaluator import evaluate_pruned
from infra.passes.identity_patching import IdentityPatcher
from infra.utils.model_utils import ModelUtils

from infra.pruning_tree_types.transform_pruning_tree import TransformPruningTree
from infra.pruning_tree_types.attention_pruning_tree import AttentionPruningTreeGenerator

model_utils = None
if cfg.MODEL is not None:
    model_utils = ModelUtils(cfg)
    print("Model loaded")

with open('./angular-13b-fine.json') as file:
    importances = json.load(file)

mlp_importances = []
attn_importances = []

for imp, name in sorted(importances):
    if 'attn' in name:
        attn_importances.append((imp, name))
    else:
        mlp_importances.append((imp, name))

# Mapping from % parameter reduction to number of iterations
FIGURE_POINTS = {
    "0%": 0,
    "7%": 6,
    "12%": 15,
    "19%": 25,
    "27%": 37,
    "34%": 50,
    "41%": 63,
    "49%": 74,
    "55%": 84,
}

mlp_idx = 0
attn_idx = 0
for iter in range(FIGURE_POINTS["0%"]):
    if iter % 3 == 2:
        _, name = attn_importances[attn_idx]
        name_parts = name.split('.')
        layer_idx = int(name_parts[0])
        attn_module = model_utils.model.model.layers[layer_idx].self_attn
        attn_generator = AttentionPruningTreeGenerator(cfg, attn_module)
        for attn_tree in attn_generator.get_trees(model_utils):
            attn_tree.prune()
        attn_idx += 1
    else:
        _, name = mlp_importances[mlp_idx]
        name_parts = name.split('.')
        layer_idx = int(name_parts[0])
        proj_type = name_parts[1]
        mlp_module = model_utils.model.model.layers[layer_idx].mlp
        mlp_proj = getattr(mlp_module, proj_type)
        tree = TransformPruningTree(cfg, model_utils, root_module=mlp_proj)
        tree.prune()
        mlp_idx += 1

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
