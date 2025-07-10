"""
Defines a single pruning tree collection pass.
The pruning tree collection pass enumerates all pruning trees, ranks them by importance,
optionally saves a CSV file (location set in config) containing a list of
tree-importance tuples, and also returns the list of sorted tree-importance tuples.
"""

import csv
import os

from torch.nn import Module

from config.config_protocol import ConfigProtocol
from infra.utils.model_utils import ModelUtils
from infra.utils.module_utils.pruning_tree_collection_utils import is_attention_type, is_transform_type, meets_exclusion_criteria
from infra.pruning_tree_types.attention_pruning_tree import AttentionPruningTreeGenerator
from infra.pruning_tree_types.pruning_tree import PruningTree
from infra.pruning_tree_types.transform_pruning_tree import TransformPruningTree

def is_transform_tree_root(cfg: ConfigProtocol, model_utils: ModelUtils, module: Module) -> bool: # TODO: move to module utils?
    """Checks whether the given module can be treated as the root of a
    transform pruning tree.

    Args:
        model_utils (ModelUtils): See class docstring.
        module (Module): The PyTorch module in question.
    Returns:
        bool: True if module can be treated as the root of a transform
            pruning tree, False otherwise.
    """
    module_name = model_utils.module_to_name[module]
    is_transform = is_transform_type(cfg, type(module))
    is_not_excluded = not meets_exclusion_criteria(cfg, module, module_name)
    return is_transform and is_not_excluded

def is_attention_tree_root(cfg: ConfigProtocol, module: Module) -> bool:
    """Checks whether the given module can be treated as the root of an
    attention pruning tree.

    Args:
        model_utils (ModelUtils): See class docstring.
        module (Module): The PyTorch module in question.
    Returns:
        bool: True if module can be treated as the root of an attention
        pruning tree, False otherwise.
    """
    return is_attention_type(cfg, type(module))

def collect_pruning_trees(cfg: ConfigProtocol, model_utils: ModelUtils, iteration: int) -> list[tuple[float, PruningTree]]:
    """Enumerates and ranks all pruning trees in the model.

    Args:
        model_utils (ModelUtils): See class docstring.
        iteration (int): Used to distinguish between saved importance files if multiple
            iterations are done.
    Returns:
        list[tuple[float, PruningTree]]: Pruning trees sorted by ascending importance.
    Side-effects:
        Creates a CSV file containing the data from the return object.
    """
    trees = []
    trees_as_str = []
    tree_importances = []
    
    for module in model_utils.model.modules():
        if is_transform_tree_root(cfg, model_utils, module):
            pruning_tree = TransformPruningTree(cfg, model_utils, module)
            importance = pruning_tree.get_importance()
            trees.append(pruning_tree)
            trees_as_str.append(str(pruning_tree))
            tree_importances.append(importance)
        if is_attention_tree_root(cfg, module):
            attention_tree_generator = AttentionPruningTreeGenerator(module)
            for pruning_tree in attention_tree_generator.get_trees(cfg, model_utils):
                importance = pruning_tree.get_importance()
                trees.append(pruning_tree)
                trees_as_str.append(str(pruning_tree))
                tree_importances.append(importance)

    importances_and_tree_strs = sorted(zip(tree_importances, trees_as_str), key=lambda x: x[0])
    importances_and_trees = sorted(zip(tree_importances, trees), key=lambda x: x[0])

    if cfg.IMPORTANCES_SAVE_PATH:
        path_without_ext, ext = os.path.splitext(cfg.IMPORTANCES_SAVE_PATH)
        path_with_iter_num = f"{path_without_ext}-iter{iteration + 1}{ext}"
        print(f"Saving groups and importances to {path_with_iter_num}")
        with open(f"{path_with_iter_num}", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['importance', 'group'])  # header
            writer.writerows(importances_and_tree_strs)

    return importances_and_trees
