import csv
import os

from torch.nn import Module

from pruning_group import AttentionPruningGroupGenerator, TransformPruningGroup
import utils.constants as c
from utils.functional import is_attention_type, is_transform_type, meets_exclusion_criteria
from utils.model_utils import ModelUtils

def is_group_root(model_utils: ModelUtils, module: Module) -> bool:
    is_transform = is_transform_type(type(module))
    is_not_excluded = not meets_exclusion_criteria(model_utils, module)
    return is_transform and is_not_excluded

def is_attention_root(model_utils: ModelUtils, module: Module) -> bool:
    return is_attention_type(type(module))

def collect_groups(model_utils: ModelUtils, iteration: int, save_path: str):
    groups = []
    groups_as_str = []
    group_importances = []
    
    for module in model_utils.model.modules():
        if is_group_root(model_utils, module):
            pruning_group = TransformPruningGroup(model_utils, module)
            importance = pruning_group.get_importance()
            groups.append(pruning_group)
            groups_as_str.append(str(pruning_group))
            group_importances.append(importance)
        if is_attention_root(model_utils, module):
            attention_group_generator = AttentionPruningGroupGenerator(module)
            for pruning_group in attention_group_generator.get_groups(model_utils):
                importance = pruning_group.get_importance()
                groups.append(pruning_group)
                groups_as_str.append(str(pruning_group))
                group_importances.append(importance)

    importances_and_group_strs = sorted(zip(group_importances, groups_as_str), key=lambda x: x[0])
    importances_and_groups = sorted(zip(group_importances, groups), key=lambda x: x[0])

    if save_path:
        path_without_ext, ext = os.path.splitext(save_path)
        path_with_iter_num = f"{path_without_ext}-iter{iteration + 1}{ext}"
        print(f"Saving groups and importances to {path_with_iter_num}")
        with open(f"{path_with_iter_num}", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['importance', 'group'])  # header
            writer.writerows(importances_and_group_strs)

    return importances_and_groups
