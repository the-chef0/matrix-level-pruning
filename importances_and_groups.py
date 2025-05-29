import csv
import os

from torch.nn import Linear, Module

from pruning_group import PruningGroup
import utils.constants as c
from utils.model_utils import ModelUtils

def is_group_root(module: Module, model_utils: ModelUtils) -> bool:
    is_transform = False
    for transform_type in c.BASE_TRANSFORM_TYPES:
        if issubclass(type(module), transform_type):
            is_transform = True
            break

    module_name = model_utils.module_to_name[module]
    is_not_forbidden = not any(kw in module_name for kw in c.FORBIDDEN_TRANSFORM_KEYWORDS)
    return is_transform and is_not_forbidden

def collect_groups(model_utils: ModelUtils, iteration: int, save_path: str):
    groups = []
    groups_as_str = []
    group_importances = []
    
    for module in model_utils.model.modules():
        if is_group_root(module, model_utils):
            pruning_group = PruningGroup(model_utils, module)
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
