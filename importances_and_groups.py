import csv

from base_model_utils import BaseModelUtils
from pruning_group import PruningGroup

from torch.nn import Linear, Module

def is_pruning_candidate(module: Module, model_utils: BaseModelUtils) -> bool:
    module_name = model_utils.module_to_name[module]
    is_matrix = isinstance(module, Linear)
    is_included = not any(excl in module_name for excl in model_utils.pruning_excluded_keywords)
    return is_matrix and is_included

def collect_groups(model_utils: BaseModelUtils, save_path: str):
    groups = []
    groups_as_str = []
    group_importances = []
    
    for module in model_utils.base_model.modules():
        if is_pruning_candidate(module, model_utils):
            pruning_group = PruningGroup(model_utils, module)
            importance = pruning_group.get_importance()
            groups.append(pruning_group)
            groups_as_str.append(str(pruning_group))
            group_importances.append(importance)

    importances_and_group_strs = sorted(zip(group_importances, groups_as_str))
    importances_and_groups = sorted(zip(group_importances, groups))

    print(f"Saving groups and importances to {save_path}")
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['importance', 'group'])  # header
        writer.writerows(importances_and_group_strs)

    return importances_and_groups
