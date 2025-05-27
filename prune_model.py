import os
from typing import List, Tuple

from base_model_utils import BaseModelUtils
from pruning_group import PruningGroup

import torch
from torch.nn import Identity

def replace_module_by_name(model_utils: BaseModelUtils, module_name, new_module):
    # Split the module name into parts
    parts = module_name.split('.')
    
    # Get the parent module (everything except the last part)
    parent = model_utils.base_model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    # Replace the module
    setattr(parent, parts[-1], new_module)

def prune(model_utils: BaseModelUtils, importances_and_groups: List[Tuple[float, PruningGroup]], groups_to_prune: int, pruned_model_save_dir: str):

    for _, pruning_group in importances_and_groups[:groups_to_prune]:
        print(f"Pruning group: {str(pruning_group)}")
        # Prune transform dependency chain
        chain_importance_ranking = pruning_group.get_transform_chain_importance_ranking()
        num_channels_to_prune = pruning_group.root_dim_high - pruning_group.root_dim_low
        idxs_to_prune = [idx for (importance, idx) in chain_importance_ranking[:num_channels_to_prune]]
        pruning_group.transform_group.prune(idxs_to_prune)

        # Prune transform root
        root_module = pruning_group.get_transform_root_module()
        root_module_name = model_utils.module_to_name[root_module]
        replace_module_by_name(model_utils, root_module_name, Identity())

        # Prune operations
        operation_modules = pruning_group.get_operation_modules()
        for op_module in operation_modules:
            op_module_name = model_utils.module_to_name[op_module]
            replace_module_by_name(model_utils, op_module_name, Identity())

    pruned_model = model_utils.base_model

    model_utils.tokenizer.save_pretrained(pruned_model_save_dir)
    torch.save(pruned_model, os.path.join(pruned_model_save_dir, "model.pth"))
    print("Pruned model saved")
