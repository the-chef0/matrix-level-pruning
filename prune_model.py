from pruning_group import PruningGroup
from utils.base_model_utils import BaseModelUtils

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

def prune(model_utils: BaseModelUtils, group_to_prune: PruningGroup):
    print(f"Pruning group: {str(group_to_prune)}")
    
    # Prune transform group dimension dependency chain, only if the root is a
    # rectangular matrix and has a dimension dependency chain
    if group_to_prune.transform_group_chain:
        chain_importance_ranking = group_to_prune.get_transform_chain_importance_ranking()
        num_channels_to_prune = group_to_prune.root_dim_high - group_to_prune.root_dim_low
        idxs_to_prune = [idx for (importance, idx) in chain_importance_ranking[:num_channels_to_prune]]
        group_to_prune.transform_group.prune(idxs_to_prune)

    # Prune transform group root
    root_module = group_to_prune.get_transform_root_module()
    root_module_name = model_utils.module_to_name[root_module]
    replace_module_by_name(model_utils, root_module_name, Identity())

    # Prune operations
    operation_modules = group_to_prune.get_operation_modules()
    for op_module in operation_modules:
        op_module_name = model_utils.module_to_name[op_module]
        replace_module_by_name(model_utils, op_module_name, Identity())
