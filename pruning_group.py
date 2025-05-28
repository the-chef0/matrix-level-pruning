from utils.base_model_utils import BaseModelUtils
from utils.dependency_direction import DependencyDirection
from utils.pruning_group_utils import get_operation_group, get_transform_chain_direction

import numpy as np
from torch.nn import Module, Linear
from torch_pruning.pruner.importance import GroupMagnitudeImportance

class PruningGroup:
    def __init__(self, model_utils: BaseModelUtils, module: Module):
        transform_chain_direction = get_transform_chain_direction(module)
        self.root_dim_low, self.root_dim_high = self.get_module_dims(module)
        self.channel_idxs = [i for i in range(self.root_dim_high)]

        if transform_chain_direction == DependencyDirection.not_needed:
            self.transform_group = model_utils.dep_graph.get_pruning_group(module, DependencyDirection.forward, self.channel_idxs)
            self.transform_group_root = self.transform_group[:1]
            self.transform_group_chain = None
        else:
            self.transform_group = model_utils.dep_graph.get_pruning_group(module, transform_chain_direction, self.channel_idxs)
            self.transform_group_root = self.transform_group[:1]
            self.transform_group_chain = self.transform_group[1:]

        self.operation_group = list(get_operation_group(module, model_utils))
        self.importance_fn = GroupMagnitudeImportance(normalizer=None)
        self.model_utils = model_utils
        
        self.importance = None
        self.transform_chain_importance_ranking = None

    def get_module_dims(self, module: Module):
        root_dim_low = np.min([module.in_features, module.out_features])
        root_dim_high = np.max([module.in_features, module.out_features])
        return root_dim_low, root_dim_high

    def get_transform_chain_importance_ranking(self):
        if self.transform_chain_importance_ranking is None:
            importance = self.importance_fn(self.transform_group_chain).cpu().numpy()
            all_channel_idxs = [i for i in range(self.root_dim_high)]
            self.transform_chain_importance_ranking = sorted(zip(importance, all_channel_idxs))
        return self.transform_chain_importance_ranking

    def get_transform_chain_importance(self):
        if self.transform_group_chain is not None:
            chain_importance_ranking = self.get_transform_chain_importance_ranking()
            importances_ranked = [importance for (importance, idx) in chain_importance_ranking]
            return np.sum(importances_ranked[:self.root_dim_low])
        else:
            return 0

    def get_transform_root_importance(self):
        importance = self.importance_fn(self.transform_group_root).cpu().numpy()
        return np.sum(importance)

    def get_operation_importance(self):
        return 0

    def get_importance(self):
        if self.importance is None:
            self.importance = self.get_transform_root_importance() + \
                self.get_operation_importance() + \
                self.get_transform_chain_importance()
            
        return self.importance

    def get_transform_root_module(self):
        return self.transform_group_root[0].dep.source.module
    
    def get_transform_chain_modules(self):
        modules = []
        if self.transform_group_chain:
            for item in self.transform_group_chain:
                if isinstance(item.dep.target.module, Linear):
                    modules.append(item.dep.target.module)
        return modules

    def get_operation_modules(self):
        return [node.module for node in self.operation_group]

    def __str__(self):
        root_module = self.get_transform_root_module()
        root_str = f"Root:\n{self.model_utils.module_to_name[root_module]}\n"

        operation_str = "Operations:\n"
        for op in self.operation_group:
            operation_str += f"{op.module}\n"

        chain_modules = self.get_transform_chain_modules()
        chain_str = "Dimension dependency chain:\n"
        for module in chain_modules:
            chain_str += f"{self.model_utils.module_to_name[module]}\n"

        return root_str + operation_str + chain_str
