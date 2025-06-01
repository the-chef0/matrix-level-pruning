import numpy as np
from torch.nn import Module
from torch_pruning.pruner.importance import GroupMagnitudeImportance

from utils.dependency_utils import RootDependencyUtils, DependencyDirection
from utils.functional import is_transform_type
from utils.model_utils import ModelUtils
from utils.operation_group_utils import get_operation_group

class PruningGroup:
    def __init__(self, model_utils: ModelUtils, root_module: Module):
        self.root_dependency_utils = RootDependencyUtils(model_utils, root_module)
        self.root_dim_low, self.root_dim_high = self.get_module_dims(root_module)
        self.channel_idxs = [i for i in range(self.root_dim_high)]

        self.transform_group = model_utils.dep_graph.get_pruning_group(
            root_module,
            self.root_dependency_utils.fn,
            self.channel_idxs
        )
        self.transform_group_root = self.transform_group[:1]

        if self.root_dependency_utils.direction != DependencyDirection.NOT_APPLICABLE:
            self.transform_group_chain = self.transform_group[1:]
        else:
            self.transform_group_chain = None
            
        self.operation_group = list(get_operation_group(model_utils, root_module))
        self.importance_fn = GroupMagnitudeImportance(normalizer=None)
        self.model_utils = model_utils
        
        self.importance = None
        self.transform_chain_importance_ranking = None

    def get_module_dims(self, module: Module):
        root_dim_low = np.min([self.root_dependency_utils.in_channels, self.root_dependency_utils.out_channels])
        root_dim_high = np.max([self.root_dependency_utils.in_channels, self.root_dependency_utils.out_channels])
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
                item_module = item.dep.target.module
                if is_transform_type(type(item_module)):
                    modules.append(item_module)
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
