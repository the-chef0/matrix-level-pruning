from typing import List, Set

from base_model_utils import BaseModelUtils
from dependency_direction import DependencyDirection

import numpy as np
from torch.nn import Module, Linear
from torch_pruning.dependency import Node
from torch_pruning.pruner.importance import GroupMagnitudeImportance

def get_transform_chain_direction(module: Module) -> DependencyDirection:
    assert isinstance(module, Linear), "Only linear layers are supported for now"

    if module.in_features > module.out_features:
        return DependencyDirection.backward
    elif module.in_features < module.out_features:
        return DependencyDirection.forward
    else:
        return DependencyDirection.not_needed

def get_adjacent_nodes(node: Node, search_direction: DependencyDirection) -> List[Module]:
    if search_direction == DependencyDirection.forward:
        return node.outputs
    elif search_direction == DependencyDirection.backward:
        return node.inputs
    else:
        raise ValueError(f"Invalid search direction: {search_direction}")
    
def find_closest_operation_nodes(source_node: Node, search_direction: DependencyDirection, target_types: List, operation_nodes: Set[Node]) -> Set[Node]:
    branches = get_adjacent_nodes(source_node, search_direction)
    for branch_node in branches:
        if type(branch_node.module) in target_types:
            operation_nodes.add(branch_node)
        elif isinstance(branch_node.module, Linear): # TODO: include all transforms from TP
            continue
        else:
            operation_nodes.union(
                find_closest_operation_nodes(branch_node, search_direction, target_types, operation_nodes)
            )
    return operation_nodes

def is_operation_prunable(nonlinearity_node: Node, search_direction: DependencyDirection) -> int:
    def num_dependent_matrices(node, direction):
        num_dependent_paths = 0
        if isinstance(node.module, Linear):
            return 1

        branches = get_adjacent_nodes(node, direction)
        for branch_node in branches:
            num_dependent_paths += num_dependent_matrices(branch_node, direction)
            if num_dependent_paths >= 2:
                break
        
        return num_dependent_paths
    
    return num_dependent_matrices(nonlinearity_node, search_direction) <= 1

def get_operation_group(module: Module, model_utils: BaseModelUtils) -> Set[Node]:
    module_node = model_utils.dep_graph.module2node[module]
    depth_pruning_group = set()
    
    operation_nodes_fw = find_closest_operation_nodes(
        source_node=module_node,
        search_direction=DependencyDirection.forward,
        target_types=model_utils.nonlinearities_forward,
        operation_nodes=set()
    )

    for fw_node in operation_nodes_fw:
        if is_operation_prunable(fw_node, DependencyDirection.backward):
            depth_pruning_group.add(fw_node)

    operation_nodes_bw = find_closest_operation_nodes(
        source_node=module_node,
        search_direction=DependencyDirection.backward,
        target_types=model_utils.nonlinearities_backward,
        operation_nodes=set()
    )

    for bw_node in operation_nodes_bw:
        if is_operation_prunable(bw_node, DependencyDirection.forward):
            depth_pruning_group.add(bw_node)

    return depth_pruning_group

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
