from typing import List, Set

from torch.nn import Linear, Module
from torch_pruning.dependency import Node

from .model_utils import ModelUtils
from .dependency_direction import DependencyDirection

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

def get_operation_group(module: Module, model_utils: ModelUtils) -> Set[Node]:
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