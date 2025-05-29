from typing import List, Set, Type

from torch.nn import Linear, Module
from torch_pruning.dependency import Node

from . import constants as c
from .dependency_utils import DependencyDirection
from .model_utils import ModelUtils

def is_transform_type(node_module_type: Type):
    found = False
    for transform_type in c.BASE_TRANSFORM_TYPES:
        if issubclass(node_module_type, transform_type):
            found = True
            break

    return found

def get_adjacent_nodes(node: Node, search_direction: DependencyDirection) -> List[Module]:
    if search_direction == DependencyDirection.FORWARD:
        return node.outputs
    elif search_direction == DependencyDirection.BACKWARD:
        return node.inputs
    else:
        raise ValueError(f"Invalid search direction: {search_direction}")
    
def find_adjacent_operation_nodes(source_node: Node, search_direction: DependencyDirection, target_types: List, operation_nodes: Set[Node]) -> Set[Node]:
    branches = get_adjacent_nodes(source_node, search_direction)
    for branch_node in branches:
        branch_node_module_type = type(branch_node.module)
        if branch_node_module_type in target_types:
            operation_nodes.add(branch_node)
        if is_transform_type(branch_node_module_type):
            continue
        
        operation_nodes.union(
            find_adjacent_operation_nodes(branch_node, search_direction, target_types, operation_nodes)
        )

    return operation_nodes

def is_operation_prunable(nonlinearity_node: Node, search_direction: DependencyDirection) -> int:
    def num_dependent_transforms(node, direction):
        num_dependent_paths = 0
        if is_transform_type(type(node.module)):
            return 1

        branches = get_adjacent_nodes(node, direction)
        for branch_node in branches:
            num_dependent_paths += num_dependent_transforms(branch_node, direction)
            if num_dependent_paths >= 2:
                break
        
        return num_dependent_paths
    
    return num_dependent_transforms(nonlinearity_node, search_direction) <= 1

def get_operation_group(model_utils: ModelUtils, module: Module) -> Set[Node]:
    module_node = model_utils.dep_graph.module2node[module]
    depth_pruning_group = set()
    
    operation_nodes = find_adjacent_operation_nodes(
        source_node=module_node,
        search_direction=DependencyDirection.FORWARD,
        target_types=c.BASE_OPERATION_TYPES,
        operation_nodes=set()
    )

    for node in operation_nodes:
        if is_operation_prunable(node, DependencyDirection.BACKWARD):
            depth_pruning_group.add(node)

    return depth_pruning_group