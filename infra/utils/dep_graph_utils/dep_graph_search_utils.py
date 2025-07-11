from typing import List, Set, Type

from torch.nn import Linear, Module
from torch_pruning.dependency import DependencyGraph, Group, Node

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.dep_graph_helper import DependencyDirection
from infra.utils.model_utils import ModelUtils
from infra.utils.module_utils.pruning_tree_collection_utils import is_transform_type

def get_adjacent_nodes(node: Node, search_direction: DependencyDirection) -> List[Module]:
    if search_direction == DependencyDirection.FORWARD:
        return node.outputs
    elif search_direction == DependencyDirection.BACKWARD:
        return node.inputs
    else:
        raise ValueError(f"Invalid search direction: {search_direction}")
    
def find_adjacent_operation_nodes(cfg: ConfigProtocol, source_node: Node, search_direction: DependencyDirection, operation_nodes: Set[Node]) -> Set[Node]:
    branches = get_adjacent_nodes(source_node, search_direction)
    
    for branch_node in branches:
        branch_node_module_type = type(branch_node.module)
        if branch_node_module_type in cfg.BASE_OPERATION_TYPES:
            operation_nodes.add(branch_node)
        if is_transform_type(cfg, branch_node_module_type):
            continue
        
        operation_nodes.union(
            find_adjacent_operation_nodes(cfg, branch_node, search_direction, operation_nodes)
        )

    return operation_nodes

def is_operation_prunable(cfg: ConfigProtocol, nonlinearity_node: Node, search_direction: DependencyDirection) -> int:
    def num_dependent_transforms(node, direction):
        num_dependent_paths = 0
        if is_transform_type(cfg, type(node.module)):
            return 1

        branches = get_adjacent_nodes(node, direction)
        for branch_node in branches:
            num_dependent_paths += num_dependent_transforms(branch_node, direction)
            if num_dependent_paths >= 2:
                break
        
        return num_dependent_paths
    
    return num_dependent_transforms(nonlinearity_node, search_direction) <= 1

def get_op_subtree(cfg: ConfigProtocol, module_node: Node) -> Set[Node]:
    depth_pruning_group = set()
    
    operation_nodes = find_adjacent_operation_nodes(
        cfg=cfg,
        source_node=module_node,
        search_direction=DependencyDirection.FORWARD,
        operation_nodes=set()
    )

    for node in operation_nodes:
        if is_operation_prunable(cfg, node, DependencyDirection.BACKWARD):
            depth_pruning_group.add(node)

    return depth_pruning_group

def get_param_subtree_singleton(dep_graph: DependencyGraph, module: Module, idxs: list, pruning_fn):
    full_subtree = dep_graph.get_pruning_group(module, pruning_fn, idxs)
    singleton_subtree = Group()
    setattr(singleton_subtree, '_group', full_subtree[:1])
    setattr(singleton_subtree, '_DG', dep_graph)
    return singleton_subtree
