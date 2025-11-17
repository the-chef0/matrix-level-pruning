"""Contains logic for analyzing the DepGraph representation of the model.
"""

"""Contains logic for analyzing the DepGraph representation of the model.
"""

from typing import List, Set

from torch.nn import Module, Identity
from torch_pruning.dependency import DependencyGraph, Group, Node
from torch_pruning.ops import _ConcatOp, _ElementWiseOp
from torch_pruning.pruner.function import BasePruningFunc

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.dep_graph_helper import DependencyDirection
from infra.utils.module_utils.pruning_tree_collection_utils import is_transform_type, is_identity_module

def get_adjacent_nodes(node: Node, search_direction: DependencyDirection) -> List[Node]:
    """For a given node, gets a list of either its successors or predecessors depending on
    search_direction.

    Args:
        node (Node): The given node.
        search_direction (DependencyDirection): An abstraction for the direction.
    Returns:
        List[Node]: The list of neighboring nodes in the direction.
    """
    if search_direction == DependencyDirection.FORWARD:
        return node.outputs
    elif search_direction == DependencyDirection.BACKWARD:
        return node.inputs
    else:
        raise ValueError(f"Invalid search direction: {search_direction}")
    
def find_adjacent_op_nodes(cfg: ConfigProtocol, source_node: Node, \
    search_direction: DependencyDirection, operation_nodes: Set[Node]) -> Set[Node]:
    """Searches in the direction given by search_direction for nearby operation nodes before
    running into other transforms.

    Args:
        cfg (ConfigProtocol): See class.
        source_node (Node): The node to start the search from.
        search_direction (DependencyDirection): An abstraction for the direction to search in.
        operation_nodes (Set[Node]): A set of all operation nodes found so far.
    Returns:
        Set[Node]: All operation nodes that can be found in the given direction before another
        transform.
    """
    base_operation_types = cfg.BASE_ACT_TYPES if search_direction == DependencyDirection.FORWARD \
        else cfg.BASE_NORM_TYPES

    branches = get_adjacent_nodes(source_node, search_direction)
    
    for branch_node in branches:
        branch_node_module_type = type(branch_node.module)
        if branch_node_module_type in base_operation_types:
            operation_nodes.add(branch_node)
        if is_transform_type(cfg, branch_node_module_type):
            continue
        # This is becoming more and more necessary - better write an abstraction
        if isinstance(branch_node.module, _ElementWiseOp):
            node_grad_fn_type = type(branch_node.grad_fn)
            grad_fn_name = node_grad_fn_type.__name__
            if grad_fn_name in ['AddBackward0', 'MulBackward0']:
                continue
        
        operation_nodes.union(
            find_adjacent_op_nodes(cfg, branch_node, search_direction, operation_nodes)
        )

    return operation_nodes

def is_op_prunable(cfg: ConfigProtocol, op_node: Node, \
    search_direction: DependencyDirection, expected_dependent_nodes: set[Node] = None) -> bool:
    """Searches from an operation node, back in the direction of its root transform, to see if
    there are any transforms other than the root that use it. If there is only one path back to a
    transform, it is only the root, and we can prune the operation together with it. 
    Otherwise, there are other transforms that depend on op_node.

    Args:
        cfg (ConfigProtocol): See class.
        op_node (Node): The operation node to start the search from.
        search_direction (DependencyDirection): The direction in the DepGraph pointing to the root
            transform that this process was triggered from.
    Returns:
        bool: Whether or not the operation is prunable.
    """

    def get_dependent_transforms(node: Node, direction: DependencyDirection) -> set:
        dep_transforms = set()
        if isinstance(node.module, _ElementWiseOp):
            node_grad_fn_type = type(node.grad_fn)
            grad_fn_name = node_grad_fn_type.__name__
            if grad_fn_name in ['AddBackward0', 'MulBackward0']:
                return dep_transforms
            
        if is_transform_type(cfg, type(node.module)) and not is_identity_module(node.module):
            dep_transforms.add(node)
            return dep_transforms

        branches = get_adjacent_nodes(node, direction)
        for branch_node in branches:
            dep_transforms = dep_transforms | get_dependent_transforms(branch_node, direction)
            if len(dep_transforms) > len(expected_dependent_nodes):
                break
        
        return dep_transforms
    
    dep_transforms = get_dependent_transforms(op_node, search_direction)
    return dep_transforms == expected_dependent_nodes

def find_nearest_nonid_module_node(source_node: Node, modules: set, \
    search_direction: DependencyDirection) -> Node:
    """Searches in search_direction for the depthwise nearest node that represents a module in
    the model.

    Args:
        source_node (Node): The node to initiate the search from.
        modules (set): A set of modules in the model to check against.
        search_direction (DependencyDirection): The direction to search in.
    Returns:
        Node: The depthwise nearest node that represents a module in the model.
    """

    # TODO: There is a way to rewrite this without nested functions but I couldn't get it working
    # and there are more important things to do
    def recursive_case(node: Node):
        branches = get_adjacent_nodes(node, search_direction)
        for branch in branches:
            if branch.module in modules and not is_identity_module(branch.module):
                return branch
            else:
                return recursive_case(branch)
            
    if source_node.module in modules and not is_identity_module(source_node.module):
        return source_node
    else:
        return recursive_case(source_node)

def get_op_subtree(cfg: ConfigProtocol, root_node: Node, expected_dependent_nodes: set[Node] = None)\
    -> Set[Node]:
    """Finds all operation nodes that might be coupled to the module represented by the root node, 
    and returns those that are depended on by only the root node.

    Args:
        cfg (ConfigProtocol): See class.
        root_node (Node): The node representing the module that is the root of the parameter
            subtree.
    """
    op_subtree = set()
    
    act_nodes = find_adjacent_op_nodes(
        cfg=cfg,
        source_node=root_node,
        search_direction=DependencyDirection.FORWARD,
        operation_nodes=set()
    )

    norm_nodes = find_adjacent_op_nodes(
        cfg=cfg,
        source_node=root_node,
        search_direction=DependencyDirection.BACKWARD,
        operation_nodes=set()
    )

    if not expected_dependent_nodes:
        expected_dependent_nodes = set([root_node])

    for node in act_nodes:
        if is_op_prunable(cfg, node, DependencyDirection.BACKWARD, expected_dependent_nodes):
            op_subtree.add(node)

    for node in norm_nodes:
        if is_op_prunable(cfg, node, DependencyDirection.FORWARD, expected_dependent_nodes):
            op_subtree.add(node)

    return op_subtree

def get_param_subtree_singleton(dep_graph: DependencyGraph, module: Module, idxs: list, \
    pruning_fn: BasePruningFunc) -> Group:
    """Creates a DepGraph Group object containg only the node representing the given module,
    allowing for width pruning functionality on isolated modules. This is because by default,
    the Group always grabs all dependent parameters, not just the root.

    Args:
        dep_graph (DepGraph): The relevant DepGraph instance.
        module (Module): The module to make a singleton from.
        idxs (list): The selected indexes to width-prune on the resulting node.
        pruning_fn (BasePruningFunc): The pruner class that makes sure the correct type gets
            assigned to the node (linear/conv).
    """

    full_subtree = dep_graph.get_pruning_group(module, pruning_fn, idxs)
    singleton_subtree = Group()
    setattr(singleton_subtree, '_group', full_subtree[:1])
    setattr(singleton_subtree, '_DG', dep_graph)
    return singleton_subtree
