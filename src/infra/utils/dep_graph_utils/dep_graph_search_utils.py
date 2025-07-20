from typing import List, Set

from torch.nn import Module, Identity
from torch_pruning.dependency import DependencyGraph, Group, Node
from torch_pruning.pruner.function import BasePruningFunc

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.dep_graph_helper import DependencyDirection
from infra.utils.module_utils.pruning_tree_collection_utils import is_transform_type

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
        cfg (ConfigProtocol): See class docstring.
        source_node (Node): The node to start the search from.
        search_direction (DependencyDirection): An abstraction for the direction to search in.
        operation_nodes (Set[Node]): A set of all operation nodes found so far.
    Returns:
        Set[Node]: All operation nodes that can be found in the given direction before another
        transform.
    """

    branches = get_adjacent_nodes(source_node, search_direction)
    
    for branch_node in branches:
        branch_node_module_type = type(branch_node.module)
        if branch_node_module_type in cfg.BASE_OPERATION_TYPES:
            operation_nodes.add(branch_node)
        if is_transform_type(cfg, branch_node_module_type):
            continue
        
        operation_nodes.union(
            find_adjacent_op_nodes(cfg, branch_node, search_direction, operation_nodes)
        )

    return operation_nodes

def is_op_prunable(cfg: ConfigProtocol, op_node: Node, \
    search_direction: DependencyDirection) -> bool:
    """Searches from an operation node, back in the direction of its root transform, to see if
    there are any transforms other than the root that use it. If there is only one path back to a
    transform, it is only the root, and we can prune the operation together with it. 
    Otherwise, there are other transforms that depend on op_node.

    Args:
        cfg (ConfigProtocol): See class docstring.
        op_node (Node): The operation node to start the search from.
        search_direction (DependencyDirection): The direction in the DepGraph pointing to the root
            transform that this process was triggered from.
    Returns:
        bool: Whether or not the operation is prunable.
    """

    def num_dependent_transforms(node: Node, direction: DependencyDirection) -> int:
        num_dependent_paths = 0
        if is_transform_type(cfg, type(node.module)):
            return 1

        branches = get_adjacent_nodes(node, direction)
        for branch_node in branches:
            num_dependent_paths += num_dependent_transforms(branch_node, direction)
            if num_dependent_paths >= 2:
                break
        
        return num_dependent_paths
    
    return num_dependent_transforms(op_node, search_direction) <= 1

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
            if branch.module in modules and not isinstance(branch.module, Identity):
                return branch
            else:
                return recursive_case(branch)
            
    if source_node.module in modules and not isinstance(source_node.module, Identity):
        return source_node
    else:
        return recursive_case(source_node)

def get_op_subtree(cfg: ConfigProtocol, root_node: Node) -> Set[Node]:
    """Finds all operation nodes that might be coupled to the module represented by the root node, 
    and returns those that are depended on by only the root node.

    Args:
        cfg (ConfigProtocol): See class docstring.
        root_node (Node): The node representing the module that is the root of the parameter
            subtree.
    """
    op_subtree = set()
    
    operation_nodes = find_adjacent_op_nodes(
        cfg=cfg,
        source_node=root_node,
        search_direction=DependencyDirection.FORWARD,
        operation_nodes=set()
    )

    for node in operation_nodes:
        if is_op_prunable(cfg, node, DependencyDirection.BACKWARD):
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
