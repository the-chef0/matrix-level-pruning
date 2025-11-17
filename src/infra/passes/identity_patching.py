import numpy as np
from torch import nn
from torch_pruning.dependency import Node
from torch_pruning.ops import _ConcatOp, _ElementWiseOp

from config.config_protocol import ConfigProtocol
from infra.utils.module_utils.identity_types import AdditiveIdentity, ConcatenativeIdentity, \
    MultiplicativeIdentity
from infra.utils.model_utils import ModelUtils
from infra.utils.module_utils.pruning_tree_collection_utils import is_identity_module
from infra.utils.dep_graph_utils.dep_graph_helper import DependencyDirection
from infra.utils.dep_graph_utils.dep_graph_search_utils import find_nearest_nonid_module_node, \
    get_param_subtree_singleton

class IdentityPatcher:
    """
    Handles situations where, due to pruning, an element-wise arithmetic
    operation or a concat operation receives all identical operands.

    For example, before pruning there might be
    SiLU(gate_proj(x)) * up_proj(x)
    and after pruning it becomes
    id(id(x)) * id(x) = x * x.

    This pass can turn it into something like
    id(id(x)) * 1 = x,
    and from there, a model compiler will likely optimize away
    the redundant multiplication when compiling for inference.
    """

    def __init__(self, cfg: ConfigProtocol, model_utils: ModelUtils):
        self.cfg = cfg
        self.model_utils = model_utils
        self.model_utils.initialize_module_set()

        self.ARITHMETIC_TYPE_TO_IDENTITY_TYPE = {
            'AddBackward0': AdditiveIdentity,
            'MulBackward0': MultiplicativeIdentity
        }
        self.ARITHMETIC_TYPE_NAMES = set(self.ARITHMETIC_TYPE_TO_IDENTITY_TYPE.keys())

    def find_identity_operand_nodes(self, operands: list[Node]) -> list[Node]:
        """Given a list of operands (inputs) in an arithmetic node, extracts the ones that
        are linked to a module of the nn.Identity type.

        Args:
            operands (list[Node]): The list of operands/inputs
        Returns:
            list[Node]: A subset of operand nodes that are linked to a module of the
                nn.Identity type.
        """
        id_operand_nodes = []

        for op_node in operands:
            if is_identity_module(op_node.module):
                id_operand_nodes.append(op_node)

        return id_operand_nodes

    def get_redundant_concat_idxs(self, concat_node: Node) -> list[int]:
        """Given a concat node with all identical inputs, returns a list of
        indexes that cover all but the last input component of the resulting tensor.

        Args:
            concat_node (Node): A concat node with all identical inputs.
        Returns:
            list[int]: A list of indexes covering all but the last component of the
                concatenated tensor.
        """
        concat_sizes = concat_node.module.concat_sizes
        end_idx = np.sum(concat_sizes[:-1])
        return list(range(end_idx))
    
    def adjust_concat_successor_dims(self, concat_node: Node) -> None:
        """Given a concat node with all identical inputs, prunes the input
        dimension of all successor nodes so that they can take only one
        of the inputs.

        Args:
            concat_node (Node): A concat node with all identical inputs.
        """
        redundant_idxs = self.get_redundant_concat_idxs(concat_node)
        successor_module_node = find_nearest_nonid_module_node(
            source_node=concat_node,
            modules=self.model_utils.model_modules,
            search_direction=DependencyDirection.FORWARD
        )
        successor_module = successor_module_node.module
        
        pruner = self.model_utils.dep_graph.get_pruner_of_module(successor_module)
        setattr(concat_node, 'dependencies', [])
        successor_subtree_singleton = get_param_subtree_singleton(
            dep_graph=self.model_utils.dep_graph,
            module=successor_module,
            idxs=redundant_idxs,
            pruning_fn=pruner.prune_in_channels
        )
        
        successor_subtree_singleton.prune()

    def patch_arithmetic_node_operands(self, arithmetic_node: Node, grad_fn_name: str) -> None:
        """Checks if an arithmetic node has both identical operands,
        either directly or via identity nodes, and if it does,
        replaces one of them with the appropriate arithmetic identity element.
        
        Args:
            arithmetic_node (Node): An arithmetic operation node.
            grad_fn_name (str): The name of the node's Autograd gradient function
                to determine what kind of operation it is.
        """
        identity_operand_nodes = self.find_identity_operand_nodes(arithmetic_node.inputs)

        if identity_operand_nodes:
            fst_identity_module = identity_operand_nodes[0].module
            fst_operand = arithmetic_node.inputs[0]
            snd_operand = arithmetic_node.inputs[1]
            fst_pred = find_nearest_nonid_module_node(
                source_node=fst_operand,
                modules=self.model_utils.model_modules,
                search_direction=DependencyDirection.BACKWARD
            )
            snd_pred = find_nearest_nonid_module_node(
                source_node=snd_operand,
                modules=self.model_utils.model_modules,
                search_direction=DependencyDirection.BACKWARD
            )

            if fst_pred == snd_pred:
                arithmetic_identity_type = self.ARITHMETIC_TYPE_TO_IDENTITY_TYPE[grad_fn_name]
                identity_module_name = self.model_utils.module_to_name[fst_identity_module]
                print(f"Patching identity in {identity_module_name} with {arithmetic_identity_type.__name__}")
                self.model_utils.replace_module_by_name(
                    module_name=identity_module_name,
                    new_module=arithmetic_identity_type(fst_identity_module.device),
                )

    def patch_concat_node_operands(self, concat_node: Node) -> None:
        """Checks if a concat node has all identical operands,
        either directly or via identity nodes, and if it does,
        replaces all but one with a concatenative identity element.

        Args:
            concat_node (Node): A concat operation node.
        """
        identity_operand_nodes = self.find_identity_operand_nodes(concat_node.inputs)
        
        if identity_operand_nodes:
            fst_identity_module = identity_operand_nodes[0].module
            predecessors = []
            for operand in concat_node.inputs:
                predecessors.append(find_nearest_nonid_module_node(
                    source_node=operand,
                    modules=self.model_utils.model_modules,
                    search_direction=DependencyDirection.BACKWARD
                ))
           
            if all(pred == predecessors[0] for pred in predecessors):
                for id_operand_node in identity_operand_nodes[1:]:
                    identity_module = id_operand_node.module
                    identity_module_name = self.model_utils.module_to_name[identity_module]
                    print(f"Patching identity in {identity_module_name} with {ConcatenativeIdentity.__name__}")
                    self.model_utils.replace_module_by_name(
                        module_name=identity_module_name,
                        new_module=ConcatenativeIdentity(fst_identity_module.device),
                    )
                
                self.adjust_concat_successor_dims(concat_node)

    def patch(self):
        """Finds all artithmetic operation nodes and concat operation nodes
        in the DepGraph representation of the model and applies their
        corresponding patching functions.
        """
        all_nodes = self.model_utils.dep_graph.module2node.values()
        for node in all_nodes:
            if isinstance(node.module, _ElementWiseOp):
                node_grad_fn_type = type(node.grad_fn)
                grad_fn_name = node_grad_fn_type.__name__

                if grad_fn_name in self.ARITHMETIC_TYPE_NAMES:
                    self.patch_arithmetic_node_operands(node, grad_fn_name)

            elif isinstance(node.module, _ConcatOp) and node.module.concat_sizes is not None:
                self.patch_concat_node_operands(node)
