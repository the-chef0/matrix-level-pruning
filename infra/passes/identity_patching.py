import numpy as np
from torch import nn
from torch_pruning.dependency import Node
from torch_pruning.ops import _ConcatOp, _ElementWiseOp

from infra.utils.module_utils.identity_types import AdditiveIdentity, ConcatenativeIdentity, MultiplicativeIdentity
from infra.utils.model_utils import ModelUtils
from infra.utils.dep_graph_utils.dep_graph_search_utils import get_param_subtree_singleton

class IdentityPatcher:

    def __init__(self, model_utils: ModelUtils):
        self.model_utils = model_utils
        self.model_modules = set(model_utils.model.modules()) #TODO: move to model utils?

        self.ARITHMETIC_TYPE_TO_IDENTITY_TYPE = {
            'AddBackward0': AdditiveIdentity,
            'MulBackward0': MultiplicativeIdentity
        }
        self.ARITHMETIC_TYPE_NAMES = set(self.ARITHMETIC_TYPE_TO_IDENTITY_TYPE.keys())

    def find_identity_operand_nodes(self, operands: list[Node]):
        id_operand_nodes = []

        for op_node in operands:
            if isinstance(op_node.module, nn.Identity):
                id_operand_nodes.append(op_node)

        return id_operand_nodes

    # TODO: make a generalized dep graph DFS function to use both here and for operations?
    # also there has got to be a less stupid way to do this than with the nested function
    def get_nearest_predecessor_module_node(self, node: Node):
        def recursive_case(node: Node):
            for inp in node.inputs:
                if inp.module in self.model_modules and not isinstance(inp.module, nn.Identity):
                    return inp
                else:
                    return recursive_case(inp)

        if node.module in self.model_modules and not isinstance(node.module, nn.Identity):
            return node
        else:
            return recursive_case(node)

    def get_nearest_successor_module_node(self, node: Node):
        def recursive_case(node: Node):
            for out in node.outputs:
                if out.module in self.model_modules and not isinstance(out.module, nn.Identity):
                    return out
                else:
                    return recursive_case(out)

        if node.module in self.model_modules and not isinstance(node.module, nn.Identity):
            return node
        else:
            return recursive_case(node)

    def patch_arithmetic_node_operands(self, node: Node, grad_fn_name: str):
        identity_operand_nodes = self.find_identity_operand_nodes(node.inputs)

        if identity_operand_nodes:
            fst_identity_module = identity_operand_nodes[0].module
            fst_operand = node.inputs[0]
            snd_operand = node.inputs[1]
            fst_pred = self.get_nearest_predecessor_module_node(fst_operand)
            snd_pred = self.get_nearest_predecessor_module_node(snd_operand)

            if fst_pred == snd_pred:
                arithmetic_identity_type = self.ARITHMETIC_TYPE_TO_IDENTITY_TYPE[grad_fn_name]
                identity_module_name = self.model_utils.module_to_name[fst_identity_module]
                print(f"Patching identity in {identity_module_name} with {arithmetic_identity_type.__name__}")
                self.model_utils.replace_module_by_name(
                    module_name=identity_module_name,
                    new_module=arithmetic_identity_type()
                )

    def adjust_concat_successor_dims(self, concat_node: Node):
        concat_sizes = concat_node.module.concat_sizes
        end_idx = np.sum(concat_sizes[:-1])
        idxs = list(range(end_idx))

        successor_module = self.get_nearest_successor_module_node(concat_node).module
        pruner = self.model_utils.dep_graph.get_pruner_of_module(successor_module)
        setattr(concat_node, 'dependencies', [])
        successor_subtree_singleton = get_param_subtree_singleton(
            model_utils=self.model_utils,
            module=successor_module,
            idxs=idxs,
            pruning_fn=pruner.prune_in_channels
        )
        successor_subtree_singleton.prune()

    def patch_concat_node_operands(self, node: Node):
        identity_operand_nodes = self.find_identity_operand_nodes(node.inputs)
        
        if identity_operand_nodes:
            predecessors = []
            for operand in node.inputs:
                predecessors.append(self.get_nearest_predecessor_module_node(operand))
            
            if all(pred == predecessors[0] for pred in predecessors):
                for id_operand_node in identity_operand_nodes[1:]:
                    identity_module = id_operand_node.module
                    identity_module_name = self.model_utils.module_to_name[identity_module]
                    self.model_utils.replace_module_by_name(
                        model_utils=self.model_utils,
                        module_name=identity_module_name,
                        new_module=ConcatenativeIdentity()
                    )
                
                self.adjust_concat_successor_dims(node)

    def patch(self):
        all_nodes = self.model_utils.dep_graph.module2node.values()
        for node in all_nodes:
            if isinstance(node.module, _ElementWiseOp):
                node_grad_fn_type = type(node.grad_fn)
                grad_fn_name = node_grad_fn_type.__name__

                if grad_fn_name in self.ARITHMETIC_TYPE_NAMES:
                    self.patch_arithmetic_node_operands(node, grad_fn_name)

            elif isinstance(node.module, _ConcatOp) and node.module.concat_sizes is not None:
                self.patch_concat_node_operands(node)
