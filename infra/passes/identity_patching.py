import numpy as np
from torch import nn
from torch_pruning.dependency import Node
from torch_pruning.ops import _ConcatOp, _ElementWiseOp

from config.config_protocol import ConfigProtocol
from infra.utils.module_utils.identity_types import AdditiveIdentity, ConcatenativeIdentity, \
    MultiplicativeIdentity
from infra.utils.model_utils import ModelUtils
from infra.utils.dep_graph_utils.dep_graph_helper import DependencyDirection
from infra.utils.dep_graph_utils.dep_graph_search_utils import find_nearest_nonid_module_node, \
    get_param_subtree_singleton

class IdentityPatcher:

    def __init__(self, cfg: ConfigProtocol, model_utils: ModelUtils):
        self.cfg = cfg
        self.model_utils = model_utils
        self.model_utils.initialize_module_set()

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

    def get_redundant_concat_idxs(self, concat_node: Node):
        concat_sizes = concat_node.module.concat_sizes
        end_idx = np.sum(concat_sizes[:-1])
        return list(range(end_idx))
    
    def adjust_concat_successor_dims(self, concat_node: Node):
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

    def patch_arithmetic_node_operands(self, node: Node, grad_fn_name: str):
        identity_operand_nodes = self.find_identity_operand_nodes(node.inputs)

        if identity_operand_nodes:
            fst_identity_module = identity_operand_nodes[0].module
            fst_operand = node.inputs[0]
            snd_operand = node.inputs[1]
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
                    new_module=arithmetic_identity_type(device=self.cfg.DEVICE)
                )

    def patch_concat_node_operands(self, node: Node):
        identity_operand_nodes = self.find_identity_operand_nodes(node.inputs)
        
        if identity_operand_nodes:
            predecessors = []
            for operand in node.inputs:
                predecessors.append(find_nearest_nonid_module_node(
                    source_node=operand,
                    modules=self.model_utils.model_modules,
                    search_direction=DependencyDirection.BACKWARD
                ))
            
            if all(pred == predecessors[0] for pred in predecessors):
                for id_operand_node in identity_operand_nodes[1:]:
                    identity_module = id_operand_node.module
                    identity_module_name = self.model_utils.module_to_name[identity_module]
                    self.model_utils.replace_module_by_name(
                        model_utils=self.model_utils,
                        module_name=identity_module_name,
                        new_module=ConcatenativeIdentity(device=self.cfg.DEVICE)
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
