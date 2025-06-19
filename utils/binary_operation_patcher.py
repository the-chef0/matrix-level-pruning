from enum import Enum

from torch_pruning.dependency import Node
from transformers.selective_binary_ops import SelectiveAdd, SelectiveMultiply

from .model_utils import ModelUtils

class MemoKeys(Enum):
    NEST = 0
    PRIM = 1

class BinaryOperationPatcher:

    def __init__(self, model_utils: ModelUtils):
        self.model_utils = model_utils
        self.operand_memo_table = {}
        self.selective_binop_types = set([SelectiveAdd, SelectiveMultiply])

    def is_binop(self, node: Node):
        module_type = type(node.module)
        return module_type in self.selective_binop_types

    def partition_operands(self, binop_node: Node):
        nested_binops = []
        primitive = []

        for operand in binop_node.inputs:
            if self.is_binop(operand):
                nested_binops.append(operand)
            else:
                primitive.append(operand)

        return nested_binops, primitive

    def get_partitioned_operands(self, binop_node: Node):
        if binop_node in self.operand_memo_table:
            nested = self.operand_memo_table[binop_node][MemoKeys.NEST]
            primitive = self.operand_memo_table[binop_node][MemoKeys.PRIM]
        else:
            nested, primitive = self.partition_operands(binop_node)
            self.operand_memo_table[binop_node] = {}
            self.operand_memo_table[binop_node][MemoKeys.NEST] = nested
            self.operand_memo_table[binop_node][MemoKeys.PRIM] = primitive

        return nested, primitive

    def unfold_operands(self, binop_node: Node):
        unfolded = []
        nested, primitive = self.get_partitioned_operands(binop_node)
        unfolded.extend(primitive)

        for n in nested:
            unfolded.extend(self.unfold_operands(n))

        return unfolded

    def patch(self):
        binop_nodes = []

        for _, candidate_node in self.model_utils.dep_graph.module2node.items():
            if self.is_binop(candidate_node):
                binop_nodes.append(candidate_node)

        for binop_node in binop_nodes:
            operands = self.unfold_operands(binop_node)
            if all(op == operands[0] for op in operands):
                node_module_name = self.model_utils.module_to_name[binop_node.module]
                print(f"{node_module_name} has both inputs pruned - enabling bypass")
                binop_node.module.bypass = True
