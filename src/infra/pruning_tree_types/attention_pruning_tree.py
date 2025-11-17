from typing import Iterator
from typing import Iterator

import numpy as np
from torch.nn import Module
import torch_pruning as tp
from torch_pruning.pruner.importance import GroupMagnitudeImportance, GroupTaylorImportance

from config.config_protocol import ConfigProtocol, MHAProjection
from infra.pruning_tree_types.pruning_tree import PruningTree
from infra.utils.dep_graph_utils.dep_graph_search_utils import get_op_subtree, \
    get_param_subtree_singleton
from infra.utils.model_utils import ModelUtils
from infra.utils.module_utils.identity_types import IdentityWithGrad

class AttentionPruningTree(PruningTree):
    """Each attention pruning tree covers all the coupled structures involved in
    pruning one attention head. If h is the number of dimensions in one head,
    then each tree consists of at least a parameter subtree of 4 nodes:
    h rows of the Q, K and V projection matrices, and h columns of the O projection matrix.
    If there is only one attention head remaining, there can also be an operation
    subtree consisting of any operations after the attention module that have
    no other inputs.
    """
    def __init__(self,  cfg: ConfigProtocol, model_utils: ModelUtils, attention_module: Module, \
                 qo_idxs: list, kv_idxs: list, dims_per_head: int):
        """
        Args:
            cfg (ConfigProtocol): See class.
            model_utils (ModelUtils): See class.
            cfg (ConfigProtocol): See class.
            model_utils (ModelUtils): See class.
            attention_module (Module): The parent module containing this attention head.
            qo_idxs (list): A list of indices corresponding to the rows of the Q head
                grouping and the columns on the O projection matrix pertaining to this
                attention head.
            kv_idxs (list): A list of indices corresponding to the rows of the K/V projection
                matrices pertaining to this attention head.
        """
        super().__init__(model_utils)
        self.cfg = cfg
        self.model_utils = model_utils
        self.module = attention_module
        self.importance_fn = GroupTaylorImportance(normalizer=None, group_reduction=None)
        self.qo_idxs = qo_idxs
        self.kv_idxs = kv_idxs
        self.dims_per_head = dims_per_head

        # Use the Q, K, V and O projection variable names as defined in the config to
        # obtain the relevant modules from the attention parent module.
        q_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.Q])
        k_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.K])
        v_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.V])
        o_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.O])
        self.device = q_proj.weight.device

        # The parameter subtree is defined here and consists of 4 Torch-Pruning Group objects,
        # each encapsulating the information to prune one head from each of the projection
        # matrices.
        q_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=q_proj,
            idxs=self.qo_idxs,
            pruning_fn=tp.prune_linear_out_channels
        )
        k_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=k_proj,
            idxs=self.kv_idxs,
            pruning_fn=tp.prune_linear_out_channels
        )
        v_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=v_proj,
            idxs=self.kv_idxs,
            pruning_fn=tp.prune_linear_out_channels
        )
        o_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=o_proj,
            idxs=self.qo_idxs,
            pruning_fn=tp.prune_linear_in_channels
        )
        self.param_subtree = [
            q_singleton,
            k_singleton,
            v_singleton,
            o_singleton
        ]

        self.op_subtree = None
        # When there is only one head left, we can start pruning operations
        if self.module.k_proj.out_features == dims_per_head:
            q_proj_node = model_utils.dep_graph.module2node[attention_module.q_proj]
            k_proj_node = model_utils.dep_graph.module2node[attention_module.k_proj]
            v_proj_node = model_utils.dep_graph.module2node[attention_module.v_proj]
            o_proj_node = model_utils.dep_graph.module2node[attention_module.o_proj]

            qkv_side_subtree = get_op_subtree(
                cfg, q_proj_node,
                allowed_transform_nodes=set([q_proj_node, k_proj_node, v_proj_node])
            )
            o_side_subtree = get_op_subtree(cfg, o_proj_node)

            self.op_subtree = list(qkv_side_subtree | o_side_subtree)
    
    def get_param_subtree_importance(self) -> float:
        param_subtree_importances = [
            np.mean(self.importance_fn(param_subtree_node).cpu().numpy())
            for param_subtree_node in self.param_subtree
        ]
        return np.mean(param_subtree_importances)
    
    def get_op_importance(self) -> float:
        # Activations and other operations are treated as 0 importance for now, but keeping
        # this flexible.
        return 0
    
    def get_importance(self) -> float:
        return np.mean([
            self.get_param_subtree_importance(),
            self.get_op_importance()
        ])
    
    def update_op_subtree(self):
        # When there is only one head left, we can start pruning operations
        if self.module.k_proj.out_features == self.dims_per_head:
            q_proj_node = self.model_utils.dep_graph.module2node[self.module.q_proj]
            k_proj_node = self.model_utils.dep_graph.module2node[self.module.k_proj]
            v_proj_node = self.model_utils.dep_graph.module2node[self.module.v_proj]
            o_proj_node = self.model_utils.dep_graph.module2node[self.module.o_proj]

            qkv_side_subtree = get_op_subtree(
                self.cfg, q_proj_node,
                expected_dependent_nodes=set([q_proj_node, k_proj_node, v_proj_node])
            )
            o_side_subtree = get_op_subtree(self.cfg, o_proj_node)

            self.op_subtree = list(qkv_side_subtree | o_side_subtree)

    def prune(self, skip_listeners: bool = False) -> None:
        self.update_op_subtree()
        print(f"Pruning {self}")
        # Only pruning attention heads, not operations
        if self.op_subtree is None:
            for param_subtree_node in self.param_subtree:
                param_subtree_node.prune()
        # Pruning the last remaining head means we can prune the entire attention module,
        # along with any operations that might belong to it
        else:
            operation_modules = [node.module for node in self.op_subtree]
            for op_module in operation_modules:
                op_module_name = self.model_utils.module_to_name[op_module]
                self.model_utils.replace_module_by_name(op_module_name, IdentityWithGrad(self.device))
            
            attention_module_name = self.model_utils.module_to_name[self.module]
            self.model_utils.replace_module_by_name(attention_module_name, IdentityWithGrad(self.device))

        if not skip_listeners:
            self.call_post_prune_listeners()

    def __str__(self):
        module_str = f"{self.model_utils.module_to_name[self.module]}\n"
        kv_str = f"K/V head ({self.kv_idxs[0]}, {self.kv_idxs[-1]})\n"
        qo_str = f"Q/0 heads ({self.qo_idxs[0]}, {self.qo_idxs[-1]})\n"

        operation_str = ""
        if self.op_subtree is not None:
            operation_str += "Operations:\n"
            for op in self.op_subtree:
                operation_str += f"{op.module}\n"

        return module_str + kv_str + qo_str + operation_str

class AttentionPruningTreeGenerator:
    """Takes care of generating attention pruning trees, since each tree is
    parameterized by by indices that define the rows/columns corresponding
    to the given head in the Q, K, V and O weight matrices.
    With MHA, there are always at most as many K and V heads as there are
    Q heads. For MQA/GQA, we need to know which K and V heads correspond
    to which groups of Q heads.
    """
    def __init__(self, cfg: ConfigProtocol, attention_module: Module):
        self.cfg = cfg

        # Use the Q, K, V and O projection variable names as defined in the config to
        # obtain the relevant modules from the attention parent module.
        q_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.Q])
        k_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.K])
        self.module = attention_module

        # Many HuggingFace attention implementations rely on this config attribute so
        # hopefully it generalizes.
        self.dims_per_head = self.module.config.hidden_size // self.module.config.num_attention_heads
        self.num_kv_heads = k_proj.out_features // self.dims_per_head
        num_q_heads = q_proj.out_features // self.dims_per_head
        # In the case of MQA/GQA, we need to know how many Q heads map to one K/V head
        self.num_q_groups = num_q_heads // self.num_kv_heads
        # If one head has length e.g. 64 and there are 4 Q heads per K/V head,
        # then one group of Q heads is 256 long.
        self.dims_per_q_group = self.num_q_groups * self.dims_per_head

    def get_trees(self, model_utils: ModelUtils) -> Iterator[AttentionPruningTree]:
        # Generate attention trees and their indices by incrementing the
        # head index and offsetting the dimensions per Q group and K/V head
        # based on the head index.
        for head_idx in range(self.num_kv_heads):
            kv_start_idx = head_idx * self.dims_per_head
            kv_end_idx = (head_idx + 1) * self.dims_per_head
            qo_start_idx = head_idx * self.dims_per_q_group
            qo_end_idx = (head_idx + 1) * self.dims_per_q_group
            
            kv_idxs = list(range(kv_start_idx, kv_end_idx))
            qo_idxs = list(range(qo_start_idx, qo_end_idx))

            yield AttentionPruningTree(
                cfg=self.cfg,
                model_utils=model_utils,
                attention_module=self.module,
                qo_idxs=qo_idxs,
                kv_idxs=kv_idxs,
                dims_per_head=self.dims_per_head
            )
