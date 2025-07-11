import numpy as np
from torch.nn import Module
import torch_pruning as tp
from torch_pruning.pruner.importance import GroupMagnitudeImportance

from config.config_protocol import ConfigProtocol, MHAProjection
from infra.pruning_tree_types.pruning_tree import PruningTree
from infra.utils.dep_graph_utils.dep_graph_search_utils import get_op_subtree, get_param_subtree_singleton
from infra.utils.model_utils import ModelUtils
from infra.utils.module_utils.identity_types import IdentityWithGrad

class AttentionPruningTree(PruningTree):

    def __init__(self,  cfg: ConfigProtocol, model_utils: ModelUtils, attention_module: Module, \
                 qo_idxs: list, kv_idxs: list, dims_per_head: int):     
        self.model_utils = model_utils
        self.module = attention_module
        self.importance_fn = GroupMagnitudeImportance(normalizer=None, group_reduction=None)
        self.qo_idxs = qo_idxs
        self.kv_idxs = kv_idxs

        q_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.Q])
        k_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.K])
        v_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.V])
        o_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.O])

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
            o_proj_node = model_utils.dep_graph.module2node[attention_module.o_proj]
            self.op_subtree = list(get_op_subtree(cfg, o_proj_node))
    
    def get_param_subtree_importance(self):
        param_subtree_importances = [
            np.sum(self.importance_fn(param_subtree_node).cpu().numpy())
            for param_subtree_node in self.param_subtree
        ]
        return np.sum(param_subtree_importances)
    
    def get_op_importance(self):
        return 0
    
    def get_importance(self):
        return self.get_param_subtree_importance() \
            + self.get_op_importance()

    def prune(self):
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
                self.model_utils.replace_module_by_name(op_module_name, IdentityWithGrad())
            
            attention_module_name = self.model_utils.module_to_name[self.module]
            self.model_utils.replace_module_by_name(attention_module_name, IdentityWithGrad())

    def __str__(self):
        module_str = f"{self.model_utils.module_to_name[self.module]}\n"
        kv_str = f"K/V head ({self.kv_idxs[0]}, {self.kv_idxs[-1]})\n"
        qo_str = f"Q/0 heads ({self.qo_idxs[0]}, {self.qo_idxs[-1]})\n"

        operation_str = "Operations:\n"
        if self.op_subtree is not None:
            for op in self.op_subtree:
                operation_str += f"{op.module}\n"

        return module_str + kv_str + qo_str + operation_str

class AttentionPruningTreeGenerator:
    def __init__(self, cfg: ConfigProtocol, attention_module: Module):
        self.cfg = cfg

        q_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.Q])
        k_proj = getattr(attention_module, cfg.MHA_PROJECTION_NAME_MAPPING[MHAProjection.K])
        self.module = attention_module

        self.dims_per_head = self.module.config.hidden_size // self.module.config.num_attention_heads
        self.num_kv_heads = k_proj.out_features // self.dims_per_head
        num_q_heads = q_proj.out_features // self.dims_per_head
        self.num_q_groups = num_q_heads // self.num_kv_heads
        self.dims_per_q_group = self.num_q_groups * self.dims_per_head

    def get_trees(self, model_utils: ModelUtils):
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
