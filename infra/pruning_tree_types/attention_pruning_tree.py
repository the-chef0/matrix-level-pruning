import numpy as np
from torch.nn import Module
import torch_pruning as tp
from torch_pruning.pruner.importance import GroupMagnitudeImportance

from config.config_protocol import ConfigProtocol
from infra.pruning_tree_types.pruning_tree import PruningTree
from infra.utils.dep_graph_utils.dep_graph_search_utils import get_operation_group, get_param_subtree_singleton
from infra.utils.model_utils import ModelUtils
from infra.utils.module_utils.identity_types import IdentityWithGrad

# TODO: fix everything here
class AttentionPruningTree(PruningTree):

    def __init__(self,  cfg: ConfigProtocol, model_utils: ModelUtils, attention_module: Module, qo_idxs: list, kv_idxs: list, dims_per_head: int):
        self.model_utils = model_utils
        self.module = attention_module
        self.importance_fn = GroupMagnitudeImportance(normalizer=None, group_reduction=None)
        self.qo_idxs_flat = np.array(qo_idxs).flatten()
        self.qo_idxs = qo_idxs
        self.kv_idxs = kv_idxs

        self.q_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=self.module.q_proj,
            idxs=self.qo_idxs_flat,
            pruning_fn=tp.prune_linear_out_channels
        )
        self.k_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=self.module.k_proj,
            idxs=self.kv_idxs,
            pruning_fn=tp.prune_linear_out_channels
        )
        self.v_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=self.module.v_proj,
            idxs=self.kv_idxs,
            pruning_fn=tp.prune_linear_out_channels
        )
        self.o_singleton = get_param_subtree_singleton(
            dep_graph=model_utils.dep_graph,
            module=self.module.o_proj,
            idxs=self.qo_idxs_flat,
            pruning_fn=tp.prune_linear_in_channels
        )

        self.operation_group = None
        # When there is only one head left, we can start pruning operations
        if self.module.k_proj.out_features == dims_per_head:
            o_proj_node = model_utils.dep_graph.module2node[attention_module.o_proj]
            self.operation_group = list(get_operation_group(cfg, o_proj_node))

    def get_importance(self):
        q_heads_importance = np.sum(self.importance_fn(self.q_singleton).cpu().numpy())
        k_heads_importance = np.sum(self.importance_fn(self.k_singleton).cpu().numpy())
        v_heads_importance = np.sum(self.importance_fn(self.v_singleton).cpu().numpy())
        o_heads_importance = np.sum(self.importance_fn(self.o_singleton).cpu().numpy())
        return q_heads_importance \
            + k_heads_importance \
            + v_heads_importance \
            + o_heads_importance

    def prune(self):
        # Only pruning attention heads, not operations
        if self.operation_group is None:
            self.q_singleton.prune()
            self.k_singleton.prune()
            self.v_singleton.prune()
            self.o_singleton.prune()
        # Pruning the last remaining head means we can prune the entire attention module,
        # along with any operations that might belong to it
        else:
            operation_modules = [node.module for node in self.operation_group]
            for op_module in operation_modules:
                op_module_name = self.model_utils.module_to_name[op_module]
                self.model_utils.replace_module_by_name(op_module_name, IdentityWithGrad())
            
            attention_module_name = self.model_utils.module_to_name[self.module]
            self.model_utils.replace_module_by_name(attention_module_name, IdentityWithGrad())

    def __str__(self):
        module_str = f"{self.model_utils.module_to_name[self.module]}\n"
        kv_str = f"K/V head ({self.kv_idxs[0]}, {self.kv_idxs[-1]})\n"
        qo_str = "Q/O heads:\n"
        for qo_head_idxs in self.qo_idxs:
            start_idx, end_idx = qo_head_idxs[0], qo_head_idxs[-1]
            qo_str += f"({start_idx}, {end_idx})\n"

        operation_str = "Operations:\n"
        if self.operation_group is not None:
            for op in self.operation_group:
                operation_str += f"{op.module}\n"

        return module_str + kv_str + qo_str + operation_str

class AttentionPruningTreeGenerator:
    def __init__(self, attention_module: Module):
        self.module = attention_module
        self.dims_per_head = self.module.config.hidden_size // self.module.config.num_attention_heads
        assert self.module.k_proj.out_features == self.module.v_proj.out_features
        self.num_kv_heads = self.module.k_proj.out_features // self.dims_per_head
        num_q_heads = self.module.q_proj.out_features // self.dims_per_head
        self.num_kv_groups = num_q_heads // self.num_kv_heads

    def get_trees(self, cfg: ConfigProtocol, model_utils: ModelUtils):
        for kv_head_offset in range(self.num_kv_heads):
            kv_head_start_idx = kv_head_offset * self.dims_per_head
            kv_head_end_idx = (kv_head_offset + 1) * self.dims_per_head
            kv_idxs = list(range(kv_head_start_idx, kv_head_end_idx))

            qo_idxs = []
            for q_head_offset in range(self.num_kv_groups):
                q_head_start_idx = kv_head_start_idx + (q_head_offset * self.module.k_proj.out_features)
                q_head_end_idx = kv_head_end_idx + (q_head_offset * self.module.k_proj.out_features)
                q_head_idxs = list(range(q_head_start_idx, q_head_end_idx))
                qo_idxs.append(q_head_idxs)

            yield AttentionPruningTree(
                cfg=cfg,
                model_utils=model_utils,
                attention_module=self.module,
                qo_idxs=qo_idxs,
                kv_idxs=kv_idxs,
                dims_per_head=self.dims_per_head
            )
