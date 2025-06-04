import numpy as np
from torch.nn import Linear, Module
import torch_pruning as tp
from torch_pruning.dependency import Group
from torch_pruning.pruner.importance import GroupMagnitudeImportance

from utils.dependency_utils import RootDependencyUtils, DependencyDirection
from utils.functional import is_transform_type
from utils.model_utils import ModelUtils
from utils.operation_group_utils import get_operation_group

class AttentionPruningGroup:
    def get_solo_group(self, model_utils: ModelUtils, proj: Linear, idxs: list, pruning_fn):
        full_group = model_utils.dep_graph.get_pruning_group(proj, pruning_fn, idxs)
        solo_group = Group()
        solo_group._group = full_group[:1]
        solo_group._DG = model_utils.dep_graph
        return solo_group

    def __init__(self, model_utils: ModelUtils, attention_module: Module, qo_idxs: list, kv_idxs: list):
        self.importance_fn = GroupMagnitudeImportance(normalizer=None, group_reduction=None)
        qo_idxs_flat = np.array(qo_idxs).flatten()
        self.q_group_solo = self.get_solo_group(model_utils, attention_module.q_proj, qo_idxs_flat, tp.prune_linear_out_channels)
        self.k_group_solo = self.get_solo_group(model_utils, attention_module.k_proj, kv_idxs, tp.prune_linear_out_channels)
        self.v_group_solo = self.get_solo_group(model_utils, attention_module.v_proj, kv_idxs, tp.prune_linear_out_channels)
        self.o_group_solo = self.get_solo_group(model_utils, attention_module.o_proj, qo_idxs_flat, tp.prune_linear_in_channels)
        self.qo_idxs = qo_idxs
        self.kv_idxs = kv_idxs

    def get_importance(self):
        q_heads_importance = np.sum(self.importance_fn(self.q_group_solo).cpu().numpy())
        k_heads_importance = np.sum(self.importance_fn(self.k_group_solo).cpu().numpy())
        v_heads_importance = np.sum(self.importance_fn(self.v_group_solo).cpu().numpy())
        o_heads_importance = np.sum(self.importance_fn(self.o_group_solo).cpu().numpy())
        return q_heads_importance \
            + k_heads_importance \
            + v_heads_importance \
            + o_heads_importance

    def __str__(self):
        kv_str = f"K/V head ({self.kv_idxs[0]}, {self.kv_idxs[-1]})\n"
        qo_str = "Q/O heads:\n"
        for qo_head_idxs in self.qo_idxs:
            start_idx, end_idx = qo_head_idxs[0], qo_head_idxs[-1]
            qo_str += f"({start_idx}, {end_idx})\n"
        return kv_str + qo_str

class AttentionPruningGroupGenerator:
    def __init__(self, attention_module: Module):
        self.attention_module = attention_module
        self.dims_per_head = attention_module.config.hidden_size // attention_module.config.num_attention_heads
        assert attention_module.k_proj.out_features == attention_module.v_proj.out_features
        self.num_kv_heads = attention_module.k_proj.out_features // self.dims_per_head
        num_q_heads = attention_module.q_proj.out_features // self.dims_per_head
        self.num_kv_groups = num_q_heads // self.num_kv_heads

    def get_groups(self, model_utils: ModelUtils):
        for kv_head_offset in range(self.num_kv_heads):
            kv_head_start_idx = kv_head_offset * self.dims_per_head
            kv_head_end_idx = (kv_head_offset + 1) * self.dims_per_head
            kv_idxs = list(range(kv_head_start_idx, kv_head_end_idx))

            qo_idxs = []
            for q_head_offset in range(self.num_kv_groups):
                q_head_start_idx = kv_head_start_idx + (q_head_offset * self.attention_module.k_proj.out_features)
                q_head_end_idx = kv_head_end_idx + (q_head_offset * self.attention_module.k_proj.out_features)
                q_head_idxs = list(range(q_head_start_idx, q_head_end_idx))
                qo_idxs.append(q_head_idxs)

            yield AttentionPruningGroup(model_utils, self.attention_module, qo_idxs, kv_idxs)

class PruningGroup:
    def __init__(self, model_utils: ModelUtils, root_module: Module):
        self.root_dependency_utils = RootDependencyUtils(model_utils, root_module)
        self.root_dim_low, self.root_dim_high = self.get_module_dims(root_module)
        self.channel_idxs = [i for i in range(self.root_dim_high)]

        self.transform_group = model_utils.dep_graph.get_pruning_group(
            root_module,
            self.root_dependency_utils.fn,
            self.channel_idxs
        )
        self.transform_group_root = self.transform_group[:1]

        if self.root_dependency_utils.direction != DependencyDirection.NOT_APPLICABLE:
            self.transform_group_chain = self.transform_group[1:]
        else:
            self.transform_group_chain = None
            
        self.operation_group = list(get_operation_group(model_utils, root_module))
        self.importance_fn = GroupMagnitudeImportance(normalizer=None)
        self.model_utils = model_utils
        
        self.importance = None
        self.transform_chain_importance_ranking = None

    def get_module_dims(self, module: Module):
        root_dim_low = np.min([self.root_dependency_utils.in_channels, self.root_dependency_utils.out_channels])
        root_dim_high = np.max([self.root_dependency_utils.in_channels, self.root_dependency_utils.out_channels])
        return root_dim_low, root_dim_high

    def get_transform_chain_importance_ranking(self):
        if self.transform_chain_importance_ranking is None:
            importance = self.importance_fn(self.transform_group_chain).cpu().numpy()
            all_channel_idxs = [i for i in range(self.root_dim_high)]
            self.transform_chain_importance_ranking = sorted(zip(importance, all_channel_idxs))
        return self.transform_chain_importance_ranking

    def get_transform_chain_importance(self):
        if self.transform_group_chain is not None:
            chain_importance_ranking = self.get_transform_chain_importance_ranking()
            importances_ranked = [importance for (importance, idx) in chain_importance_ranking]
            return np.sum(importances_ranked[:self.root_dim_low])
        else:
            return 0

    def get_transform_root_importance(self):
        importance = self.importance_fn(self.transform_group_root).cpu().numpy()
        return np.sum(importance)

    def get_operation_importance(self):
        return 0

    def get_importance(self):
        if self.importance is None:
            self.importance = self.get_transform_root_importance() + \
                self.get_operation_importance() + \
                self.get_transform_chain_importance()
            
        return self.importance

    def get_transform_root_module(self):
        return self.transform_group_root[0].dep.source.module
    
    def get_transform_chain_modules(self):
        modules = []
        if self.transform_group_chain:
            for item in self.transform_group_chain:
                item_module = item.dep.target.module
                if is_transform_type(type(item_module)):
                    modules.append(item_module)
        return modules

    def get_operation_modules(self):
        return [node.module for node in self.operation_group]

    def __str__(self):
        root_module = self.get_transform_root_module()
        root_str = f"Root:\n{self.model_utils.module_to_name[root_module]}\n"

        operation_str = "Operations:\n"
        for op in self.operation_group:
            operation_str += f"{op.module}\n"

        chain_modules = self.get_transform_chain_modules()
        chain_str = "Dimension dependency chain:\n"
        for module in chain_modules:
            chain_str += f"{self.model_utils.module_to_name[module]}\n"

        return root_str + operation_str + chain_str
