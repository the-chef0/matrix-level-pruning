from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from torch.nn import Identity, Linear, Module
import torch_pruning as tp
from torch_pruning.dependency import Group
from torch_pruning.pruner.importance import GroupMagnitudeImportance

from utils.dependency_utils import RootDependencyUtils, DependencyDirection
from utils.functional import is_transform_type
from utils.model_utils import ModelUtils
from utils.operation_group_utils import get_operation_group

def replace_module_by_name(model_utils: ModelUtils, module_name, new_module):
    # Split the module name into parts
    parts = module_name.split('.')
    
    # Get the parent module (everything except the last part)
    parent = model_utils.model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    # Replace the module
    setattr(parent, parts[-1], new_module)

def get_operation_modules(operation_group: list):
    return [node.module for node in operation_group]

class PruningGroup(ABC):

    @abstractmethod
    def get_importance(self):
        pass

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

class AttentionPruningGroup(ABC):
    def get_singleton_group(self, model_utils: ModelUtils, proj: Linear, idxs: list, pruning_fn):
        full_group = model_utils.dep_graph.get_pruning_group(proj, pruning_fn, idxs)
        singleton_group = Group()
        setattr(singleton_group, '_group', full_group[:1])
        setattr(singleton_group, '_DG', model_utils.dep_graph)
        return singleton_group

    def __init__(self, model_utils: ModelUtils, attention_module: Module, qo_idxs: list, kv_idxs: list, dims_per_head: int):
        self.model_utils = model_utils
        self.module = attention_module
        self.importance_fn = GroupMagnitudeImportance(normalizer=None, group_reduction=None)
        self.qo_idxs_flat = np.array(qo_idxs).flatten()
        self.qo_idxs = qo_idxs
        self.kv_idxs = kv_idxs

        self.q_singleton = self.get_singleton_group(model_utils, self.module.q_proj, self.qo_idxs_flat, tp.prune_linear_out_channels)
        self.k_singleton = self.get_singleton_group(model_utils, self.module.k_proj, self.kv_idxs, tp.prune_linear_out_channels)
        self.v_singleton = self.get_singleton_group(model_utils, self.module.v_proj, self.kv_idxs, tp.prune_linear_out_channels)
        self.o_singleton = self.get_singleton_group(model_utils, self.module.o_proj, self.qo_idxs_flat, tp.prune_linear_in_channels)

        self.operation_group = None
        # When there is only one head left, we can start pruning operations
        if self.module.k_proj.out_features == dims_per_head:
            self.operation_group = list(get_operation_group(model_utils, self.module.o_proj))

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
            operation_modules = get_operation_modules(self.operation_group)
            for op_module in operation_modules:
                op_module_name = self.model_utils.module_to_name[op_module]
                replace_module_by_name(self.model_utils, op_module_name, Identity())
            
            attention_module_name = self.model_utils.module_to_name[self.module]
            replace_module_by_name(self.model_utils, attention_module_name, Identity())

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

class AttentionPruningGroupGenerator:
    def __init__(self, attention_module: Module):
        self.module = attention_module
        self.dims_per_head = self.module.config.hidden_size // self.module.config.num_attention_heads
        assert self.module.k_proj.out_features == self.module.v_proj.out_features
        self.num_kv_heads = self.module.k_proj.out_features // self.dims_per_head
        num_q_heads = self.module.q_proj.out_features // self.dims_per_head
        self.num_kv_groups = num_q_heads // self.num_kv_heads

    def get_groups(self, model_utils: ModelUtils):
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

            yield AttentionPruningGroup(
                model_utils=model_utils,
                attention_module=self.module,
                qo_idxs=qo_idxs,
                kv_idxs=kv_idxs,
                dims_per_head=self.dims_per_head
            )

class TransformPruningGroup(ABC):
    def __init__(self, model_utils: ModelUtils, root_module: Module):
        self.model_utils = model_utils
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

    def prune(self):
        # Prune transform group dimension dependency chain, only if the root is a
        # rectangular matrix and has a dimension dependency chain
        if self.transform_group_chain:
            importance_ranking = self.get_transform_chain_importance_ranking()
            num_channels_to_prune = self.root_dim_high - self.root_dim_low
            idxs_to_prune = [idx for (importance, idx) in importance_ranking[:num_channels_to_prune]]
            self.transform_group.prune(idxs_to_prune)

        # Prune transform group root
        root_module = self.get_transform_root_module()
        root_module_name = self.model_utils.module_to_name[root_module]
        replace_module_by_name(self.model_utils, root_module_name, Identity())

        # Prune operations
        operation_modules = get_operation_modules(self.operation_group)
        for op_module in operation_modules:
            op_module_name = self.model_utils.module_to_name[op_module]
            replace_module_by_name(self.model_utils, op_module_name, Identity())

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
