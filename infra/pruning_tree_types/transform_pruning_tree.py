import numpy as np
from torch.nn import Module
from torch_pruning.pruner.importance import GroupMagnitudeImportance

from config.config_protocol import ConfigProtocol
from infra.pruning_tree_types.pruning_tree import PruningTree
from infra.utils.dep_graph_utils.dep_graph_helper import DepGraphHelper, DependencyDirection
from infra.utils.dep_graph_utils.dep_graph_search_utils import get_op_subtree
from infra.utils.module_utils.pruning_tree_collection_utils import is_transform_type
from infra.utils.module_utils.identity_types import IdentityWithGrad
from infra.utils.model_utils import ModelUtils

class TransformPruningTree(PruningTree):
    """ Defines a transform pruning tree, i.e. a pruning tree rooted at either
    a Linear module or a Conv module that meets the criteria defined in config
    and pruning_tree_collection_utils.

    First, it obtains a Torch-Pruning Group object rooted at the root module.
    If the root module contains a square parameter matrix, the module can be removed
    (i.e. be replaced by an identity module) without breaking the dimensions in
    the rest of the model.
    If the root module contains a rectangular parameter matrix, we need to
    know which dependent parameter matrices need to have their input/output
    dimensions adjusted via width pruning to resolve the dimension conflict.
    To do this, we create a Torch-Pruning Group object rooted at the group module
    which contains information on how the in/out channels on the root module
    depend on the in/out channels of the dependent modules.
    This object defines the parameter subtree, from which we extract the root
    and the dependencies.
    We also create an operation subtree object which contains activation functions,
    normalization functions, etc. that are coupled only to the root module and can 
    be removed along with it.
    """
    def __init__(self, cfg: ConfigProtocol, model_utils: ModelUtils, root_module: Module):
        self.cfg = cfg
        self.model_utils = model_utils
        self.dg_helper = DepGraphHelper(model_utils, root_module)
        self.root_dim_low, self.root_dim_high = self.get_module_dims(root_module)
        self.dim_idxs = [i for i in range(self.root_dim_high)]

        self.param_subtree = model_utils.dep_graph.get_pruning_group(
            module=root_module,
            pruning_fn=self.dg_helper.fn,
            idxs=self.dim_idxs
        )
        self.param_subtree_root = self.param_subtree[:1]

        if self.dg_helper.direction != DependencyDirection.NOT_APPLICABLE:
            self.param_subtree_deps = self.param_subtree[1:]
        else:
            self.param_subtree_deps = None
            
        root_module_node = model_utils.dep_graph.module2node[root_module]
        self.op_subtree = list(get_op_subtree(cfg, root_module_node)) # TODO: rewrite this to get nodes and convert to modules using some fn in module_utils?
        self.importance_fn = GroupMagnitudeImportance(normalizer=None)
        self.model_utils = model_utils
        
        self.importance = None
        self.dep_importance_ranking = None

    def get_module_dims(self, module: Module):
        root_dim_low = np.min([self.dg_helper.in_channels, self.dg_helper.out_channels])
        root_dim_high = np.max([self.dg_helper.in_channels, self.dg_helper.out_channels])
        return root_dim_low, root_dim_high

    def get_dep_importance_ranking(self):
        if self.dep_importance_ranking is None:
            importance = self.importance_fn(self.param_subtree_deps).cpu().numpy()
            all_channel_idxs = [i for i in range(self.root_dim_high)]
            self.dep_importance_ranking = sorted(zip(importance, all_channel_idxs))
        return self.dep_importance_ranking

    def get_dep_importance(self):
        if self.param_subtree_deps is not None:
            dep_importance_ranking = self.get_dep_importance_ranking()
            importances_ranked = [importance for (importance, idx) in dep_importance_ranking]
            return np.sum(importances_ranked[:self.root_dim_low])
        else:
            return 0

    def get_root_importance(self):
        importance = self.importance_fn(self.param_subtree_root).cpu().numpy()
        return np.sum(importance)

    def get_op_importance(self):
        return 0

    def get_importance(self):
        if self.importance is None:
            self.importance = self.get_root_importance() + \
                self.get_op_importance() + \
                self.get_dep_importance()
            
        return self.importance

    def get_root_module(self):
        return self.param_subtree_root[0].dep.source.module
    
    def get_dep_modules(self):
        modules = []
        if self.param_subtree_deps:
            for item in self.param_subtree_deps:
                item_module = item.dep.target.module
                if is_transform_type(self.cfg, type(item_module)):
                    modules.append(item_module)
        return modules

    def prune(self):
        if self.param_subtree_deps:
            importance_ranking = self.get_dep_importance_ranking()
            num_channels_to_prune = self.root_dim_high - self.root_dim_low
            idxs_to_prune = [idx for (importance, idx) in importance_ranking[:num_channels_to_prune]]
            self.param_subtree.prune(idxs_to_prune)

        root_module = self.get_root_module()
        root_module_name = self.model_utils.module_to_name[root_module]
        self.model_utils.replace_module_by_name(root_module_name, IdentityWithGrad())

        operation_modules = [node.module for node in self.op_subtree]
        for op_module in operation_modules:
            op_module_name = self.model_utils.module_to_name[op_module]
            self.model_utils.replace_module_by_name(op_module_name, IdentityWithGrad())

    def __str__(self):
        root_module = self.get_root_module()
        root_str = f"Root:\n{self.model_utils.module_to_name[root_module]}\n"

        operation_str = "Operations:\n"
        for op in self.op_subtree:
            operation_str += f"{op.module}\n"

        chain_modules = self.get_dep_modules()
        chain_str = "Dimension dependency chain:\n"
        for module in chain_modules:
            chain_str += f"{self.model_utils.module_to_name[module]}\n"

        return root_str + operation_str + chain_str
