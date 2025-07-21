from typing import Callable

import numpy as np
from torch.nn import Module
from torch_pruning.pruner.importance import GroupMagnitudeImportance, GroupTaylorImportance

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
        super().__init__(model_utils)
        self.cfg = cfg
        self.model_utils = model_utils
        # Helps keep track of where to look for dimension dependencies based on the in/out channel
        # relationship, and what type of node it is in the DepGraph (linear/conv).
        self.dg_helper = DepGraphHelper(model_utils, root_module)
        self.root_dim_low, self.root_dim_high = self.get_module_dims()
        self.dim_idxs = [i for i in range(self.root_dim_high)]

        # Obtain a DepGraph Group instance that models which parameters have dimension
        # dependencies on the root module, i.e. tells us which matrices need to be width-pruned
        # to reconcile dimensions after replacing the root with an identity.
        self.param_subtree = model_utils.dep_graph.get_pruning_group(
            module=root_module,
            pruning_fn=self.dg_helper.fn,
            idxs=self.dim_idxs
        )
        # Treat the root as a separate object because it will treated differently than the
        # dependencies
        self.param_subtree_root = self.param_subtree[:1]

        # If the root has a square matrix, we don't need a "direction" that the dependencies are in.
        # If the root e.g. maps from a higher to lower dimensional space and we prune it, we need to
        # adjust dimensions in the forward directions because the following layers were expecting 
        # the smaller input.
        if self.dg_helper.direction != DependencyDirection.NOT_APPLICABLE:
            # The remainder of the Group object contains the dependent layers, and we isolate
            # those as the dependency part of the parameter subtree.
            # Formulate as a list of singleton Group objects if channel dimensions become
            # a problem.
            self.param_subtree_deps = self.param_subtree[1:]
        else:
            self.param_subtree_deps = None
            
        root_module_node = model_utils.dep_graph.module2node[root_module]
        self.op_subtree = list(get_op_subtree(cfg, root_module_node))
        # Unlike in attention_pruning_tree, the group_reduction arg in GroupMagnitudeImportance
        # is left as the default 'mean', which means it calculates channel importances separately
        # for each dep in param_subtree_deps and calculates the mean. This assumes that each dep
        # has the same number of channels to prune. This might become a problem if there are
        # ever dependencies via a concat or other shape changing node. If this happens, see the
        # comment above the param_subtree_deps definition.
        self.importance_fn = GroupTaylorImportance(normalizer=None)
        self.model_utils = model_utils
        
        self.importance = None # TODO: The logic surrounding this should not be needed anymore?
        self.dep_importance_ranking = None

    def get_module_dims(self):
        """Differentiates between and extracts the lower channel dimension and the higher channel
        dimension of the transform.

        Returns:
            int: The lower channel dimension.
            int: The higher channel dimension.
        """
        root_dim_low = np.min([self.dg_helper.in_channels, self.dg_helper.out_channels])
        root_dim_high = np.max([self.dg_helper.in_channels, self.dg_helper.out_channels])
        return root_dim_low, root_dim_high

    def get_dep_importance_ranking(self):
        """If the tree contains dimension dependencies, ranks the selected channels by importance.

        Returns:
            list[float, int]:A list of tuples sorted in ascending order on the first element,
                importance, and the second element is the index of the channel.
        """
        if self.dep_importance_ranking is None:
            importance = self.importance_fn(self.param_subtree_deps).cpu().numpy()
            all_channel_idxs = [i for i in range(self.root_dim_high)]
            self.dep_importance_ranking = sorted(zip(importance, all_channel_idxs))
        return self.dep_importance_ranking

    def get_dep_importance(self):
        if self.param_subtree_deps is not None:
            dep_importance_ranking = self.get_dep_importance_ranking()
            importances_ranked = [importance for (importance, idx) in dep_importance_ranking]
            return np.mean(importances_ranked[:self.root_dim_low])
        else:
            return 0

    def get_root_importance(self):
        importance = self.importance_fn(self.param_subtree_root).cpu().numpy()
        return np.mean(importance)

    def get_op_importance(self):
        return 0

    def get_importance(self):
        if self.importance is None:
            self.importance = np.mean([
                self.get_root_importance(),
                self.get_op_importance(),
                self.get_dep_importance()
            ])
            
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
        print(f"Pruning {self}")
        # If there are dimension dependencies, we first use the built in DepGraph width pruning
        # functionality to get the dimensions in order.
        if self.param_subtree_deps:
            importance_ranking = self.get_dep_importance_ranking()
            num_channels_to_prune = self.root_dim_high - self.root_dim_low
            idxs_to_prune = [idx for (_, idx) in importance_ranking[:num_channels_to_prune]]
            self.param_subtree.prune(idxs_to_prune)

        # With the dimensions adjusted, it is now safe to prune the root by replacing it with an
        # identity module.
        root_module = self.get_root_module()
        root_module_name = self.model_utils.module_to_name[root_module]
        self.model_utils.replace_module_by_name(root_module_name, IdentityWithGrad())
        
        # Operations coupled to the root (operation subtree) can also be pruned, if any.
        operation_modules = [node.module for node in self.op_subtree]
        for op_module in operation_modules:
            op_module_name = self.model_utils.module_to_name[op_module]
            self.model_utils.replace_module_by_name(op_module_name, IdentityWithGrad())

        self.call_post_prune_listeners()

    def __str__(self):
        root_module = self.get_root_module()
        root_str = f"Root:\n{self.model_utils.module_to_name[root_module]}\n"

        operation_str = "Operations:\n"
        for op in self.op_subtree:
            operation_str += f"{op.module}\n"

        chain_modules = self.get_dep_modules()
        chain_str = "Dimension dependencies:\n"
        for module in chain_modules:
            chain_str += f"{self.model_utils.module_to_name[module]}\n"

        return root_str + operation_str + chain_str
