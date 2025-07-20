from typing import Sequence

import torch
from torch import nn
from torch_pruning.pruner.function import BasePruningFunc

class OperationPruner(BasePruningFunc):
    """A placeholder pruner class for DepGraph. The DepGraph only includes nodes for modules
    that have a pruner associated with it. This can be used to include arbitrary non-parametric
    module types (e.g. activations) in the DepGraph. See DEP_GRAPH_ARGS the config.
    """

    def prune_out_channels(self, layer, idxs):
        return layer
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer):
        return None
    
    def get_in_channels(self, layer):
        return None

class RMSNormPruner(BasePruningFunc):
    """RMSNorm modules need to be included in the DepGraph, but they also need width pruning
    functionality because they are parametrized.
    """

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.weight.size(0))) - set(idxs))
        keep_idxs.sort()
        
        layer.weight = torch.nn.Parameter(
            layer.weight[keep_idxs]
        )
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.size(0)

    def get_in_channels(self, layer):
        return layer.weight.size(0)
