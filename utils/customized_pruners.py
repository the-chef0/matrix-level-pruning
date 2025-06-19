from typing import Sequence

import torch
from torch import nn
from torch_pruning.pruner.function import BasePruningFunc

class OperationPruner(BasePruningFunc):

    def prune_out_channels(self, layer, idxs):
        return layer
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer):
        return None
    
    def get_in_channels(self, layer):
        return None

class RMSNormPruner(BasePruningFunc):

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