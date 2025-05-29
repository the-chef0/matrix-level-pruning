from dataclasses import dataclass
from enum import Enum
from typing import Callable

from torch.nn import Module
import torch_pruning as tp
from torch_pruning.pruner.function import BasePruningFunc

from .model_utils import ModelUtils

class DependencyDirection(Enum):
    FORWARD = 'foward'
    BACKWARD = 'backward'
    NOT_APPLICABLE = 'not_needed'

class RootDependencyUtils:
    def __init__(self, model_utils: ModelUtils, module: Module):
        tp_module_pruner = model_utils.dep_graph.get_pruner_of_module(module)
        in_channels = tp_module_pruner.get_in_channels(module)
        out_channels = tp_module_pruner.get_out_channels(module)

        if in_channels > out_channels:
            self.direction = DependencyDirection.BACKWARD
            self.fn = tp_module_pruner.prune_in_channels
        elif in_channels < out_channels:
            self.direction = DependencyDirection.FORWARD
            self.fn = tp_module_pruner.prune_out_channels
        else:
            self.direction = DependencyDirection.NOT_APPLICABLE
            self.fn = tp_module_pruner.prune_out_channels # Placeholder

        @property
        def label(self):
            return self.direction

        @property
        def fn(self):
            return self.fn