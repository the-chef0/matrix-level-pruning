from enum import Enum

from torch.nn import Module

from infra.utils.model_utils import ModelUtils

class DependencyDirection(Enum):
    FORWARD = 'foward'
    BACKWARD = 'backward'
    NOT_APPLICABLE = 'not_needed'

class DepGraphHelper:
    def __init__(self, model_utils: ModelUtils, module: Module):
        tp_module_pruner = model_utils.dep_graph.get_pruner_of_module(module)
        self.in_channels = tp_module_pruner.get_in_channels(module)
        self.out_channels = tp_module_pruner.get_out_channels(module)

        if self.in_channels > self.out_channels:
            self.direction = DependencyDirection.BACKWARD
            self.fn = tp_module_pruner.prune_in_channels
        elif self.in_channels < self.out_channels:
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
