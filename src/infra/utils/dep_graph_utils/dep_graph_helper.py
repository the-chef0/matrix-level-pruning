from enum import Enum

from torch.nn import Module

from infra.utils.model_utils import ModelUtils

class DependencyDirection(Enum):
    """For given transform module, this is an abstraction that captures in which direction to search
    in the DepGraph for different types of dependencies and coupled structures.
    """
    FORWARD = 'foward'
    BACKWARD = 'backward'
    NOT_APPLICABLE = 'not_needed'

class DepGraphHelper:
    def __init__(self, model_utils: ModelUtils, module: Module):
        tp_module_pruner = model_utils.dep_graph.get_pruner_of_module(module)
        self.in_channels = tp_module_pruner.get_in_channels(module)
        self.out_channels = tp_module_pruner.get_out_channels(module)

        if self.in_channels > self.out_channels:
            # The transform maps from higher dimensions to lower dimensions -> if we want to prune it
            # out, we need to look for dimension dependencies backwards.
            self.direction = DependencyDirection.BACKWARD
            self.fn = tp_module_pruner.prune_in_channels
        elif self.in_channels < self.out_channels:
            # Analogous logic here
            self.direction = DependencyDirection.FORWARD
            self.fn = tp_module_pruner.prune_out_channels
        else:
            # If the transform does not change the dimension, it can be removed without dimension
            # conflicts, so we don't need to search for dimension dependencies at all
            self.direction = DependencyDirection.NOT_APPLICABLE
            self.fn = tp_module_pruner.prune_out_channels # Placeholder

        @property
        def label(self):
            return self.direction

        @property
        def fn(self):
            return self.fn
