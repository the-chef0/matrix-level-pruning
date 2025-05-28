from dataclasses import dataclass
from typing import Callable

import torch_pruning as tp

@dataclass
class DependencyDirection:
    backward: Callable = tp.prune_linear_in_channels
    forward: Callable = tp.prune_linear_out_channels
    not_needed: Callable = None
