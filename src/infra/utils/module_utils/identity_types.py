"""
Types used in the identity patching pass.
"""

from typing import Any

import torch
from torch import nn, Tensor

class IdentityFunction(torch.autograd.Function):
    """
    An identity Autograd function that just clones the input to the output. This is required
    because we need identity nodes to be recognized in the DepGraph, but that only happens if
    they create unique instances of Autograd functions.
    """
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class IdentityWithGrad(nn.Identity):
    """Uses the IdentityFunction Autograd function to create an Identity module where every
    instance has a unique Autograd function instance. An Identity module indicates that a module
    has been pruned there, so we need unique instances everywhere in the DepGraph to build logic
    around these pruned modules.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return IdentityFunction.apply(input)

class AdditiveIdentity(nn.Identity):
    """A module that returns an all-zeroes tensor with the same shape as the
    input, representing an identity element under addition. This is used in situations where an
    addition node in the DepGraph has two inputs from two different IdentityWithGrad nodes, 
    indicating that modules producing both operands have been pruned. One of the IdentityWithGrad 
    instances is replaced with this, the addition becomes an addition-with-zero, and can be 
    optimized away when compiling for inference.
    """

    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        return torch.zeros_like(input).to(self.device)

class MultiplicativeIdentity(nn.Identity):
    """Returns an all-ones tensor with the same shape as the input. A multiplicative analog to
    AdditiveIdentity.
    """

    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        return torch.ones_like(input).to(self.device)

class ConcatenativeIdentity(nn.Identity):
    """A module that returns an empty tensor, representing an identity element under concatenation.
    Used in situations where a concat node has all inputs from different IdentityWithGrad nodes,
    indicating that modules producing all operands were pruned. This could indicate that we have
    multiple copies of the same input being concatenated, and we can optimize it by only keeping
    one of them. The rest are replaced with an instance of this, and when compiling for inference,
    the compiler might optimize them away.
    """
    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        return torch.empty(0).to(self.device) # TODO: parametrize device in config
