from typing import Any

import torch
from torch import nn, Tensor

class IdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class IdentityWithGrad(nn.Identity):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return IdentityFunction.apply(input)

class AdditiveIdentity(nn.Identity):

    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        return torch.zeros_like(input).to(self.device)

class MultiplicativeIdentity(nn.Identity):

    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        return torch.ones_like(input).to(self.device)

class ConcatenativeIdentity(nn.Identity):
    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        return torch.empty(0).to(self.device) # TODO: parametrize device in config
