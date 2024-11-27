"""
This file contains the GoLU activation function to be directly used in PyTorch.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function

from torch.utils.cpp_extension import load

# Compile the CUDA file
quick_golu_extension = load(
    name="quick_golu_extension",
    sources=["./litgpt/golu/quick_golu_cuda_kernel.cu", "./golu/quick_golu_cuda.cpp"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math", "--extended-lambda"],
    verbose=True
)

@torch.jit.script
def _golu_forward(
    x: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float
) -> torch.Tensor:
    # Return the forward pass of GoLU Activation
    return x * alpha * torch.exp(-beta * torch.exp(-gamma * x))


class GoLUCUDAQUICKFunction(Function):

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        alpha: float,
        beta: float,
        gamma: float
    ) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.gamma = gamma
        return _golu_forward(x, alpha, beta, gamma)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None]:
        return quick_golu_extension.quick_golu_backward(
            grad_output, ctx.saved_tensors[0], ctx.alpha, ctx.beta, ctx.gamma), None, None, None


class GoLUCUDAQUICK(Module):
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0
    ) -> None:
        """
        Initialize the GoLU Activation Layer.
        
        Gompertz Linear Unit (GoLU) is an element-wise and self-gated activation function. It uses the 
        Gompertz Function as a gate. The GoLU activation and Gompertz function are defined as follows,

        GoLU(x) = x * gompertz(x), where
        gompertz(x) = alpha * exp(-beta * exp(-gamma * x))

        The respective alpha, beta and gamma values are set to 1.0 by default.
        Note - Don't set beta and gamma to negative values, else the Gompertz gate looses its classical S-shape.
        
        Example usage:
        
            1. GoLU with default setting:
                activation_function = GoLU()
            
            2. Change default initialization of parameters:
                activation_function = GoLU(alpha=0.8, beta=1.2, gamma=0.9)

        Args:
            alpha (float, optional): Controls the upper asymptote or scale of the gate. Defaults to 1.0.
            beta (float, optional): Controls the gate displacement along input-axis. Defaults to 1.0.
            gamma (float, optional): Controls the growth rate of the gate. Defaults to 1.0.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        return GoLUCUDAQUICKFunction.apply(x, self.alpha, self.beta, self.gamma)
