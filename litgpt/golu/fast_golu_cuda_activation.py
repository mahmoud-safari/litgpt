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
fast_golu_extension = load(
    name="fast_golu_extension",
    sources=["./litgpt/golu/fast_golu_cuda_kernel.cu", "./golu/fast_golu_cuda.cpp"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math", "--extended-lambda"],
    verbose=True
)


class GoLUCUDAFASTFunction(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return fast_golu_extension.fast_golu_forward(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return fast_golu_extension.fast_golu_backward(grad_output, ctx.saved_tensors[0])


class GoLUCUDAFAST(Module):
    
    def __init__(self) -> None:
        """
        Initialize the GoLU Activation Layer.
        
        Gompertz Linear Unit (GoLU) is an element-wise and self-gated activation function. It uses the 
        Gompertz Function as a gate. The GoLU activation and Gompertz function are defined as follows,

        GoLU(x) = x * gompertz(x), where
        gompertz(x) = exp(-exp(-x))
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return GoLUCUDAFASTFunction.apply(x)
