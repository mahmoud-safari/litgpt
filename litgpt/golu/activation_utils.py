"""
The following file contains utils for calling different activations
"""

from typing import Union
from torch.nn import Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish
from golu.golu_cuda_activation import GoLUCUDA
from golu.golu_jit_activation import GoLUJIT
from golu.fast_golu_cuda_activation import GoLUCUDAFAST
from golu.quick_golu_activation import GoLUCUDAQUICK


def get_activation_function(
    activation: str = ''
) -> Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish, GoLUJIT, GoLUCUDA]:
    """
    This is a helper function that helps to get any activation function from torch.
    
    Example Usage - 

        1. To get ReLU
            
            activation_function = get_activation_function(activation='ReLU')

    Args:
        activation (str, optional): The name of the activation to use. Defaults to ''. Choices include 'Sigmoid', \
            'Tanh', 'ReLU', 'Softplus', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'CELU', 'GELU', 'GELU_tanh', \
                'Swish', 'Mish', 'GoLUJIT'and 'GoLUCUDA'

    Raises:
        ValueError: If the activation string doesn't match any of the known activations, it raises a ValueError.

    Returns:
        Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish, GoLUJIT, GoLUCUDA]: \
            Get activation
    """
    
    if activation == 'Sigmoid':
        return Sigmoid()
    
    elif activation == 'Tanh':
        return Tanh()
    
    elif activation == 'ReLU':
        return ReLU()
    
    elif activation == 'Softplus':
        return Softplus()
    
    elif activation == 'LeakyReLU':
        return LeakyReLU()
    
    elif activation == 'PReLU':
        return PReLU()
    
    elif activation == 'ELU':
        return ELU()
    
    elif activation == 'SELU':
        return SELU()
    
    elif activation == 'CELU':
        return CELU()
    
    elif activation == 'GELU':
        return GELU()

    elif activation == 'GELU_tanh':
        return GELU(approximate='tanh')
    
    elif activation == 'Swish':
        return SiLU()
    
    elif activation == 'Mish':
        return Mish()
    
    elif activation == 'GoLUJIT':
        return GoLUJIT()
    
    elif activation == 'GoLUJIT_alpha':
        return GoLUJIT(
            requires_grad = [True, False, False], clamp_alpha=[]
        )
    
    elif activation == 'GoLUJIT_beta':
        return GoLUJIT(
            requires_grad = [False, True, False], clamp_beta=[0.02]
        )
    
    elif activation == 'GoLUJIT_gamma':
        return GoLUJIT(
            requires_grad = [False, False, True], clamp_gamma=[0.3, 1.0]
        )
    
    elif activation == 'GoLUJIT_alpha_beta':
        return GoLUJIT(
            requires_grad = [True, True, False], clamp_alpha=[], clamp_beta=[0.02]
        )
    
    elif activation == 'GoLUJIT_alpha_gamma':
        return GoLUJIT(
            requires_grad = [True, False, True], clamp_alpha=[], clamp_gamma=[0.3, 1.0]
        )
    
    elif activation == 'GoLUJIT_beta_gamma':
        return GoLUJIT(
            requires_grad = [False, True, True], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    
    elif activation == 'GoLUJIT_alpha_beta_gamma':
        return GoLUJIT(
            requires_grad = [True, True, True], clamp_alpha=[], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    
    elif activation == 'GoLUCUDA':
        return GoLUCUDA()
    
    elif activation == 'GoLUCUDAFAST':
        return GoLUCUDAFAST()
    
    elif activation == 'GoLUCUDAQUICK':
        return GoLUCUDAQUICK()
    
    else:
        raise ValueError(f"The activation named {activation} doesn't exists!")


def replace_activation_by_torch_module(
    module, old_activation, new_activation
):
    for name, child in module.named_children():
        if isinstance(child, old_activation):
            setattr(module, name, get_activation_function(activation=new_activation))
        else:
            # Recurse into child modules
            replace_activation_by_torch_module(child, old_activation, new_activation)
    return module


def replace_activation_by_name(
    module, attr_name, new_activation
):
    for name, child in module.named_children():
        if name == attr_name:
            setattr(module, name, get_activation_function(activation=new_activation))
        else:
            replace_activation_by_name(child, attr_name, new_activation)
    return module


def update_golu_parameters(
    module, new_alpha, new_beta, new_gamma
):
    for name, child in module.named_children():
        if isinstance(child, GoLUCUDA):
            child.alpha = new_alpha
            child.beta = new_beta
            child.gamma = new_gamma
        else:
            # Recurse into child modules
            update_golu_parameters(child, new_alpha, new_beta, new_gamma)
    return module
