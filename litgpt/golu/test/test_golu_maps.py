"The following file prints the output and gradients of GoLU Activation"

import torch
from golu.golu_cuda_activation import GoLUCUDA
from golu.fast_golu_cuda_activation import GoLUCUDAFAST
from golu.quick_golu_activation import GoLUCUDAQUICK

x = torch.linspace(-10, 10, 50, requires_grad=True, dtype=torch.float32, device='cuda')
activation_function = GoLUCUDA().to(dtype=torch.float32, device='cuda')
y = activation_function(x)
loss = y.sum()
loss.backward()
print('\n\nGoLUCUDA #################################################################')
print(f'\nInput - \n{x}\n\n')
print(f'Activation Map - \n{y}\n\n')
print(f'Gradients - \n{x.grad}')

x_fast = torch.linspace(-10, 10, 50, requires_grad=True, dtype=torch.float32, device='cuda')
activation_function = GoLUCUDAFAST().to(dtype=torch.float32, device='cuda')
y_fast = activation_function(x_fast)
loss_fast = y_fast.sum()
loss_fast.backward()
print('\n\nGoLUCUDAFAST #############################################################')
print(f'\nInput - \n{x_fast}\n\n')
print(f'Activation Map - \n{y_fast}\n\n')
print(f'Gradients - \n{x_fast.grad}')

x_quick = torch.linspace(-10, 10, 50, requires_grad=True, dtype=torch.float32, device='cuda')
activation_function = GoLUCUDAQUICK().to(dtype=torch.float32, device='cuda')
y_quick = activation_function(x_quick)
loss_quick = y_quick.sum()
loss_quick.backward()
print('\n\nGoLUCUDAQUICK ############################################################')
print(f'\nInput - \n{x_quick}\n\n')
print(f'Activation Map - \n{y_quick}\n\n')
print(f'Gradients - \n{x_quick.grad}')

# Check if the activation maps and gradients are close enough
assert torch.allclose(y, y_fast, atol=1e-6), "Activation outputs are not close enough!"
assert torch.allclose(x.grad, x_fast.grad, atol=1e-6), "Gradients are not close enough!"

assert torch.allclose(y, y_quick, atol=1e-6), "Activation outputs are not close enough!"
assert torch.allclose(x.grad, x_quick.grad, atol=1e-6), "Gradients are not close enough!"
