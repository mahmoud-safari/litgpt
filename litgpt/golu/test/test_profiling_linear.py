import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
import csv

from golu.activation_utils import get_activation_function

# Define the 2-layer MLP model
class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activation1 = get_activation_function(activation_fn)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.activation2 = get_activation_function(activation_fn)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation2(out)
        return out

# List of activation functions to test
activations = [
    'Sigmoid',
    'Tanh',
    'ReLU',
    'Softplus',
    'LeakyReLU',
    'PReLU',
    'ELU',
    'SELU',
    'CELU',
    'GELU',
    'GELU_tanh',
    'Swish',
    'Mish',
    'GoLUJIT',
    'GoLUCUDA',
    'GoLUCUDAFAST',
    'GoLUCUDAQUICK',
]

# List of datatypes to test
datatypes = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]

# Prepare to collect results
results = []

# Model and data parameters
input_size = 512
hidden_size = 512
output_size = 10
batch_size = 64

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CUDA context with a dummy operation if using CUDA
if device.type == 'cuda':
    _ = torch.randn(1).to(device)

# Loop over different datatypes and activations
for dtype in datatypes:
    dtype_name = 'float32' if dtype == torch.float32 else 'float16' if dtype == torch.float16 else 'bfloat16'
    
    for activation_name in activations:
        print(f"Profiling activation: {activation_name} with datatype: {dtype_name}")
        model = TwoLayerMLP(input_size, hidden_size, output_size, activation_name).to(device).to(dtype)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss().to(device).to(dtype)

        # Generate random input and target tensors
        inputs = torch.randn(batch_size, input_size).to(device).to(dtype)
        targets = torch.randn(batch_size, output_size).to(device).to(dtype)

        # Warm-up step
        for _ in range(10):
            with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):  # Enable autocast for float16
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Profile the forward pass
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as forward_prof:
            with record_function("forward_pass"):
                with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):  # Enable autocast for float16
                    outputs = model(inputs)

        torch.cuda.synchronize()

        # Profile the backward pass
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as backward_prof:
            with record_function("backward_pass"):
                with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):  # Enable autocast for float16
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        torch.cuda.synchronize()

        # Get the flops
        forward_flops = sum(
            [event.flops for event in forward_prof.key_averages() if hasattr(event, 'flops') and event.flops is not None]
        )
        backward_flops = sum(
            [event.flops for event in backward_prof.key_averages() if hasattr(event, 'flops') and event.flops is not None]
        )

        # Get the CUDA and CPU times for forward pass
        forward_cuda_time = sum(
            [event.cuda_time for event in forward_prof.key_averages() if event.cuda_time is not None]
        )
        forward_cpu_time = sum(
            [event.cpu_time for event in forward_prof.key_averages() if event.cpu_time is not None]
        )

        # Get the CUDA and CPU times for backward pass
        backward_cuda_time = sum(
            [event.cuda_time for event in backward_prof.key_averages() if event.cuda_time is not None]
        )
        backward_cpu_time = sum(
            [event.cpu_time for event in backward_prof.key_averages() if event.cpu_time is not None]
        )

        # Collect results
        results.append({
            'Activation': activation_name,
            'Datatype': dtype_name,
            'Forward Flops': forward_flops,
            'Backward Flops': backward_flops,
            'Forward CUDA Time (us)': forward_cuda_time,
            'Backward CUDA Time (us)': backward_cuda_time,
            'Forward CPU Time (us)': forward_cpu_time,
            'Backward CPU Time (us)': backward_cpu_time
        })

# Output the results to a CSV file
csv_filename = './litgpt/golu/test/activation_profiling_results_linear.csv'
fieldnames = ['Activation', 'Datatype', 'Forward Flops', 'Backward Flops', 'Forward CUDA Time (us)', 'Backward CUDA Time (us)', 'Forward CPU Time (us)', 'Backward CPU Time (us)']
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nProfiling complete. Results saved to '{csv_filename}'.")
