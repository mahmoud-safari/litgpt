import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
import csv

from golu.activation_utils import get_activation_function

# Define a residual block with convolution layers
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation_fn):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.activation1 = get_activation_function(activation_fn)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = get_activation_function(activation_fn)

        # Ensure the residual path matches the output size if necessary
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Adding the residual
        out = self.activation2(out)
        return out

# Define a CNN with two residual blocks
class TwoBlockResNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation_fn):
        super(TwoBlockResNet, self).__init__()
        # Two residual blocks stacked
        self.block1 = ResidualBlock(in_channels, hidden_channels, hidden_channels, activation_fn)
        self.block2 = ResidualBlock(hidden_channels, hidden_channels, out_channels, activation_fn)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
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
in_channels = 3   # Example for RGB images
hidden_channels = 64
out_channels = 10
image_size = 64   # Example image size (64x64)
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
        model = TwoBlockResNet(in_channels, hidden_channels, out_channels, activation_name).to(device).to(dtype)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss().to(device).to(dtype)

        # Generate random input and target tensors (batch_size x in_channels x image_size x image_size)
        inputs = torch.randn(batch_size, in_channels, image_size, image_size).to(device).to(dtype)
        targets = torch.randn(batch_size, out_channels, image_size, image_size).to(device).to(dtype)

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
csv_filename = './litgpt/golu/test/activation_profiling_results_resnet.csv'
fieldnames = ['Activation', 'Datatype', 'Forward Flops', 'Backward Flops', 'Forward CUDA Time (us)', 'Backward CUDA Time (us)', 'Forward CPU Time (us)', 'Backward CPU Time (us)']
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nProfiling complete. Results saved to '{csv_filename}'.")
