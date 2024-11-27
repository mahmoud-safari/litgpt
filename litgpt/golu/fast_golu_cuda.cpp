// References - GPT-o1-preview

#include <torch/extension.h>
#include <ATen/TensorIterator.h>


// Declare the extern "C" functions
// Note - here the alpha, beta and gamma are passed once and stored as global constants in the below functions
// This further reduces creating the same variables again and again in the lambda function
extern "C" void FastGoLUForwardCUDAKernelImpl(at::TensorIteratorBase& iter);
extern "C" void FastGoLUBackwardCUDAKernelImpl(at::TensorIteratorBase& iter);


// Python interface for the forward function
at::Tensor fast_golu_forward(at::Tensor x) {
    // Create an output tensor, same size as input x
    auto output = at::empty_like(x);
    // Configure the iterator with output and inputs
    auto iter = at::TensorIteratorConfig()
                    .add_output(output)  // Separate output tensor
                    .add_input(x)
                    .build();
    FastGoLUForwardCUDAKernelImpl(iter);  // Direct call to the CUDA kernel function
    return output;  // Return the output tensor
}

// Python interface for the backward function
at::Tensor fast_golu_backward(at::Tensor grad_output, at::Tensor x) {
    // Create tensors to hold gradients
    auto grad_x = at::empty_like(x);
    // Configure the iterator with gradients and inputs
    auto iter = at::TensorIteratorConfig()
                    .add_output(grad_x)  // Gradient with respect to x
                    .add_input(grad_output)  // Incoming gradient from the next layer
                    .add_input(x)  // Input tensor x
                    .build();
    FastGoLUBackwardCUDAKernelImpl(iter);  // Direct call to the CUDA kernel function
    return grad_x;  // Return the gradients
}

// Bindings to make the functions available in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_golu_forward", &fast_golu_forward, "Fast GoLU forward (CUDA)");
    m.def("fast_golu_backward", &fast_golu_backward, "Fast GoLU backward (CUDA)");
}
