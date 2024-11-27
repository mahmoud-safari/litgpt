// References - https://github.com/pytorch/pytorch
// Check https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda for more details

#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/native/Activation.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>


extern "C" {

// GoLU CUDA Kernel Implementation
// Individual x elements are passed using a lambda function internally - so x is not a tensor here
// Void function doesn't return any value
// TensorIteratorBase - Reference to an object of type TensorIteratorBase
void FastGoLUForwardCUDAKernelImpl(at::TensorIteratorBase& iter) {

    // AT_DISPATCH_FLOATING_TYPES_AND2 is a macro from PyTorch's CUDA backend used to handle the dispatching of
    // data types in a concise and efficient manner.
    // it.dtype() is used to dispatch the results dynamically for different data types
    // "fast_golu_forward_cuda" is simply a descriptive string - it doesn't refer to any function name
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "fast_golu_forward_cuda", [&]() {
        
        // gpu_kernel is a PyTorch utility function designed to launch a CUDA kernel that applies an element-wise
        // operation to one or more input tensors
        // GPU_LAMBDA is a lambda function that using iterator (iter) applies this method to all input values
        // scalar_t is the data type of tensor being processed
        at::native::gpu_kernel(
            iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
            // Use fast math versions of exponential functions based on the data type
            #if defined(__CUDA_ARCH__)
            //x = static_cast<float>(x);
            return static_cast<scalar_t>(x * __expf(-__expf(-x)));
            #endif
        });
    });
}


// GoLU Backward CUDA Kernel Implementation
void FastGoLUBackwardCUDAKernelImpl(at::TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "fast_golu_backward_cuda", [&]() {
        at::native::gpu_kernel(
            iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            #if defined(__CUDA_ARCH__)
            //x = static_cast<float>(x);
            float inner_exp = __expf(-x);
            float grad_x = dy * __expf(-inner_exp) * (1.0f + x * inner_exp);
            grad_x = (grad_x != grad_x) ? 0.0f : grad_x;
            return grad_x;
            #endif
        });
    });
}


} // extern "C"
