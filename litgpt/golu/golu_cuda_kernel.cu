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
void GoLUForwardCUDAKernelImpl(at::TensorIteratorBase& iter, double alpha, double beta, double gamma) {

    // AT_DISPATCH_FLOATING_TYPES_AND2 is a macro from PyTorch's CUDA backend used to handle the dispatching of
    // data types in a concise and efficient manner.
    // it.dtype() is used to dispatch the results dynamically for different data types
    // "golu_cuda" is simply a descriptive string - it doesn't refer to any function name
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "golu_forward_cuda", [&]() {
        
        // This is a type alias which allows you to define a new name (opmath_t) for an existing type
        using opmath_t = at::opmath_type<scalar_t>;
        
        // Have these as globals to avoid multiple lambda reallocation
        const opmath_t alpha_ = static_cast<opmath_t>(alpha);
        const opmath_t beta_ = static_cast<opmath_t>(beta);
        const opmath_t gamma_ = static_cast<opmath_t>(gamma);
        
        // gpu_kernel is a PyTorch utility function designed to launch a CUDA kernel that applies an element-wise
        // operation to one or more input tensors
        // GPU_LAMBDA is a lambda function that using iterator (iter) applies this method to all input values
        // scalar_t is the data type of tensor being processed
        at::native::gpu_kernel(
            iter, [alpha_, beta_, gamma_] GPU_LAMBDA(scalar_t x) -> scalar_t {

            // Explicitly convert a variable from one type to another using static_cast
            // This is an inplace operation to reduce memory usage
            x = static_cast<opmath_t>(x);

            // Compute the GoLU activation using tensor elements
            // Forward pass doesn't require corner case handling as outer_exp is 0 and 1 at -inf and +inf respectively
            return static_cast<opmath_t>(
                x * alpha_ * c10::cuda::compat::exp(-beta_ * c10::cuda::compat::exp(-gamma_ * x)));
        });
    });
}


// GoLU Backward CUDA Kernel Implementation
void GoLUBackwardCUDAKernelImpl(at::TensorIteratorBase& iter, double alpha, double beta, double gamma) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "golu_backward_cuda", [&]() {
        
        // This is a type alias which allows you to define a new name (opmath_t) for an existing type
        using opmath_t = at::opmath_type<scalar_t>;
        
        // Have these as globals to avoid multiple lambda reallocation
        const opmath_t alpha_ = static_cast<opmath_t>(alpha);
        const opmath_t beta_ = static_cast<opmath_t>(beta);
        const opmath_t gamma_ = static_cast<opmath_t>(gamma);
        
        at::native::gpu_kernel(iter, [alpha_, beta_, gamma_] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            
            // Cast the input variables to correct datatype
            // This is an inplace operation to reduce memory footprint
            dy = static_cast<opmath_t>(dy);
            x = static_cast<opmath_t>(x);
            
            // Compute common terms in grad_x to reuse them - reduces 3 exp calculations to 2
            opmath_t inner_exp = c10::cuda::compat::exp(-gamma_ * x);
            
            // Compute the respective gradients for x
            opmath_t grad_x = dy * alpha_ * c10::cuda::compat::exp(-beta_ * inner_exp) * (
                opmath_t(1) + beta_ * gamma_ * x * inner_exp);
            
            // NaN is the only value in IEEE Floating point which is not equal to itself
            grad_x = (grad_x != grad_x) ? opmath_t(0) : grad_x;

            return static_cast<opmath_t>(grad_x);
        });
    });
}


} // extern "C"
