using Pkg
Pkg.activate("./")

using Flux
using CUDA

if CUDA.has_cuda_gpu()
    # Move computations to the GPU
    device = gpu

    # Create weight matrix and input tensor
    W = randn(100, 100) |> device
    input_tensor = randn(100) |> device

    # Define the loss function
    loss(x) = sum(W * x)

    for i in 1:2
        println("\nIteration $i")

        # Memory usage before the operation
        println("Memory allocated before operation: ", CUDA.memory_allocated())
        println("Memory reserved before operation: ", CUDA.memory_reserved())

        # Forward pass and loss
        L = loss(input_tensor)

        # Backward pass
        grads = gradient(() -> loss(input_tensor), params(W))

        # Memory usage after the operation
        println("Memory allocated after operation: ", CUDA.memory_allocated())
        println("Memory reserved after operation: ", CUDA.memory_reserved())
    end
else
    println("CUDA-enabled GPU is not available.")
end
