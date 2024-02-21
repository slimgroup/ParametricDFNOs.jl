using Pkg
Pkg.activate("./")

using Flux
using CUDA

# Function to convert bytes to megabytes
bytes_to_MB(x) = x / 1024^2

if CUDA.has_cuda_gpu()
    # Move computations to the GPU
    device = gpu

    # Create weight matrix and input tensor
    W = randn(10000, 65000) |> device
    input_tensor = randn(65000) |> device

    # Define the loss function
    loss(x) = sum(W * x)

    for i in 1:2
        println("\nIteration $i")

        # Memory usage before the operation
        CUDA.memory_allocated()

        # Forward pass and loss
        L = loss(input_tensor)

        # Backward pass
        @time grads = gradient(() -> loss(input_tensor), Flux.params(W))

        # Memory usage after the operation
        CUDA.memory_allocated()
    end
else
    println("CUDA-enabled GPU is not available.")
end
