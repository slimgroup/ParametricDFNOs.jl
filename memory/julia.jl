using Pkg
Pkg.activate("./")

using Flux
using CUDA

# Function to convert bytes to megabytes
bytes_to_MB(x) = x / 1024^2
x, y = parse.(Int, ARGS[1:2])

W = randn(x, y) |> gpu
input_tensor = randn(y) |> gpu
# Define the loss function

loss(x) = sum(W * x)
for i in 1:2
    println("\nIteration $i")
    # Memory usage before the operation
    CUDA.memory_status()
    # Forward pass and loss
    L = loss(input_tensor)
    # Backward pass
    @time grads = gradient(() -> loss(input_tensor), Flux.params(W))
    # Memory usage after the operation
    CUDA.memory_status()
end
