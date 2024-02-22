using Pkg
Pkg.activate("./")

using MPI, Zygote, DrWatson, CUDA, Flux, LinearAlgebra

nx, ny, nz, nt = parse.(Int, ARGS[1:4])
T = Float32

x = rand(T, 5, nx*ny*nz*nt)
y = rand(T, 1, nx*ny*nz*nt)

weights = Dict(
    :w1 => rand(T, 20, 5),
    :w2 => rand(T, 128, 20),
    :w3 => rand(T, 1, 128)
)

x = x |> gpu
y = y |> gpu
weights = map(gpu, weights)

function forward(weights, x)
    w1, w2, w3 = weights[:w1], weights[:w2], weights[:w3]
    return norm(relu.(w3 * relu.(w2 * (w1 * x))) - y)
end

gradient_weights = Zygote.gradient(weights -> forward(weights, x), weights)
