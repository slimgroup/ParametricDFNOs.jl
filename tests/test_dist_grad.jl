# source $HOME/.bash_profile
# source ~/.bashrc
# mpiexecjl --project=./ -n 2 julia tests/test_dist_grad.jl

using Pkg
Pkg.activate("./")

using MPI
using ParametricOperators
using LinearAlgebra
using Test
using Zygote

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

T = Float64

network1 = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParMatrix(T, 2, 2)

x = reshape(float(1:8), 2, 2, 2)

θ1 = init(network1)
y_out = network1(θ1) * vec(x)

grads1 = gradient(params -> norm(network1(params) * vec(x)), θ1)[1]
# rank == 0 && println(grads1)

network2 = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParMatrix(T, 2, 2)
network2 = distribute(network2, [1, 2, 1])

x = reshape(float(1:8), 2, 2, 2)
x = x[:,rank+1,:]

θ = init(network2)
y_out = network2(θ) * vec(x)

# println(y_out)

function dist_norm(input)
    s = sum(input .^ 2)

    reduce_sum = ParReduce(eltype(input))
    diff = √(reduce_sum([s])[1])

    return diff
end

grads2 = gradient(params -> dist_norm(network2(params) * vec(x)), θ)[1]
# rank == 0 && println(grads2)
(rank == 0) && @assert norm(vec(collect(values(grads1))[1]) - vec(collect(values(grads2))[1])) <= 1e-10

MPI.Finalize()
