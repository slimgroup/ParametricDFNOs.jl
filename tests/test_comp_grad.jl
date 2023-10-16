# source $HOME/.bash_profile
# source ~/.bashrc
# mpiexecjl --project=./ -n 4 julia tests/test_comp_grad.jl

using Pkg
Pkg.activate("./")

using MPI
using ParametricOperators
using LinearAlgebra
using Test
using Zygote
using Random

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

T = Float64

function get_dist_comp_grad()
    network1 = ParIdentity(T,3) ⊗ ParIdentity(T,4) ⊗ ParIdentity(T,4) ⊗ ParMatrix(T, 5, 2, "ParMatrix_LIFTS:(1)")
    network2 = ParIdentity(T,3) ⊗ ParIdentity(T,4) ⊗ ParIdentity(T,4) ⊗ ParMatrix(T, 1, 5, "ParMatrix_LIFTS:(2)")

    network1 = distribute(network1, [1, 2, 1, 1])
    network2 = distribute(network2, [1, 2, 1, 1])

    network = network2 * network1

    rng = Random.seed!(123)
    x = rand(rng, T, 2, 4, 4, 3)
    x = x[:,(rank*2)+1:(rank*2)+2,:,:]

    θ = init(network)

    ## Sanity Check
    network(θ) * vec(x)

    function dist_norm(input)
        s = sum(input .^ 2)

        reduce_sum = ParReduce(eltype(input))
        diff = √(reduce_sum([s])[1])

        return diff
    end

    grads = gradient(params -> dist_norm(network(params) * vec(x)), θ)[1]
    return grads
end

function get_comp_grad()
    network1 = ParIdentity(T,3) ⊗ ParIdentity(T,4) ⊗ ParIdentity(T,4) ⊗ ParMatrix(T, 5, 2, "ParMatrix_LIFTS:(1)")
    network2 = ParIdentity(T,3) ⊗ ParIdentity(T,4) ⊗ ParIdentity(T,4) ⊗ ParMatrix(T, 1, 5, "ParMatrix_LIFTS:(2)")

    network = network2 * network1

    rng = Random.seed!(123)
    x = rand(rng, T, 2, 4, 4, 3)

    θ = init(network)

    ## Sanity Check
    network(θ) * vec(x)

    grads = gradient(params -> norm(network(params) * vec(x)), θ)[1]
    return grads
end

grads1 = get_comp_grad()
grads2 = get_dist_comp_grad()

# rank == 0 && println(grads1)
# rank == 0 && println(grads2)

if rank == 0
    for (k, v) in grads1
        @assert norm(vec(v) - vec(grads2[k])) <= 1e-10
    end
end

MPI.Finalize()
