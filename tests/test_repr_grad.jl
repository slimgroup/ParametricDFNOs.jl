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

function get_serial_grad()
    network1 = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParMatrix(T, 2, 2, "TEST")

    x = reshape(float(1:8), 2, 2, 2)
    
    θ1 = init(network1)

    ## Sanity Check
    network1(θ1) * vec(x)
    
    grads1 = gradient(params -> norm(network1(params) * vec(x)), θ1)[1]
    return grads1
end

function get_dist_grad()
    network2 = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParMatrix(T, 2, 2, "TEST")
    network2 = distribute(network2, [2, 1, 1])
    
    x = reshape(float(1:8), 2, 2, 2)
    x = x[rank+1,:,:]
    
    θ = init(network2)

    ## Sanity Check
    network2(θ) * vec(x)

    function dist_norm(input)
        s = sum(input .^ 2)
    
        reduce_sum = ParReduce(eltype(input))
        diff = √(reduce_sum([s])[1])
    
        return diff
    end
    
    grads2 = gradient(params -> dist_norm(network2(params) * vec(x)), θ)[1]
    return grads2
end

grads1 = get_serial_grad()
grads2 = get_dist_grad()

# rank == 0 && println(grads1)
# rank == 0 && println(grads2)

if rank == 0
    for (k, v) in grads1
        @assert norm(vec(v) - vec(grads2[k])) <= 1e-10
    end
end


MPI.Finalize()
