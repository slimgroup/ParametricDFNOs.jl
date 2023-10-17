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
    fourier_x = ParDFT(Complex{T}, 10)
    fourier_y = ParDFT(Complex{T}, 10)
    fourier_t = ParDFT(T, 10)

    fft = fourier_t ⊗ fourier_y ⊗ fourier_x
    weights = ParIdentity(Complex{T},6) ⊗ ParIdentity(Complex{T},10) ⊗ ParMatrix(Complex{T}, 10, 10, "TEST")
    
    fft = distribute(fft, [1, 2, 1])
    weights = distribute(weights, [1, 2, 1])

    rng = Random.seed!(123)
    x = rand(rng, T, 10, 10, 10)
    x = x[:,(rank*5)+1:(rank*5)+5,:]

    network = fft' * weights * fft

    θ = init(network)

    function dist_norm(input)
        s = sum(input .^ 2)
    
        reduce_sum = ParReduce(eltype(input))
        diff = √(reduce_sum([s])[1])
    
        return diff
    end

    ## Sanity Check
    network(θ) * vec(x)
    println("Dist Norm: ", dist_norm(network(θ) * vec(x)))

    grads = gradient(params -> dist_norm(network(params) * vec(x)), θ)[1]
    return grads
end

function get_comp_grad()
    
    fourier_x = ParDFT(Complex{T}, 10)
    fourier_y = ParDFT(Complex{T}, 10)
    fourier_t = ParDFT(T, 10)

    fft = fourier_t ⊗ fourier_y ⊗ fourier_x
    weights = ParIdentity(Complex{T},6) ⊗ ParIdentity(Complex{T},10) ⊗ ParMatrix(Complex{T}, 10, 10, "TEST")
    
    network = fft' * weights * fft

    θ = init(network)

    rng = Random.seed!(123)
    x = rand(rng, T, 10, 10, 10)

    ## Sanity Check
    network(θ) * vec(x)
    rank == 0 && println("Serial Norm: ", norm(network(θ) * vec(x)))

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
