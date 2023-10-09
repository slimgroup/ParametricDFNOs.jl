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

network = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParMatrix(T, 2, 2)
network = distribute(network, [1, 2, 1])

x = reshape(float(1:8), 2, 2, 2)
x = vec(x[:,rank+1,:])

θ = init(network)
y_out = network(θ) * x

rank == 0 & gradient(params -> norm(network(params) * x), θ)[1]

MPI.Finalize()
