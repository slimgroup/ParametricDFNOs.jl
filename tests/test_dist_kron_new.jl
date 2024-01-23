# source $HOME/.bash_profile
# source ~/.bashrc
# plot_operator_graph(network)
# mpiexecjl --project=./ -n 2 julia tests/test_dist_kron.jl

using Pkg
Pkg.activate("./")

using MPI
using ParametricOperators
using LinearAlgebra
using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

T = Float64
network = ParIdentity(T, 2) ⊗ ParMatrix(T, 2, 2)

x = reshape(float(1:4), 2, 2)
θ = init(network)

y_true = network(θ) * vec(x)
y_true = reshape(y_true, 2, 2)

network = distribute(network(θ), [2, 1])
# θ = init(network)

x = vec(x[rank+1,:])
y_out = network(θ) * x

# rank == 0 && println("Final Vec: ", vec(y_out))
@test norm(vec(y_true[rank+1,:]) - y_out) == 0

MPI.Finalize()
