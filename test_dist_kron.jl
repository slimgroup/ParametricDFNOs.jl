# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia test_dist_kron.jl

using Pkg
Pkg.activate("./")

using MPI
using ParametricOperators
using LinearAlgebra
using Random

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

T = Float32

network = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParIdentity(T, 2)
network = distribute(network, [1, 2, 1])

x = rand(Random.seed!(1234), T, 2, 2, 2)

rank == 0 && println(x)

x = vec(x[:,rank+1,:])

# plot_operator_graph(network)

# x = rand(Random.seed!(1234), T, Domain(network))
y_out = network * x

# rank == 0 && println(vec(x))
# rank == 0 && println(vec(y_out))

println("norm ", norm(x - y_out), " @ ", rank, " ", )

MPI.Finalize()
