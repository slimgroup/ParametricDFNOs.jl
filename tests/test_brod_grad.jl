# source $HOME/.bash_profile
# source ~/.bashrc
# mpiexecjl --project=./ -n 2 julia tests/test_brod_grad.jl

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

### Serial Gradient Computation ###

network1 = ParMatrix(T, 2, 2)
x1 = reshape(float(1:4), 2, 2)

θ1 = init(network1)
grads1 = gradient(params -> sum(network1(params) * x1), θ1)[1]

### Distributed Gradient Computation, note: not the exact same bc loss is sum along PE instead of Comm ###

network2 = ParBroadcasted(ParMatrix(T, 2, 2), comm)

x2 = reshape(float(rank*2 + 1:rank*2 + 2), 2)

θ2 = init(network2)
grads2 = gradient(params -> sum(network2(params) * x2), θ2)[1]

### Test ###

(rank == 0) && @assert norm(vec(collect(values(grads1))[1]) - vec(collect(values(grads2))[1])) <= 1e-10
(rank == 0) && println(grads2)

MPI.Finalize()
