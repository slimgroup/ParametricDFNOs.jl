# source $HOME/.bash_profile
# source ~/.bashrc
# mpiexecjl --project=./ -n 2 julia tests/test_loss_grad.jl

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

network1 = ParMatrix(T, 2, 2)
x1 = reshape(float(1:4), 2, 2)

θ1 = init(network1)
grads1 = gradient(params -> norm(vec(network1(params) * x1)), θ1)[1]

# rank == 0 && println("serial out: ", network1(θ1) * x1)
# rank == 0 && println("serial grads: ", grads1)

function loss(local_pred_y, local_true_y)
    s = sum((vec(local_pred_y) - vec(local_true_y)) .^ 2)

    reduce_norm = ParReduce(eltype(local_pred_y))
    reduce_y = ParReduce(eltype(local_true_y))

    norm_diff = √(reduce_norm([s])[1])
    norm_y = √(reduce_y([sum(local_true_y .^ 2)])[1])

    return norm_diff
end

test_reduce = ParReduce(T)
broad = ParBroadcasted(ParMatrix(T, 2, 2), comm)

θ = init(broad)
x = reshape(float(rank*2+1:rank*2+2), 2)
y = zeros(2)

# println("dist out: ", rank, " --- ", broad(θ) * x)
grads2 = gradient(params -> loss(broad(params) * x, y), θ)[1]
(rank == 0) && @assert norm(vec(collect(values(grads1))[1]) - vec(collect(values(grads2))[1])) <= 1e-10

MPI.Finalize()