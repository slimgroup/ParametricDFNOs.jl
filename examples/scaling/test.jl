# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/scaling/scaling.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("../../src/utils.jl")

using .DFNO_3D
using .UTILS
using MPI
using Zygote
using DrWatson
using ParametricOperators
using CUDA

# gpu = ParametricOperators.gpu

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

partition = [1, 1]

@assert MPI.Comm_size(comm) == prod(partition)

T = Float32

a = ParMatrix(T, 10, 10)
b = ParMatrix(T, 10, 10)
c = ParMatrix(T, 10, 10)
d = ParMatrix(T, 10, 10)

e = kron(a ⊗ b, c ⊗ d)
e = distribute(e, partition)

θ = init(e)
x = rand(T, Domain(e))

e(θ) * x
MPI.Finalize()
