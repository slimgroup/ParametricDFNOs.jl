# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/scaling/scaling.jl
# mpiexecjl --project=./ -n 1 julia examples/scaling/gradient_scaling.jl 1 1 1 10 10 10 5

using Pkg
Pkg.activate("./")

# include("../../src/models/DFNO_2D/DFNO_2D.jl")
# include("../../src/utils.jl")

# using .DFNO_2D
# using .UTILS
# using MPI
using Zygote
using ParametricOperators
# using CUDA

# gpu = ParametricOperators.gpu

# MPI.Init()

T = Float64

m = ParMatrix(T, 10000, 100000)
a = rand(T, 10000, 100000)

x = rand(T, 100000)

θ = init(m)

@time m(θ) * x
@time m(θ) * x
@time m(θ) * x

@time a * x
@time a * x
@time a * x

# MPI.Finalize()
