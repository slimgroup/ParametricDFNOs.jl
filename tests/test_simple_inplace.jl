# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/scaling/scaling.jl
# mpiexecjl --project=./ -n 1 julia examples/scaling/gradient_scaling.jl 1 1 1 10 10 10 5

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_3D/DFNO_3D.jl")
include("../src/utils.jl")

using .DFNO_3D
using .UTILS
using MPI
using Zygote
using Flux
using DrWatson
using ParametricOperators
using CUDA

# gpu = ParametricOperators.gpu

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

m = ParMatrix(1000, 1000)
θ = init(m)

x = rand(1000, 1000)

function g!(x, y)
  x .= relu.(y)

  return x
end

function g1(y)
  x = Zygote.Buffer(y) # Buffer supports syntax like similar
  g!(x, y)
  return copy(x) # this step makes the Buffer immutable (w/o actually copying)
end
function g2(y)
    y = relu.(y)
    return y
end

@time gradient(w -> sum(g2(m(θ) * x)), θ)

MPI.Finalize()
