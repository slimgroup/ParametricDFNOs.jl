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

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

px, py, pz, dimx, dimy, dimz, dimt = parse.(Int, ARGS[1:7])
partition = [1,px,py,pz,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=dimx, ny=dimy, nz=dimz, nt=dimt, mt=min(dimt, 4), nblocks=4, partition=partition)
model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

x_sample = rand(modelConfig.dtype, dimx * dimy * dimz * dimt * 5 ÷ prod(partition), 1)
y_sample = rand(modelConfig.dtype, dimx * dimy * dimz * dimt * 1 ÷ prod(partition), 1)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, params, x_sample), y_sample)
    return loss
end

grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]

MPI.Finalize()
