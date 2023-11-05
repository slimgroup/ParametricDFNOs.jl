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

nodes, gpus, ntasks, px, py, pz, dimx, dimy, dimz, dimt = parse.(Int, ARGS[1:10])
config = ARGS[11]

partition = [1,px,py,pz,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=dimx, ny=dimy, nz=dimz, nt=dimt, mt=min(dimt, 4), nblocks=4, partition=partition)
model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

x_sample = rand(modelConfig.dtype, dimx * dimy * dimz * dimt * 5 ÷ prod(partition), 1)

y = DFNO_3D.forward(model, θ, x_sample)

exit()
y = DFNO_3D.forward(model, θ, x_sample)
y_time = @elapsed DFNO_3D.forward(model, θ, x_sample)
y_time = UTILS.dist_sum([y_time]) / size

y = y .+ rand(modelConfig.dtype)

function loss_helper(params)
    global loss = sum(DFNO_3D.forward(model, params, x_sample)) # UTILS.dist_loss(DFNO_3D.forward(model, params, x_sample), y)
    return loss
end

grads = gradient(params -> loss_helper(params), θ)[1]
grads = gradient(params -> loss_helper(params), θ)[1]
grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
grads_time = UTILS.dist_sum([grads_time]) / size

final_dict = @strdict nodes gpus ntasks px py pz dimx dimy dimz dimt y_time grads_time config

if rank == 0
    mkpath(projectdir("examples", "scaling", "results"))
    @tagsave(
        projectdir("examples", "scaling", "results", savename(final_dict, "jld2"; digits=6)),
        final_dict;
        safe=true
    )
end

MPI.Finalize()
