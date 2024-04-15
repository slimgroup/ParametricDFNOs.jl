include("src/models/DFNO_3D/DFNO_3D.jl")
include("src/utils.jl")

using .DFNO_3D
using .UTILS
using MPI
using Zygote
using DrWatson
using CUDA

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

global gpu_flag = parse(Bool, get(ENV, "DFNO_3D_GPU", "0"))
DFNO_3D.set_gpu_flag(gpu_flag)

# Julia requires you to manually assign the gpus, modify to your case.
DFNO_3D.gpu_flag && (CUDA.device!(rank % 4))
partition = [1, pe_count]

nodes, gpus, dimx, dimy, dimz, dimt, nblocks = parse.(Int, ARGS[1:7])
config = ARGS[8]

modesx = 16
modesy = 4
modesz = 4
modest = 2

modelConfig = DFNO_3D.ModelConfig(nx=dimx, ny=dimy, nz=dimz, nt=dimt, mx=modesx, my=modesy, mz=modesz, mt=modest, nblocks=nblocks, partition=partition)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

x_sample = rand(modelConfig.dtype, dimx * dimy * dimz * dimt * 5 ÷ prod(partition), 1)

@time y = DFNO_3D.forward(model, θ, x_sample)
@time y = DFNO_3D.forward(model, θ, x_sample)
y_time = @elapsed DFNO_3D.forward(model, θ, x_sample)
y_time = UTILS.dist_sum([y_time]) / prod(partition)

y = y .+ rand(modelConfig.dtype)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, params, x_sample), y)
    return loss
end

grads_time = 0

if config !== "weak_forward"
    rank == 0 && println("STARTING GRADIENT SCALING")
    grads = gradient(params -> loss_helper(params), θ)[1]
    grads = gradient(params -> loss_helper(params), θ)[1]
    grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
    grads_time = UTILS.dist_sum([grads_time]) / prod(partition)
end

final_dict = @strdict nodes gpus dimx dimy dimz dimt y_time grads_time config

if rank == 0
    mkpath(projectdir("scaling_results"))
    @tagsave(
        projectdir("scaling_results", savename(final_dict, "jld2"; digits=6)),
        final_dict;
        safe=true
    )
end

MPI.Finalize()
