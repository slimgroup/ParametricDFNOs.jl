# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/scaling/gradient_wrt_input.jl 10 10 10 10 4

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
using Flux

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

nx, ny, nz, nt, nblocks = parse.(Int, ARGS[1:5])
partition = [1,size]

@assert MPI.Comm_size(comm) == prod(partition)

modes = 4 # max(dim÷8, 4)
modelConfig = DFNO_3D.ModelConfig(nx=nx, ny=ny, nz=nz, nt=nt, mx=modes, my=modes, mz=modes, mt=modes, nblocks=nblocks, partition=partition, dtype=Float32)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nz * model.config.nt)
output_size = input_size * model.config.nc_out ÷ model.config.nc_in

x_sample = rand(modelConfig.dtype, input_size, 1)
y_sample = rand(modelConfig.dtype, output_size, 1)

gpu_flag && (gpu = ParametricOperators.gpu)
gpu_flag && (y_sample = y_sample |> gpu)

# GC.enable_logging(true)
@time y = DFNO_3D.forward(model, θ, x_sample)
@time y = DFNO_3D.forward(model, θ, x_sample)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, params, x_sample), y_sample)
    return loss
end

@time grads_time = @elapsed Flux.gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed Flux.gradient(params -> loss_helper(params), θ)[1]

function loss_helper_2(params)
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, θ, params), y_sample)
    return loss
end

@time grads_time = @elapsed Flux.gradient(params -> loss_helper_2(params), x_sample)[1]
@time grads_time = @elapsed Flux.gradient(params -> loss_helper_2(params), x_sample)[1]

x1 = x_sample[1:1, :]
x2 = x_sample[2:end, :]

function loss_helper_3(params)
    input = cat(params, x2, dims=1)
    println(typeof(input))
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, θ, input), y_sample)
    return loss
end

@time grads_time = @elapsed Flux.gradient(params -> loss_helper_3(params), x1)[1]
@time grads_time = @elapsed Flux.gradient(params -> loss_helper_3(params), x1)[1]

MPI.Finalize()