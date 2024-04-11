# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/scaling/gradient_scaling_2d.jl 10 10

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_2D/DFNO_2D.jl")
include("../../src/utils.jl")

using .DFNO_2D
using .UTILS
using MPI
using Zygote
using DrWatson
using ParametricOperators
using CUDA

gpu = ParametricOperators.gpu

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dim, dimt = parse.(Int, ARGS[1:2])
partition = [1,size]

@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 4)
modelConfig = DFNO_2D.ModelConfig(nblocks=4,
partition=partition,
 nt=51,
nc_mid = 128,
 nc_lift = 20, 
mx = 4, 
my = 4,
mt = 4)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nt)
output_size = input_size * model.config.nc_out ÷ model.config.nc_in

x_sample = rand(modelConfig.dtype, input_size, 1)
y_sample = rand(modelConfig.dtype, output_size, 1) |> gpu

# GC.enable_logging(true)
@time y = DFNO_2D.forward(model, θ, x_sample)
@time y = DFNO_2D.forward(model, θ, x_sample)
@time y = DFNO_2D.forward(model, θ, x_sample)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_2D.forward(model, params, x_sample), y_sample)
    return loss
end

rank == 0 && println("STARTED GRADIENT SCALING")

@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]

MPI.Finalize()
