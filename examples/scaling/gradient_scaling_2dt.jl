# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/scaling/scaling.jl
# mpiexecjl --project=./ -n 1 julia examples/scaling/gradient_scaling.jl 1 1 1 10 10 10 5

using Pkg
Pkg.activate("./")

include("../../src/models/TDFNO_2D/TDFNO_2D.jl")
include("../../src/utils.jl")

using .TDFNO_2D
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
# modelConfig = TDFNO_2D.ModelConfig(nx=dim, ny=dim, nt=dimt, mx=modes, my=modes, mt=modes, nblocks=4, partition=partition, dtype=Float32)

modes = 6
modelConfig = TDFNO_2D.ModelConfig(nblocks=1,
 partition=partition,
  nt=20,
 nc_mid = 70,
  nc_lift = 20, 
mx = modes, 
my = modes,
 mt = modes,
 TuckerRank = [5,5,3,3,3,1])


model = TDFNO_2D.Model(modelConfig)
θ = TDFNO_2D.initModel(model)

x_sample = rand(modelConfig.dtype, Domain(model.lifts), 1)
y_sample = rand(modelConfig.dtype, Range(model.projects[2]), 1) |> gpu

println("#######MODES=",modes)
# GC.enable_logging(true)
println("###FORWARD")
@time y = TDFNO_2D.forward(model, θ, x_sample)
@time y = TDFNO_2D.forward(model, θ, x_sample)
@time y = TDFNO_2D.forward(model, θ, x_sample)

function loss_helper(params)
    global loss = UTILS.dist_loss(TDFNO_2D.forward(model, params, x_sample), y_sample)
    return loss
end
println("###GRADIENT SCALING")
@time grads_time = gradient(params -> loss_helper(params), θ)[1]
@time grads_time = gradient(params -> loss_helper(params), θ)[1]
@time grads_time = gradient(params -> loss_helper(params), θ)[1]


MPI.Finalize()
