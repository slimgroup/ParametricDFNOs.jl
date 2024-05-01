module DFNO_2D

using MAT
using MPI
using Flux
using JLD2
using HDF5
using PyPlot
using Printf
using Random
using Zygote
using DrWatson
using Parameters
using ProgressMeter
using ParametricOperators
# using InvertibleNetworks:ActNorm

global model_name = "DFNO_2D"
global gpu_flag = parse(Bool, get(ENV, "DFNO_2D_GPU", "0"))
global plot_path = plotsdir(model_name)
gpu_flag && (global gpu = ParametricOperators.gpu)
global cpu = ParametricOperators.cpu

@info "DFNO_2D is using " * (gpu_flag ? "GPU" : "CPU")

include("model.jl")
include("forward.jl")
include("data.jl")
include("train.jl")
include("plot.jl")
include("weights.jl")
include("../../utils.jl")

using .UTILS

"""
    set_gpu_flag(flag::Bool)

Function to set the gpu_flag and update device accordingly. Should be set at the beginning of your script. All FNO computatation following this will use the device set.
"""
function set_gpu_flag(flag::Bool; comm=MPI.COMM_WORLD)
    global gpu_flag = flag
    rank = MPI.Comm_rank(comm)
    if gpu_flag
        global gpu = ParametricOperators.gpu
        rank == 0 && @info "DFNO_2D Switched to GPU"
    else
        global cpu = ParametricOperators.cpu
        rank == 0 && @info "DFNO_2D Switched to CPU"
    end
end

export Model, ModelConfig, DataConfig, TrainConfig, initModel, loadData, train, plotLoss, plotEvaluation, loss, saveWeights, loadWeights!, print_storage_complexity, loadDistData, gpu_flag, set_gpu_flag

end
