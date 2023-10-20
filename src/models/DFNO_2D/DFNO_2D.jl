module DFNO_2D

using MAT
using MPI
using Flux
using PyPlot
using Random
using DrWatson
using Parameters
using ProgressMeter
using ParametricOperators
using InvertibleNetworks:ActNorm

global model_name = "DFNO_2D"
global gpu_flag = parse(Bool, get(ENV, "DFNO_2D_GPU", "0"))
global plot_path = plotsdir(model_name)

@info "DFNO_2D is using " * (gpu_flag ? "GPU" : "CPU")

include("model.jl")
include("forward.jl")
include("data.jl")
include("train.jl")
include("plot.jl")

export Model, ModelConfig, TrainConfig, initModel, loadData, train, plotLoss, plotEvaluation, loss, dist_tensor

end
