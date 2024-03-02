module DFNO_3D_OLD

using MAT
using MPI
using Flux
using JLD2
using HDF5
using FileIO
using PyPlot
using Printf
using Random
using Zygote
using DrWatson
using Parameters
using ProgressMeter
using ParametricOperators

using CUDA

global model_name = "DFNO_3D_OLD"
global gpu_flag = parse(Bool, get(ENV, "DFNO_3D_OLD_GPU", "0"))
global plot_path = plotsdir(model_name)
gpu_flag && (global gpu = ParametricOperators.gpu)
global cpu = ParametricOperators.cpu

@info "DFNO_3D_OLD is using " * (gpu_flag ? "GPU" : "CPU")

include("model.jl")
include("forward.jl")
include("../../utils.jl")

using .UTILS

export Model, ModelConfig, initModel, gpu_flag

end
