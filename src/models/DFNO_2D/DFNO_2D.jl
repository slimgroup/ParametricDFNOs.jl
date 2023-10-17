module DFNO_2D

using Flux
using Parameters
using ParametricOperators

global gpu_flag = parse(Bool, get(ENV, "DFNO_2D_GPU", "0"))
@info "DFNO_2D is using " * (gpu_flag ? "GPU" : "CPU")

include("model.jl")
include("forward.jl")
# include("data.jl")
# include("plot.jl")

export Model, ModelConfig, initModel

end
