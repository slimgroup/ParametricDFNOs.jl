# source $HOME/.bash_profile
# julia examples/storage.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D

modelConfig = DFNO_2D.ModelConfig(dtype=Float64, nblocks=1)

for b in 1:10
    DFNO_2D.print_storage_complexity(modelConfig, batch=b)
end
