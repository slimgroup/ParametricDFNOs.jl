# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia examples/training/training_3d.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("../../src/utils.jl")

using .DFNO_3D
using MPI
using .UTILS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]

modelConfig = DFNO_3D.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig)

x_train, y_train, x_valid, y_valid = DFNO_3D.loadDistData(dataConfig)

trainConfig = DFNO_3D.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# # To train from a checkpoint
# filename = "/path/to/checkpoint.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
