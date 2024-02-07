# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia examples/training/training_2d.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_2D_OLD/DFNO_2D_OLD.jl")
include("../../src/utils.jl")

using .DFNO_2D_OLD
using MPI
using .UTILS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,2,2,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D_OLD.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_2D_OLD.DataConfig(modelConfig=modelConfig)

x_train, y_train, x_valid, y_valid = DFNO_2D_OLD.loadDistData(dataConfig)

trainConfig = DFNO_2D_OLD.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

model = DFNO_2D_OLD.Model(modelConfig)
θ = DFNO_2D_OLD.initModel(model)

DFNO_2D_OLD.train!(trainConfig, model, θ)

MPI.Finalize()
