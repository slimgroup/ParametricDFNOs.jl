# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia examples/training_2d.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")
include("../src/utils.jl")

using .DFNO_2D
using MPI
using .UTILS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]

modelConfig = DFNO_2D.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_2D.DataConfig(modelConfig=modelConfig, ntrain=1, nvalid=1)

x_train, y_train, x_valid, y_valid = DFNO_2D.loadDistData(dataConfig)

trainConfig = DFNO_2D.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_train,
    y_valid=y_train,
    nbatch=1
)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
