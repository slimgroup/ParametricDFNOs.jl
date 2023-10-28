# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,2,2,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_2D.DataConfig(modelConfig=modelConfig)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

# # To train from a checkpoint
# filename = "/path/to/checkpoint.jld2"
# DFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

x_train, y_train, x_valid, y_valid = DFNO_2D.loadDistData(dataConfig)

trainConfig = DFNO_2D.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
