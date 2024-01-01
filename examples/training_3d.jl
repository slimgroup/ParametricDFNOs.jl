# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_3D/DFNO_3D.jl")

using .DFNO_3D
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

partition = [1,size]

modelConfig = DFNO_3D.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig) # Or Provide custom datafile path, channels and permutation. see examples/perlmutter

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# # To train from a checkpoint
# filename = "/path/to/checkpoint.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

x_train, y_train, x_valid, y_valid = DFNO_3D.loadDistData(dataConfig)

trainConfig = DFNO_3D.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
