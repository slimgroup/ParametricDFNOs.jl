# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("src/models/DFNO_3D/DFNO_3D.jl")

using .DFNO_3D
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,2,2,1,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, ntrain=1, nvalid=1)

# model = DFNO_3D.Model(modelConfig)
# θ = DFNO_3D.initModel(model)

x_train, y_train, x_valid, y_valid = DFNO_3D.loadDistData(dataConfig)

# trainConfig = DFNO_3D.TrainConfig(
#     epochs=200,
#     x_train=x_train,
#     y_train=y_train,
#     x_valid=x_valid,
#     y_valid=y_valid,
# )

# DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
