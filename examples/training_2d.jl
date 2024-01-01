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
dataConfig = DFNO_2D.DataConfig(modelConfig=modelConfig)

x_train, y_train, x_valid, y_valid = DFNO_2D.loadDistData(dataConfig)

trainConfig = DFNO_2D.TrainConfig(
    epochs=1,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

# # To train from a checkpoint
# filename = "/path/to/checkpoint.jld2"
# DFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

# x_sample_cpu = x_train[:, :, 1:1]
# y_sample_cpu = y_train[:, :, 1:1]

# x_global_shape = (modelConfig.nc_in * modelConfig.nt, modelConfig.nx * modelConfig.ny)
# y_global_shape = (modelConfig.nc_out * modelConfig.nt, modelConfig.nx * modelConfig.ny)

# x_sample_global = UTILS.collect_dist_tensor(x_sample_cpu, x_global_shape, modelConfig.partition, comm)
# y_sample_global = UTILS.collect_dist_tensor(y_sample_cpu, y_global_shape, modelConfig.partition, comm)

# if rank > 0
#     MPI.Finalize()
#     exit()
# end

# plotEvaluation(modelConfig, trainConfig, x_sample_global, y_sample_global, y_sample_global)
DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
