# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia examples/training/training_2d.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_2D/DFNO_2D.jl")
include("../../src/utils.jl")

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
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    plot_every=10
)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

# # To train from a checkpoint
# filename = "ep=80_mt=4_mx=4_my=4_nblocks=4_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt=51_nx=64_ny=64_p=1.jld2"
# DFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
