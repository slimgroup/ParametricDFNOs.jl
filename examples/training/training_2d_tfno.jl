# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia examples/training/training_2d.jl

using Pkg
Pkg.activate("./")

include("../../src/models/TDFNO_2D/TDFNO_2D.jl")
include("../../src/utils.jl")

using .TDFNO_2D
using MPI
using .UTILS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]

modelConfig = TDFNO_2D.ModelConfig(nblocks=4,
 partition=partition,
  nt=11,
 nc_mid = 50,
  nc_lift = 20, 
mx = 4, 
my = 4,
 mt = 4)
dataConfig = TDFNO_2D.DataConfig(modelConfig=modelConfig)

x_train, y_train, x_valid, y_valid = TDFNO_2D.loadDistData(dataConfig)
# x_train, y_train, x_valid, y_valid = TDFNO_2D.loadData(comm)


trainConfig = TDFNO_2D.TrainConfig(
    epochs=100,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    plot_every=1
)

model = TDFNO_2D.Model(modelConfig)
θ = TDFNO_2D.initModel(model)

# # To train from a checkpoint
# filename = "ep=80_mt=4_mx=4_my=4_nblocks=4_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt=51_nx=64_ny=64_p=1.jld2"
# TDFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

TDFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
