# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
using MPI
using DrWatson
using LinearAlgebra
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,2,2,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D.ModelConfig(nblocks=1, partition=partition)
model = DFNO_2D.Model(modelConfig)

θ = DFNO_2D.initModel(model)

# # To save starting weights
# labels = Dict{String, Any}("dtype" => "Float32", "exp" => "serial_start")
# DFNO_2D.saveWeights(θ, model, additional=labels)

# # To train from a checkpoint
# filename = "dtype=Float32_exp=serial_start_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2"
# DFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

x_train, y_train, x_valid, y_valid = DFNO_2D.loadData(partition)

trainConfig = DFNO_2D.TrainConfig(
    epochs=1,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
