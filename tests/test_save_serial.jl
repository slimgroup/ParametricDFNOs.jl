# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia main.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")
include("../src/utils.jl")

using .DFNO_2D
using .UTILS
using MPI
using Random
using Zygote
using DrWatson
using LinearAlgebra
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,1,1,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D.ModelConfig(partition=partition)
model = DFNO_2D.Model(modelConfig)

θ_save = DFNO_2D.initModel(model)
# DFNO_2D.loadWeights!(θ_save, "exp=serial_test_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2", "θ_save", partition)

rng = Random.seed!(1234)

x = rand(rng, DDT(model.lifts), 4*64*64*51)
y = rand(rng, DDT(model.lifts), 1*64*64*51)

y_out = DFNO_2D.forward(model, θ_save, x)

grads = gradient(params -> UTILS.dist_loss(DFNO_2D.forward(model, params, x), y), θ_save)[1]
grads = DFNO_2D.collectWeights(grads, model)

exp = "serial_test"
serial_test = @strdict exp y_out grads

DFNO_2D.saveWeights(θ_save, model, additional=serial_test)

MPI.Finalize()
