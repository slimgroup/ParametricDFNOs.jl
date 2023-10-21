# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia tests/test_load_dist.jl

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

partition = [1,2,2,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D.ModelConfig(partition=partition)
model = DFNO_2D.Model(modelConfig)

θ = DFNO_2D.initModel(model)
grads_true = DFNO_2D.initModel(model)

# Load existing stuff from serially trained FNO
weights_file = "exp=dist_save_test_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2"

DFNO_2D.loadWeights!(θ, weights_file, "θ_save", partition)
DFNO_2D.loadWeights!(grads_true, weights_file, "grads", partition)

file = projectdir("weights", "DFNO_2D", weights_file)
y_save = load(file)["y_out"]

rng = Random.seed!(1234)

x = rand(rng, DDT(model.lifts), 4*64*64*51)
y = rand(rng, DDT(model.lifts), 1*64*64*51)

shape_in = (4,64,64,51)
shape_out = (1,64,64,51)

x = vec(UTILS.dist_tensor(x, shape_in, partition))
y = vec(UTILS.dist_tensor(y, shape_out, partition))
y_true = vec(UTILS.dist_tensor(y_save, shape_out, partition))

y_out = DFNO_2D.forward(model, θ, x)
y_norm = UTILS.dist_loss(y_out, y_true)

grads = gradient(params -> UTILS.dist_loss(DFNO_2D.forward(model, params, x), y), θ)[1]

for (k, v) in grads
    @assert norm(v - grads_true[k]) <= 1e-10
end

@assert y_norm <= 1e-10

MPI.Finalize()
