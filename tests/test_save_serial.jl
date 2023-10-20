# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia main.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
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

θ = DFNO_2D.initModel(model)
rng = Random.seed!(1234)

x = rand(rng, DDT(model.lifts), 4*64*64*51)
y = rand(rng, DDT(model.lifts), 1*64*64*51)

y_out = DFNO_2D.forward(model, θ, x)

grads = gradient(params -> loss(DFNO_2D.forward(model, params, x), y), θ)[1]
model = "serial"

serial_test = @strdict model y_out grads θ

mkpath(projectdir("gradient_test"))
@tagsave(
    projectdir("gradient_test", savename(serial_test, "jld2"; digits=6)),
    serial_test;
    safe=true
)

MPI.Finalize()
