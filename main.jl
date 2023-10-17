# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia fno4co2.jl

using Pkg
Pkg.activate("./")

include("src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
using MPI
using LinearAlgebra
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

config = DFNO_2D.ModelConfig(partition=[1,1,1,1])
model = DFNO_2D.Model(config)

θ = DFNO_2D.initModel(model)

MPI.Finalize()
