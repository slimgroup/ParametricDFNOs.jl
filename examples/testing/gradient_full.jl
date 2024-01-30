# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/testing/gradient_full.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("./grad_test.jl")

using .DFNO_3D
using MPI
using .GRADIENT_TESTS
using ParametricOperators
using Zygote
using LinearAlgebra

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)
dim = 10

partition = [1,pe_count]
@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 4)
modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, nt=dim, mx=modes, my=modes, mz=modes, mt=modes, nblocks=4, partition=partition, dtype=Float64)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

################### Testing gradient wrt permeability: ##################

x0 = rand(modelConfig.dtype, Domain(model.lifts))
Δx = rand(modelConfig.dtype, Domain(model.lifts))

function J(x)
    return norm(DFNO_3D.forward(model, θ, x))
end

g = gradient(rx -> J(rx), x0)[1]
println(size(g))

grad_test(J, x0, Δx, gradient(rx -> J(rx), x0)[1])

#########################################################################

MPI.Finalize()
