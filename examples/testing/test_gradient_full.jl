# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/testing/test_gradient_full.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("./grad_test.jl")
include("../../src/utils.jl")

using .DFNO_3D
using .UTILS
using MPI
using .GRADIENT_TESTS
using ParametricOperators
using Zygote
using LinearAlgebra
using Flux

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)
dim = 10

partition = [1,pe_count]
@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 2)
modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, nt=dim, mx=modes, my=modes, mz=modes, mt=modes, nblocks=2, partition=partition, dtype=Float64, nc_lift=5, nc_mid=20)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

x0 = rand(modelConfig.dtype, Domain(model.lifts))
y = rand(modelConfig.dtype, Range(model.projects[2]))

# ################### Testing gradient wrt permeability: ##################

# for i in 1:100
#     Δx = rand(modelConfig.dtype, Domain(model.lifts))

#     function J(x)
#         return UTILS.dist_loss(DFNO_3D.forward(model, θ, x), y)
#     end

#     g = gradient(rx -> J(rx), x0)[1]

#     grad_test(J, x0, Δx, gradient(rx -> J(rx), x0)[1])
# end

#########################################################################

# grad_test for a dictionary
function grad_test_dict(model, θ, x0, Δθ, full_gradient)
    for key in keys(θ)
        println("Testing gradient for key: $key")

        # Extract the relevant gradient for the current key
        g_key = full_gradient[key]

        if isnothing(g_key)
            println("Gradient for key $key is Nothing. Skipping...")
            continue
        end

        # Create a J function for the current key
        function J_key_specific(x)
            θ_modified = Dict(k => (k == key ? x : v) for (k, v) in θ)
            return UTILS.dist_loss(DFNO_3D.forward(model, θ_modified, x0), y)
        end

        x0_key = θ[key]  # Initial value for the parameter of interest
        Δx_key = Δθ[key]  # Perturbation for the parameter

        # Perform the gradient test for the current key
        finite_diff_grad_test(J_key_specific, x0_key, Δx_key, g_key, stol=1e-4)
    end
end

############### Testing gradient wrt each of the weights ################

# First, create a perturbation dictionary Δθ with the same keys as θ
Δθ = Dict(key => randn(size(value)) for (key, value) in θ)

# Define a function J for the full model forward pass
function J_full(params)
    return UTILS.dist_loss(DFNO_3D.forward(model, params, x0), y)
end

full_gradient = gradient(params -> J_full(params), θ)[1]
grad_test_dict(model, θ, x0, Δθ, full_gradient)

#########################################################################

MPI.Finalize()
