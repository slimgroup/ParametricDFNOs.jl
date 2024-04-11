# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/testing/simple_grad.jl

using Pkg
Pkg.activate("./")

# include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("./grad_test.jl")
include("../src/utils.jl")

# using .DFNO_3D
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

matrix = ParMatrix(Float64, 10, 10)
θ = init(matrix)
x = rand(Float64, 10)

for key in keys(θ)
    function J(input)
        θ[key] = input
        return norm((matrix(θ) * matrix(θ) * x))
    end

    original_matrix = θ[key]
    perturbation_scales = [0.05, 0.1, 0.2, 0.4]

    for scale in perturbation_scales
        perturbation = original_matrix .* scale
        g = gradient(Δθ -> J(Δθ), original_matrix)[1]
        println("Testing for key: $key with perturbation scale: $scale")
        # grad_test(J, original_matrix, perturbation, g)
        finite_diff_grad_test(J, original_matrix, perturbation, g)
    end
end


### Naive Ax as a sanity check ###

# for i in 1:100
#     w = rand(Float64, 10, 10)
#     x = rand(Float64, 10)

#     function J(input)
#         return norm(input * input * x)
#     end

#     g = gradient(w -> J(w), w)[1]
#     GRADIENT_TESTS.grad_test_slim(J, w, w .* .001, g, maxiter=20)
# end

##################################

### Naive Ax with A as glorot init ###

# for i in 1:100
#     scale = sqrt(24.0f0 / sum((10, 10)))
#     w = (rand(Float64, (10, 10)) .- 0.5f0) .* scale
#     w = permutedims(w, [2, 1])

#     x = rand(Float64, 10)

#     function J(input)
#         return norm(input * x)
#     end

#     g = gradient(w -> J(w), w)[1]
#     grad_test(J, w, w .* .2, g)
# end

######################################

MPI.Finalize()
