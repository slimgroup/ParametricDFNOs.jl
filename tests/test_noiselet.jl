using Pkg
Pkg.activate("./")

using Zygote
using ParametricOperators
using LinearAlgebra

function noiselet_matrix(n::Int)

    # Check if n is a power of 2
    !ispow2(n) && error("The input argument should be of form 2^k")

    # Base noiselet matrix and transform
    N = [1]
    T = 0.5 * [1 - im 1 + im; 1 + im 1 - im]

    for _ = 1:log2(n)
        N = kron(T, N)
    end

    return N
end

n = 4

A = ParMatrix(ComplexF64, n, n)
mat = noiselet_matrix(n)
x = rand(n)

θ = init(A)
grad = gradient(Δθ -> 0.5 * norm(A(Δθ) * mat * x), θ)
