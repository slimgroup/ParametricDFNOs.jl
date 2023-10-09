using Pkg
Pkg.activate("./")

using ParametricOperators

A = ParMatrix(Float32, 4, 5)
θ = init(A)
x = randn(Float32, 5)
y = randn(Float32, 4)
output1 = A(θ) * x
adj = A'
output2 = adj(θ) * y
