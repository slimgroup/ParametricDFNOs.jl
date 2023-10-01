using Pkg
Pkg.activate("./")

using OMEinsum
using ParametricOperators

gpu = ParametricOperators.gpu

a = rand(2, 2) |> gpu
b = rand(2, 2) |> gpu

out = einsum(EinCode(((1,2),(2,3)),(1,3)),(a,b))
println(typeof(out))
