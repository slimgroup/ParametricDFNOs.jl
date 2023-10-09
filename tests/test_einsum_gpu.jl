using Pkg
Pkg.activate("./")

using OMEinsum
using Flux

a = rand(2, 2, 3)
b = rand(2, 3)

out = ein"kij,ij->kj"(a,b); 
println(typeof(out))

out = ein"kij,ij->kj"(a |> gpu, b |> gpu); 
println(typeof(out))

out = einsum(EinCode(((3,1,2),(1,2)),(3,2)),(a |> gpu,b |> gpu))
println(typeof(out))

a = rand(20, 20, 8, 8, 4) |> gpu
b = rand(20, 8, 8, 4) |> gpu

out = einsum(EinCode(((5, 1, 2, 3, 4),(1, 2, 3, 4)),(5, 2, 3, 4)),(a,b))
println(typeof(out))
