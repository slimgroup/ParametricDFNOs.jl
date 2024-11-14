using ParametricOperators
using ParametricOperators: ⊠
using Zygote
using Flux
using LinearAlgebra
using OMEinsum

T = Float32

nc_out = 10
nc_in = 8
mt = 6

input_shape = (nc_in, mt)
weight_shape = (nc_in, nc_out, mt)

ranks = [4, 3, 2]

G = ParMatrix(T, ranks[2], prod(ranks[[1,3]]))

Uo = ParMatrix(T, weight_shape[2], ranks[2])
UiT = ParMatrix(T, ranks[1], weight_shape[1])
UtT = ParMatrix(T, ranks[3], weight_shape[3])

Ut = reduce(⊠, [UtT[:, j] for j in 1:weight_shape[3]])

I = ParIdentity(T, prod(input_shape[2:end]))

O = (I ⊗ Uo) * (I ⊗ G) * (Ut ⊗ UiT)

x = rand(T, input_shape...)
θ = init(O)

y = reshape(O(θ)*vec(x), nc_out, mt)

input = x
factor_i = θ[UiT]'
factor_o = θ[Uo]
factor_m = hcat([θ[Ut.ops[i]] for i in 1:weight_shape[3]]...)'
core = reshape(θ[G], ranks[2], ranks[1], ranks[3])
core = permutedims(core, [2, 1, 3])

contract_tucker = ein"bc,ghi,bg,fh,ci->fc"(input, core, factor_i, factor_o, factor_m)
