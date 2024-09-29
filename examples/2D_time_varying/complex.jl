using ParametricOperators
using Zygote
using LinearAlgebra

T = Complex{Float32}

nc_lift = 5
mx = 8
my = 8
mt = 4

input_shape = (nc_lift, mt, mx*my)
weight_shape = (nc_lift, nc_lift, mt, mx*my)

factorization_ranks = [4, 4, 4, 4]

G = ParMatrix(T, factorization_ranks[1], prod(factorization_ranks[2:end]))

Uo = ParMatrix(T, weight_shape[1], factorization_ranks[1])
UiT = ParMatrix(T, factorization_ranks[2], weight_shape[2])
UtT = ParMatrix(T, factorization_ranks[3], weight_shape[3])
UmT = ParMatrix(T, factorization_ranks[4], weight_shape[4])

Ut = reduce(⊠, [UtT[:, j] for j in 1:weight_shape[3]])
Um = reduce(⊠, [UmT[:, j] for j in 1:weight_shape[4]])

I = ParIdentity(T, prod(input_shape[2:end]))

O = (I ⊗ Uo) * (I ⊗ G) * (Um ⊗ Ut ⊗ UiT)

x = rand(T, input_shape...)
θ = init(O)

Zygote.gradient(θ -> norm(O(θ) * vec(x)), θ)
