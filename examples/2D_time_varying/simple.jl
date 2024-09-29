using ParametricOperators
using Zygote
using LinearAlgebra

T = Float32

n_m, n_i, n_o = 10, 5, 5
r_m, r_i, r_o = 2, 3, 4

G = ParMatrix(T, r_o, r_m*r_i)

# Use after defining indexing into Parametrized adjoint operators
Um = ParMatrix(T, n_m, r_m)
Ui = ParMatrix(T, n_i, r_i)
Uo = ParMatrix(T, n_o, r_o)

UmT = ParMatrix(T, r_m, n_m)
UiT = ParMatrix(T, r_i, n_i)
UoT = ParMatrix(T, r_o, n_o)

θ = init(G)
init!(UmT, θ)
init!(UiT, θ)
init!(UoT, θ)

x = rand(T, n_i, n_m)

# Approach One:
O1 = vcat([UmT(θ)[:, j] ⊗ UiT(θ) for j in 1:n_m]...)
x1 = [x[:, j] for j in 1:n_m]
out1 = O1 .* x1

# Approach Two:
I2 = ParIdentity(T, n_m)
O2 = ParBlockDiagonal([UmT(θ)[:, j] for j in 1:n_m]...) ⊗ UiT(θ)
x2 = vec(x)
out2 = O2 * x2

# Approach Three
O3 = ParBlockDiagonal([UmT[:, j] for j in 1:n_m]...) ⊗ UiT
θ3 = init(O3)
x3 = vec(x)
out3 = O3(θ3) * x3

@time Zygote.gradient(θ -> norm(ParBlockDiagonal([UmT(θ)[:, j] for j in 1:n_m]...) ⊗ UiT(θ) * x2), θ)
@time Zygote.gradient(θ -> norm((vcat([UmT(θ)[:, j] ⊗ UiT(θ) for j in 1:n_m]...)) .* x1)[1], θ)
@time Zygote.gradient(θ -> norm(O3(θ) * x3)[1], θ3)

# # Computing with core tensor

# G(θ)*out1[1]
# G2 = I2 ⊗ G(θ)
# G2 * out2

# julia> @time Zygote.gradient(θ -> norm((vcat([UmT(θ)[:, j] ⊗ UiT(θ) for j in 1:n_m]...)) .* x1)[1], θ)
#  29.771543 seconds (79.32 M allocations: 4.264 GiB, 4.33% gc time, 99.95% compilation time)
# (Dict{Any, Any}(ParMatrix{Float32}(2, 10, UUID("23793b5f-18ec-4fe6-868c-a6a23cb59f95")) => Float32[0.021619463 0.041365314 … -0.06841033 -0.36554712; -0.14509471 -0.32989016 … 0.060822204 0.16024548], ParMatrix{Float32}(3, 5, UUID("f8db4461-2e8f-42f6-8a2f-798efbe94e88")) => Float32[-0.31470102 -0.3929607 … -0.38823116 -0.43602577; 0.3756008 0.3853066 … 0.4614981 0.404879; -0.7263116 -0.47961524 … -0.90268576 -0.7108831]),)

# julia> @time Zygote.gradient(θ -> norm(ParBlockDiagonal([UmT(θ)[:, j] for j in 1:n_m]...) ⊗ UiT(θ) * x2), θ)
#  14.979252 seconds (26.36 M allocations: 1.664 GiB, 4.37% gc time, 99.94% compilation time)
# (Dict{Any, Any}(ParMatrix{Float32}(3, 5, UUID("13c7469e-5eb0-4284-b0d0-cc737beffbe7")) => Float32[-0.00046107903 0.1658639 … 0.11110224 0.13540365; 0.08330176 -0.18010893 … -0.10970314 -0.103416115; -0.04102877 -0.039695833 … 0.03779302 -0.0093477545]),)

# julia> @time Zygote.gradient(θ -> norm(O3(θ) * x3)[1], θ3)
#  15.246902 seconds (27.19 M allocations: 1.733 GiB, 4.33% gc time, 99.93% compilation time)
# (Dict{Any, Any}(ParMatrix{Float32}(3, 5, UUID("a04431a1-fcb0-4b69-8310-3db9760bb7b7")) => Float32[-0.56285256 -0.547991 … -0.5048329 -0.6589144; -0.72398496 -0.6652122 … -0.57530206 -0.8249138; -0.07074441 -0.09562595 … -0.037364732 -0.10802621]),)
