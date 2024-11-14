using ParametricOperators
using Zygote
using Flux
using LinearAlgebra

T = Float32

n_m, n_i, n_o = 10, 5, 5
r_m, r_i, r_o = 2, 3, 4

UmT = ParMatrix(T, r_m, n_m)
UiT = ParMatrix(T, r_i, n_i)

# Final Case 
O = ParBlockDiagonal([UmT[:, j] for j in 1:n_m]...) ⊗ UiT
θ = init(O)
x = vec(rand(T, n_i, n_m))
out = O(θ) * x

grads = Zygote.gradient(θ -> norm(O(θ) * x)[1], θ)[1]
opt = Flux.Optimise.ADAMW(0.01, (0.9f0, 0.999f0), 1f-4)

for (k, v) in θ
    Flux.Optimise.update!(opt, v, grads[k])
end

# # Final Case - 1
# B = ParBlockDiagonal([UmT[:, j] for j in 1:n_m]...)

# x = vec(rand(T, n_m))
# θ = init(B)

# out = B(θ) * x
# grads = Zygote.gradient(θ -> norm(B(θ) * x)[1], θ)[1]
# opt = Flux.Optimise.ADAMW(0.01, (0.9f0, 0.999f0), 1f-4)

# for (k, v) in θ
#     Flux.Optimise.update!(opt, v, grads[k])
# end

# # Final Case - 2
# B = ParBlockDiagonal([ParMatrix(T, r_m, 1) for j in 1:2]...)

# x = vec(rand(T, 2))
# θ = init(B)

# out = B(θ) * x
# grads = Zygote.gradient(θ -> norm(B(θ) * x)[1], θ)[1]
# opt = Flux.Optimise.ADAMW(0.01, (0.9f0, 0.999f0), 1f-4)

# for (k, v) in θ
#     Flux.Optimise.update!(opt, v, grads[k])
# end

# # Final Case - 3
# B = ParBlockDiagonal([ParMatrix(T, r_m, 1) for j in 1:2]...)

# x = vec(rand(T, 2))
# θ = init(B)

# y = B(θ) * x
# grads = Zygote.gradient(θ -> norm(B(θ) * x), θ)[1]
# grads = Zygote.gradient(θ -> norm(B'(θ) * y), θ)[1]

# # Final Case - 4

# ### Block Diagonal
# B = ParBlockDiagonal(ParMatrix(T, r_m, 1))

# x = vec(rand(T, 1))
# θ = init(B)

# y = B(θ) * x
# grads = Zygote.gradient(θ -> norm(B(θ) * x), θ)[1]
# grads = Zygote.gradient(θ -> norm(B(θ)'*B(θ) * x), θ)[1]
# grads = Zygote.gradient(θ -> norm(B'(θ)*B(θ) * x), θ)[1]

# ### Kronecker
# K = ParMatrix(T, r_m, 1) ⊗ ParMatrix(T, r_m, 1)

# xk = vec(rand(T, Domain(K)))
# θk = init(K)

# grads = Zygote.gradient(θ -> norm(K(θ)'*K(θ)*xk), θk)[1]
# grads = Zygote.gradient(θ -> norm(K'(θ)*K(θ)*xk), θk)[1]

# # Final Case - 5

# M = ParMatrix(T, r_m, 1)

# xm = vec(rand(T, Domain(M)))
# θm = init(M)

# grads = Zygote.gradient(θ -> norm(M(θ)'*M(θ)*xm), θm)[1]
# grads = Zygote.gradient(θ -> norm(M'(θ)*M(θ)*xm), θm)[1]
