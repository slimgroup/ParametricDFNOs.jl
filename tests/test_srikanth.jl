using Pkg
Pkg.activate("./")

using ParametricOperators
using LinearAlgebra
using Zygote

n = 2
T = Float32

U = ParMatrix(T, 2*n,2*n)
V = ParMatrix(T, n,n)
W = ParMatrix(T, n,n)

X = rand(T, 2*n,2*n)
Id = ParIdentity(T, 2*n)

theta_U = init(U)
theta_V = init(V)
theta_W = init(W)

@time grads_L = gradient(theta_U -> norm((Id ⊗ (U(theta_U)*(V(theta_V)⊗W(theta_W))))*vec(X)), theta_U)[1]
@time grads_L = gradient(theta_U -> norm((Id ⊗ ((U(theta_U)[1,:])*(V(theta_V)⊗W(theta_W))))*vec(X)), theta_U)[1]
