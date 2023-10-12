using Pkg
Pkg.activate("./")

using Random
using Statistics, LinearAlgebra

rng = Random.seed!(1234)
root = rand(rng, 10, 20, 30, 2)

N = ndims(root)
ϵ = 1f-5

reduce_dims = [1:N-2; N]

μ = mean(root; dims=reduce_dims)
σ² = var(root; mean=μ, dims=reduce_dims, corrected=false)

# println(norm(vec(μ)), " ", norm(vec(σ²)))

x = (root .- μ) ./ sqrt.(σ² .+ ϵ)
x = reshape(x, (600, :))

x1 = root[1:5, :, :, :]
x2 = root[6:10, :, :, :]

function same(tensor)
    s = sum(tensor, dims=reduce_dims)
    return s
end

s1 = same(x1)
s2 = same(x2)

μ_2 = (s1 + s2) ./ prod(size(root)[reduce_dims])

@assert norm(vec(μ)) - norm(vec(μ_2)) <= 1e-10

s1 = (x1 .- μ_2) .^ 2
s2 = (x2 .- μ_2) .^ 2

s1 = same(s1)
s2 = same(s2)

σ²_2 = (s1 + s2) ./ prod(size(root)[reduce_dims])

@assert norm(vec(σ²)) - norm(vec(σ²_2)) <= 1e-10

x1 = (x1 .- μ_2) ./ sqrt.(σ²_2 .+ ϵ)
x2 = (x2 .- μ_2) ./ sqrt.(σ²_2 .+ ϵ)

x12 = root

x12[1:5, :, :, :] = x1
x12[6:10, :, :, :] = x2

@assert norm(vec(x12)) - norm(vec(x)) <= 1e-10
