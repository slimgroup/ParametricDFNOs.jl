using Pkg
Pkg.activate("./")

using DrWatson
using LinearAlgebra
using ParametricOperators

serial_file = projectdir("weights", "DFNO_2D", "exp=serial_test_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2")
dist_file = projectdir("weights", "DFNO_2D", "exp=dist_save_test_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2")

θ_1 = load(serial_file)["θ_save"]
θ_2 = load(dist_file)["θ_save"]

for (k, v) in θ_1
    @assert norm(v - θ_2[k]) <= 1e-10
end

grads_1 = load(serial_file)["grads"]
grads_2 = load(dist_file)["grads"]

for (k, v) in grads_1
    @assert norm(v - grads_2[k]) <= 1e-10
end
