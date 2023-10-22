using Pkg
Pkg.activate("./")

using DrWatson
using LinearAlgebra
using ParametricOperators

# serial_1 is obtained by trianing backup script 7. this is to make sure that all serial comparisons hereon will be from running the MPI job on 1 node
# serial_2 is obtained by running the dfno on one MPI worker

serial_1 = load(projectdir("weights", "DFNO_2D", "dtype=Float64_batch_size=2_dt=0.02_epochs=1_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2"))
serial_2 = load(projectdir("weights", "DFNO_2D", "dtype=Float64_p=1_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2"))

θ_1 = serial_1["θ_save"]
θ_2 = serial_2["θ_save"]

for (k, v) in θ_1
    @assert norm(v - θ_2[k]) <= 1e-4 # 500 times of machine precision diff adds up
end

Loss_1 = serial_1["Loss"]
Loss_2 = serial_2["Loss"]

@assert norm(Loss_1 - Loss_2) <= 1e-5

# To test below use backup 8

# x_true_1 = serial_1["x_true"]
# x_true_2 = serial_2["x_true"]

# println(norm(x_true_1 - x_true_2)) 

# y_true_1 = serial_1["y_true"]
# y_true_2 = serial_2["y_true"]

# println(norm(y_true_1 - y_true_2)) 

# y_pred_1 = serial_1["y_pred"]
# y_pred_2 = serial_2["y_pred"]

# println(norm(y_pred_1 - y_pred_2)) 
