using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
using MPI
using DrWatson
using LinearAlgebra
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,1,1,1]

@assert MPI.Comm_size(comm) == prod(partition)

filename_1 = projectdir("weights", "DFNO_2D", "dtype=Float32_p=1_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2")
filename_2 = projectdir("weights", "DFNO_2D", "dtype=Float32_p=4_mt=4_mx=4_my=4_nblocks=1_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt_in=51_nt_out=51_nx=64_ny=64.jld2")

# serial_1 = load(filename_1)
# serial_2 = load(filename_2)

modelConfig = DFNO_2D.ModelConfig(nblocks=1, partition=partition)
model = DFNO_2D.Model(modelConfig)

θ_1 = DFNO_2D.initModel(model)
θ_2 = DFNO_2D.initModel(model)

# θ_1 = serial_1["θ_save"]
# θ_2 = serial_2["θ_save"]

DFNO_2D.loadWeights!(θ_1, filename_1, "θ_save", partition)
DFNO_2D.loadWeights!(θ_2, filename_2, "θ_save", partition)

x_train, y_train, x_valid, y_valid = DFNO_2D.loadData(partition)
x = x_train[:,:,:,:,1:1]
y = y_train[:,:,:,:,1:1]

y_1 = DFNO_2D.forward(model, θ_1, x)
y_2 = DFNO_2D.forward(model, θ_2, x)

println("P=1 Y out norm: ", norm(y - y_1) / norm(y))
println("P=4 Y out norm: ", norm(y - y_2) / norm(y))

for (k, v) in θ_1
    println(k, " norm: ", norm(v - θ_2[k]))
end

MPI.Finalize()
