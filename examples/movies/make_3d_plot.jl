# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/movies/make_3d_plot.jl 20 5

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("../perlmutter/data.jl")

using .DFNO_3D
using MPI
using ParametricOperators
using PlotlyJS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]
dim, samples = parse.(Int, ARGS[1:2])

@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 4)
modest = 8
modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, mx=modes, my=modes, mz=modes, mt=modest, nblocks=4, partition=partition, dtype=Float32)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# Load Trained Weights
filename = "mt=8_mx=4_my=4_mz=4_nblocks=4_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nd=20_nt=51_nx=20_ny=20_nz=20_p=2.jld2"
DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

# Use `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5` if not copied to scratch
dataset_path = "/Users/richardr2926/Desktop/Research/Code/dfno/data/DFNO_3D/v5/$(dim)³"
x_plot, y_plot, _, _ = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), ntrain=samples, nvalid=0)

# # For random loading to test:
# x_plot = rand(modelConfig.dtype, Domain(model.lifts), samples)
# y_plot = rand(modelConfig.dtype, Range(model.projects[2]), samples)

y_predict = DFNO_3D.forward(model, θ, x_plot)

x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))

#### IMPLEMENT HERE ####

time_point = 25
channel = 1

# Extracting the 3D slice
data_slice = y_plot[channel, time_point, :, :, :, :]

# Preparing the grid
dim_x, dim_y, dim_z, _ = size(data_slice)
X, Y, Z = mgrid(1:dim_x, 1:dim_y, 1:dim_z)

# Volume rendering
display(plot(volume(
    x=X[:],
    y=Y[:],
    z=Z[:],
    value=data_slice[:],
    isomin=0.2,
    isomax=1,
    opacity=0.5,
    surface_count=17
)))

while true
    sleep(100)
end

########################

MPI.Finalize()
