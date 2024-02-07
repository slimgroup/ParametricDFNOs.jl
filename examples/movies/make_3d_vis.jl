# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/movies/make_3d_vis.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("../perlmutter/data.jl")

using .DFNO_3D
using MPI
using ParametricOperators
using GLMakie
using Makie

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]
dim, samples = parse.(Int, ARGS[1:2])

@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 4)
modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, mx=modes, my=modes, mz=modes, mt=modes, nblocks=4, partition=partition, dtype=Float32)

# Use `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5` if not copied to scratch
dataset_path = "/pscratch/sd/r/richardr/v5/$(dim)³"

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# Load Trained Weights
# filename = "mt=4_mx=4_my=4_mz=4_nblocks=4_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nt=51_nx=20_ny=20_nz=20.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

# x_plot, y_plot, _, _ = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), n=samples)

# For random loading to test:
x_plot = rand(modelConfig.dtype, Domain(model.lifts), samples)
y_plot = rand(modelConfig.dtype, Range(model.projects[2]), samples)

y_predict = DFNO_3D.forward(model, θ, x_plot)

x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))

#### IMPLEMENT HERE ####

sample_index = 1
time_step = 1
volume_to_visualize = x_plot[1, time_step, :, :, :, sample_index]

# Create a figure for plotting
fig = Figure(resolution = (800, 600))

# Create a 3D scene without axes for the volume slices
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

# Generate linear ranges for each dimension
# Here we assume that the dimensions of your data are indexed as (x, y, z)
x = LinRange(0, size(volume_to_visualize, 1), size(volume_to_visualize, 1))
y = LinRange(0, size(volume_to_visualize, 2), size(volume_to_visualize, 2))
z = LinRange(0, size(volume_to_visualize, 3), size(volume_to_visualize, 3))

# Plot the volume slices in the 3D scene
plt = volumeslices!(ax, x, y, z, volume_to_visualize)

# Create sliders for interactive exploration of the volume slices
lsg = labelslidergrid!(fig, 
    ["x plane - y axis", "x plane - z axis", "y plane - z axis"], 
    [1:length(x), 1:length(y), 1:length(z)]
)
fig[2, 1] = lsg.layout

# Connect sliders to volumeslices' update methods
sl_yz, sl_xz, sl_xy = lsg.sliders

on(sl_yz.value) do v; plt[1].update_yz(v) end
on(sl_xz.value) do v; plt[1].update_xz(v) end
on(sl_xy.value) do v; plt[1].update_xy(v) end

# Set the sliders to be close to the initial slice positions
set_close_to!(sl_yz, .5length(y))
set_close_to!(sl_xz, .5length(x))
set_close_to!(sl_xy, .5length(z))

# # Set the camera to an orthographic projection
# cam3d!(ax.scene, projectiontype=Makie.Orthographic())

# Display the figure
fig

########################

MPI.Finalize()
