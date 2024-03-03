# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/movies/make_3d_movie.jl 20 5

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("../perlmutter/data.jl")

using .DFNO_3D
using MPI
using ParametricOperators

using PyCall
@pyimport matplotlib.animation as anim
using PyPlot

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]
nblocks, dim, md, mt, samples, isLocal = parse.(Int, ARGS[1:6])

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, mx=md, my=md, mz=md, mt=mt, nblocks=nblocks, partition=partition, dtype=Float64)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# Load Trained Weights
filename = "ep=140_mt=25_mx=10_my=10_mz=10_nblocks=20_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nd=20_nt=51_ntrain=1_nvalid=1_nx=20_ny=20_nz=20_p=8.jld2"
if isLocal == 1
    global filename = "mt=8_mx=4_my=4_mz=4_nblocks=4_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nd=20_nt=51_nx=20_ny=20_nz=20_p=2.jld2"
end

DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

# Use `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5` if not copied to scratch
dataset_path = "/pscratch/sd/r/richardr/v5/$(dim)³"
if isLocal == 1
    global dataset_path = "/Users/richardr2926/Desktop/Research/Code/dfno/data/DFNO_3D/v5/$(dim)³"
end

x_plot, y_plot, _, _ = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), ntrain=samples, nvalid=0)

# # For random loading to test:
# x_plot = rand(modelConfig.dtype, Domain(model.lifts), samples)
# y_plot = rand(modelConfig.dtype, Range(model.projects[2]), samples)

y_predict = DFNO_3D.forward(model, θ, x_plot)

x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))

function showanim(filename)
    base64_video = base64encode(open(filename))
    display("text/html", """<video controls src="data:video/x-m4v;base64,$base64_video">""")
end

fig = figure(figsize=(12, (size(x_plot,6))*4))
fixed_z = modelConfig.nz ÷ 2

function make_z_frame(i)
    for s in 1:size(x_plot,6)
        subplot(size(x_plot,6),4,4*(s-1)+1)
        imshow(x_plot[1,i+1,:,:,fixed_z,s]')
        title("permeability")

        subplot(size(x_plot,6),4,4*(s-1)+2)
        imshow(y_plot[1,i+1,:,:,fixed_z,s]', vmin=0, vmax=1)
        title("true saturation")

        subplot(size(x_plot,6),4,4*(s-1)+3)
        imshow(y_predict[1,i+1,:,:,fixed_z,s]', vmin=0, vmax=1)
        title("predicted saturation")

        subplot(size(x_plot,6),4,4*(s-1)+4)
        imshow(5f0 .* abs.(y_plot[1,i+1,:,:,fixed_z,s]'-y_predict[1,i+1,:,:,fixed_z,s]'), vmin=0, vmax=1)
        title("5X abs difference")
    end
end

myanim = anim.FuncAnimation(fig, make_z_frame, frames=size(y_predict,2), interval=80)
myanim[:save]("movies/DFNO_3D/$(filename[1:end-5])_z_slice_$(dim)³.mp4", bitrate=-1, extra_args=["-vcodec", "-pix_fmt"])

fig = figure(figsize=(12, (size(x_plot,6))*4))
fixed_y = modelConfig.ny ÷ 2

function make_y_frame(i)
    for s in 1:size(x_plot,6)
        subplot(size(x_plot,6),4,4*(s-1)+1)
        imshow(x_plot[1,i+1,:,fixed_y,:,s]')
        title("permeability")

        subplot(size(x_plot,6),4,4*(s-1)+2)
        imshow(y_plot[1,i+1,:,fixed_y,:,s]', vmin=0, vmax=1)
        title("true saturation")

        subplot(size(x_plot,6),4,4*(s-1)+3)
        imshow(y_predict[1,i+1,:,fixed_y,:,s]', vmin=0, vmax=1)
        title("predicted saturation")

        subplot(size(x_plot,6),4,4*(s-1)+4)
        imshow(5f0 .* abs.(y_plot[1,i+1,:,fixed_y,:,s]'-y_predict[1,i+1,:,fixed_y,:,s]'), vmin=0, vmax=1)
        title("5X abs difference")
    end
end


myanim = anim.FuncAnimation(fig, make_y_frame, frames=size(y_predict,2), interval=80)
myanim[:save]("movies/DFNO_3D/$(filename[1:end-5])_y_slice_$(dim)³.mp4", bitrate=-1, extra_args=["-vcodec", "-pix_fmt"])

fig = figure(figsize=(12, (size(x_plot,6))*4))
fixed_x = modelConfig.nx ÷ 2

function make_x_frame(i)
    for s in 1:size(x_plot,6)
        subplot(size(x_plot,6),4,4*(s-1)+1)
        imshow(x_plot[1,i+1,fixed_x,:,:,s]')
        title("permeability")

        subplot(size(x_plot,6),4,4*(s-1)+2)
        imshow(y_plot[1,i+1,fixed_x,:,:,s]', vmin=0, vmax=1)
        title("true saturation")

        subplot(size(x_plot,6),4,4*(s-1)+3)
        imshow(y_predict[1,i+1,fixed_x,:,:,s]', vmin=0, vmax=1)
        title("predicted saturation")

        subplot(size(x_plot,6),4,4*(s-1)+4)
        imshow(5f0 .* abs.(y_plot[1,i+1,fixed_x,:,:,s]'-y_predict[1,i+1,fixed_x,:,:,s]'), vmin=0, vmax=1)
        title("5X abs difference")
    end
end


myanim = anim.FuncAnimation(fig, make_x_frame, frames=size(y_predict,2), interval=80)
myanim[:save]("movies/DFNO_3D/$(filename[1:end-5])_x_slice_$(dim)³.mp4", bitrate=-1, extra_args=["-vcodec", "-pix_fmt"])

MPI.Finalize()
