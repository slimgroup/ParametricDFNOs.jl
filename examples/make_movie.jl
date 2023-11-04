# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 1 julia examples/make_movie.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
using MPI

using PyCall
import IJulia
@pyimport matplotlib.animation as anim
using PyPlot

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,1,1,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D.ModelConfig(nblocks=4, partition=partition)
dataConfig = DFNO_2D.DataConfig(modelConfig=modelConfig, ntrain=2, nvalid=0)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

# Load Trained Weights
filename = "mt=4_mx=4_my=4_nblocks=4_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt=51_nx=64_ny=64_p=4.jld2"
DFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

x, y, _, _ = DFNO_2D.loadDistData(dataConfig)
y_pred = DFNO_2D.forward(model, θ, x)

trainConfig = DFNO_2D.TrainConfig(
    x_train=x,
    y_train=y,
    x_valid=x,
    y_valid=y)

function showanim(filename)
    base64_video = base64encode(open(filename))
    display("text/html", """<video controls src="data:video/x-m4v;base64,$base64_video">""")
end

fig = figure(figsize=(12, (size(x,5))*4))

function make_frame(i)
    for s in 1:size(x,5)
        subplot(size(x,5),4,4*(s-1)+1)
        imshow(x[1,:,:,i+1,s]')
        title("permeability")

        subplot(size(x,5),4,4*(s-1)+2)
        imshow(y[1,:,:,i+1,s]', vmin=0, vmax=1)
        title("true saturation")

        subplot(size(x,5),4,4*(s-1)+3)
        imshow(y_pred[1,:,:,i+1,s]', vmin=0, vmax=1)
        title("predicted saturation")

        subplot(size(x,5),4,4*(s-1)+4)
        imshow(5f0 .* abs.(y[1,:,:,i+1,s]'-y_pred[1,:,:,i+1,s]'), vmin=0, vmax=1)
        title("5X abs difference")
    end
end

withfig(fig) do
    myanim = anim.FuncAnimation(fig, make_frame, frames=size(y_pred,4), interval=120)
    myanim[:save]("ML4Seismic_2D.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
end

# showanim("ML4Seismic.mp4") 

MPI.Finalize()
