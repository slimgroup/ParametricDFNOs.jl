# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/perlmutter/train.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("data.jl")

using .DFNO_3D
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]
epochs, dim, ntrain, nvalid = parse.(Int, ARGS[1:4])

@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 4)
modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, mx=modes, my=modes, mz=modes, mt=modes, nblocks=4, partition=partition, dtype=Float32)

# Use `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5` if not copied to scratch
dataset_path = "/pscratch/sd/r/richardr/v5/$(dim)³"

x_train, y_train, x_valid, y_valid = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), ntrain=ntrain, nvalid=nvalid)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# # To train from a checkpoint
# filename = "mt=4_mx=4_my=4_mz=4_nblocks=4_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nt=51_nx=20_ny=20_nz=20.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

trainConfig = DFNO_3D.TrainConfig(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_train,
    y_valid=y_train,
    plot_every=2
)

DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
