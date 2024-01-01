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

dim = 20
partition = [1,pe_count]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, nt=10, nblocks=4, partition=partition)

rank == 0 && println("Free memory before loading data: ", Sys.free_memory() / 2^20)

dataset_path = "/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5/$(dim)³"
x_train, y_train, x_valid, y_valid = read_perlmutter_data(dataset_path, modelConfig, n=10)

rank == 0 && println("Free memory after loading data: ", Sys.free_memory() / 2^20)

# model = DFNO_3D.Model(modelConfig)
# θ = DFNO_3D.initModel(model)

trainConfig = DFNO_3D.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    plot_every=2
)

x_sample_cpu = x_train[:, :, 1:1]
y_sample_cpu = y_train[:, :, 1:1]

x_global_shape = (modelConfig.nc_in * modelConfig.nt * modelConfig.nx, modelConfig.ny * modelConfig.nz)
y_global_shape = (modelConfig.nc_out * modelConfig.nt * modelConfig.nx, modelConfig.ny * modelConfig.nz)

x_sample_global = UTILS.collect_dist_tensor(x_sample_cpu, x_global_shape, modelConfig.partition, comm)
y_sample_global = UTILS.collect_dist_tensor(y_sample_cpu, y_global_shape, modelConfig.partition, comm)

if rank > 0
    MPI.Finalize()
    exit()
end

plotEvaluation(modelConfig, trainConfig, x_sample_global, y_sample_global, y_sample_global)
# DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
