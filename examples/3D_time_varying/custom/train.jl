using MPI
using CUDA
using ParametricDFNOs.DFNO_3D
using ParametricDFNOs.UTILS

include("data.jl")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

global gpu_flag = parse(Bool, get(ENV, "DFNO_3D_GPU", "0"))
DFNO_3D.set_gpu_flag(gpu_flag)

# Julia requires you to manually assign the gpus, modify to your case.
DFNO_3D.gpu_flag && (CUDA.device!(rank % 4))
partition = [1, pe_count]

nblocks, dim, md, mt, ntrain, nvalid, nbatch, epochs = parse.(Int, ARGS[1:8])

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, mx=md, my=md, mz=md, mt=mt, nblocks=nblocks, partition=partition, dtype=Float32)

# Use `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5` if not copied to scratch
dataset_path = "/pscratch/sd/r/richardr/v5/$(dim)³"

x_train, y_train, x_valid, y_valid = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), ntrain=ntrain, nvalid=nvalid)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# # To train from a checkpoint
# filename = "mt=25_mx=10_my=10_mz=10_nblocks=20_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nd=20_nt=51_nx=20_ny=20_nz=20_p=8.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

trainConfig = DFNO_3D.TrainConfig(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    plot_every=1,
    nbatch=nbatch
)

DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
