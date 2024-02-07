# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/perlmutter/train.jl

using Pkg
Pkg.activate("./")

include("../../src/models/DFNO_3D/DFNO_3D.jl")
include("data.jl")

using .DFNO_3D
using MPI
using DrWatson

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

partition = [1,pe_count]
epochs, dim, samples = parse.(Int, ARGS[1:3])

@assert MPI.Comm_size(comm) == prod(partition)

modes = max(dim÷8, 4)
modelConfig = DFNO_3D.ModelConfig(nx=dim, ny=dim, nz=dim, mx=modes, my=modes, mz=modes, mt=modes, nblocks=4, partition=partition, dtype=Float32)

# Use `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5` if not copied to scratch
dataset_path = "/pscratch/sd/r/richardr/v5/$(dim)³"

# x_train, y_train, x_valid, y_valid = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), n=samples)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# # To train from a checkpoint
# filename = "ep=1_mt=4_mx=4_my=4_mz=4_nblocks=4_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nt=10_nx=20_ny=20_nz=20_p=1.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

# trainConfig = DFNO_3D.TrainConfig(
#     epochs=epochs,
#     x_train=x_train,
#     y_train=y_train,
#     x_valid=x_train,
#     y_valid=y_train,
#     plot_every=2
# )

# DFNO_3D.train!(trainConfig, model, θ)

model_name = "test"
final_dict = @strdict θ

mkpath(projectdir("weights", model_name))
@tagsave(
    projectdir("weights", model_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=false #OVERWRITES WEIGHTS
)

MPI.Finalize()
