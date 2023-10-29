# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("src/models/DFNO_3D/DFNO_3D.jl")

using .DFNO_3D
using HDF5
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,1,1,1,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=80, ny=80, nz=80, nt=55, nblocks=4, partition=partition)

#### PERLMUTTER Data Loading Hack ####

function read_perlmutter_data(path::String, modelConfig::ModelConfig; n::Int=10)

    ntrain = Int64(n ÷ 1.25)
    nvalid = Int64(n ÷ 5)

    function read_x_tensor(file_name, key, indices)
        # indices for xyzn -> cxyzn where c=n=1 (t gets introduced and broadcasted later)
        data = nothing
        h5open(file_name, "r") do file
            dataset = file[key]
            data = dataset[indices[1:3]...]
        end
        return reshape(data, 1, (size(data)...), 1)
    end
    
    function read_y_tensor(file_name, key, indices)
        # indices for xyztn -> cxyztn where c=n=1
        data = zeros(modelConfig.dtype, map(range -> length(range), indices[1:4]))
        h5open(file_name, "r") do file
            times = file[key]
            println(length(times))
            for t in indices[4]
                data[:, :, :, t - indices[4][1] + 1] = file[times[t]][indices[1:3]...]
            end
        end
        return reshape(data, 1, (size(data)...), 1)
    end
    
    x_train = zeros(modelConfig.dtype, modelConfig.nc_in ÷ modelConfig.partition[1], modelConfig.nx ÷ modelConfig.partition[2], modelConfig.ny ÷ modelConfig.partition[3], modelConfig.nz ÷ modelConfig.partition[4], modelConfig.nt ÷ modelConfig.partition[5], ntrain)
    y_train = zeros(modelConfig.dtype, modelConfig.nc_out ÷ modelConfig.partition[1], modelConfig.nx ÷ modelConfig.partition[2], modelConfig.ny ÷ modelConfig.partition[3], modelConfig.nz ÷ modelConfig.partition[4], modelConfig.nt ÷ modelConfig.partition[5], ntrain)
    x_valid = zeros(modelConfig.dtype, modelConfig.nc_in ÷ modelConfig.partition[1], modelConfig.nx ÷ modelConfig.partition[2], modelConfig.ny ÷ modelConfig.partition[3], modelConfig.nz ÷ modelConfig.partition[4], modelConfig.nt ÷ modelConfig.partition[5], nvalid)
    y_valid = zeros(modelConfig.dtype, modelConfig.nc_out ÷ modelConfig.partition[1], modelConfig.nx ÷ modelConfig.partition[2], modelConfig.ny ÷ modelConfig.partition[3], modelConfig.nz ÷ modelConfig.partition[4], modelConfig.nt ÷ modelConfig.partition[5], nvalid)

    idx = 1

    for entry in readdir(path; join=true)

        perm_file = entry * "/inputs.jld2"
        conc_file = entry * "/outputs.jld2"

        dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, 
                                        ntrain=1, 
                                        nvalid=0, 
                                        perm_file=perm_file,
                                        conc_file=conc_file,
                                        perm_key="K",
                                        conc_key="saturations")

        x, y, _, _ = DFNO_3D.loadDistData(dataConfig, 
        dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)

        if idx <= ntrain
            x_train[:,:,:,:,:,idx] = x[:,:,:,:,:,1]
            y_train[:,:,:,:,:,idx] = y[:,:,:,:,:,1]
        else
            x_valid[:,:,:,:,:,idx-ntrain] = x[:,:,:,:,:,1]
            y_valid[:,:,:,:,:,idx-ntrain] = y[:,:,:,:,:,1]
        end
        idx += 1
    end

    return x_train, y_train, x_valid, y_valid
end

dataset_path = "/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5/80³"
x_train, y_train, x_valid, y_valid = read_perlmutter_data(dataset_path, modelConfig)

#################################

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

trainConfig = DFNO_3D.TrainConfig(
    epochs=2,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)

DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
