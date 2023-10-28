# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("src/models/DFNO_3D/DFNO_3D.jl")

using .DFNO_3D
using HLD5
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,1,1,1,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_3D.ModelConfig(nx=20, ny=20, nz=20, nt=55, nblocks=4, partition=partition)

#### PERLMUTTER Data Loading Hack ####

function read_x_tensor(file_name, key, indices)
    # indices for xyzn -> cxyzn where c=n=1 (t gets introduced and broadcasted later)
    data = nothing
    h5open(file_name, "r") do file
        dataset = file[key]
        data = dataset[reverse(indices[1:3])...]
    end
    data = permutedims(data, [3,2,1])
    return reshape(data, 1, (size(data)...), 1)
end

function read_y_tensor(file_name, key, indices)
    # indices for xyztn -> cxyztn where c=n=1
    data = zeros(modelConfig.dtype, map(range -> length(range), reverse(indices[1:4])))
    h5open(file_name, "r") do file
        times = file[key]
        for t in indices[4]
            data[t - indices[4][1] + 1, :, :, :] = times[t][reverse(indices[1:3])...]
        end
    end
    data = permutedims(data, [4,3,2,1])
    return reshape(data, 1, (size(data)...), 1)
end

function read_perlmutter_data(path::String)
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

        x_train, y_train, x_valid, y_valid = DFNO_3D.loadDistData(dataConfig, 
        dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)

        println(size(x_train), size(y_train), size(x_valid), size(y_valid))
        break
    end
end

dataset_path = "/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5/20³"
read_perlmutter_data(dataset_path)

#################################

# model = DFNO_3D.Model(modelConfig)
# θ = DFNO_3D.initModel(model)

# trainConfig = DFNO_3D.TrainConfig(
#     epochs=200,
#     x_train=x_train,
#     y_train=y_train,
#     x_valid=x_valid,
#     y_valid=y_valid,
# )

# DFNO_3D.train!(trainConfig, model, θ)

MPI.Finalize()
