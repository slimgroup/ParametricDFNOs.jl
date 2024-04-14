### Custom 3D Time varying FNO 

!!! warning "Not a executable example"
    We provide this example to understand how to use the `loadDistData` function from [2D Data Loading](@ref) and [3D Data Loading](@ref) to perform distributed reading with any custom dataset.

See [Data Partitioning](@ref) for how partitioning is handled.

In this case, the data is stored in the following format:

```
samples/
├── sample1/
│   ├── inputs.jld2
│   └── outputs.jld2
└── sample2/
    ├── inputs.jld2
    └── outputs.jld2
```

We define a distributed read by providing `dist_read_x_tensor` and `dist_read_y_tensor`.

`data.jl`:

```julia
using HDF5
using ParametricDFNOs.DFNO_3D

#### PERLMUTTER Data Loading ####

function read_perlmutter_data(path::String, modelConfig::ModelConfig, rank::Int; ntrain::Int=1000, nvalid::Int=100)

    n = ntrain + nvalid

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
            for t in indices[4]
                data[:, :, :, t - indices[4][1] + 1] = file[times[t]][indices[1:3]...]
            end
        end
        return reshape(data, 1, (size(data)...), 1)
    end
    
    x_train = zeros(modelConfig.dtype, modelConfig.nc_in * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], ntrain)
    y_train = zeros(modelConfig.dtype, modelConfig.nc_out * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], ntrain)
    x_valid = zeros(modelConfig.dtype, modelConfig.nc_in * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], nvalid)
    y_valid = zeros(modelConfig.dtype, modelConfig.nc_out * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], nvalid)

    idx = 1

    for entry in readdir(path; join=true)
        try
            x_file = entry * "/inputs.jld2"
            y_file = entry * "/outputs.jld2"

            dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, 
                                            ntrain=1, 
                                            nvalid=0, 
                                            x_file=x_file,
                                            y_file=y_file,
                                            x_key="K",
                                            y_key="saturations")

            x, y, _, _ = DFNO_3D.loadDistData(dataConfig, 
            dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)

            if idx <= ntrain
                x_train[:,:,idx] = x[:,:,1]
                y_train[:,:,idx] = y[:,:,1]
            else
                x_valid[:,:,idx-ntrain] = x[:,:,1]
                y_valid[:,:,idx-ntrain] = y[:,:,1]
            end
            (rank == 0) && println("Loaded data sample no. $(idx) / $(n)")
            idx == n && break
            idx += 1
        catch e
            (rank == 0) && println("Failed to load data sample no. $(idx). Error: $e")
            continue
        end
    end

    return x_train, y_train, x_valid, y_valid
end
```

We can now use this in our normal training regime such as [Training 2D Time varying FNO](@ref) by doing:

```julia
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

dataset_path = "/pscratch/sd/r/richardr/v5/$(dim)³"

x_train, y_train, x_valid, y_valid = read_perlmutter_data(dataset_path, modelConfig, MPI.Comm_rank(comm), ntrain=ntrain, nvalid=nvalid)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

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
```
