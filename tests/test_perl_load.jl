using Pkg

Pkg.activate("./")

using MAT
using JLD2
using HDF5
using PyPlot
using Printf
using Random
using DrWatson
using Parameters
using ProgressMeter

@with_kw struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nz::Int = 64
    nt::Int = 51
    nc_in::Int = 4
    nc_mid::Int = 128
    nc_lift::Int = 20
    nc_out::Int = 1
    mx::Int = 4
    my::Int = 4
    mz::Int = 4
    mt::Int = 4
    nblocks::Int = 1
    dtype::DataType = Float32
    partition::Vector{Int} = [1, 2, 2, 2, 1]
end

@with_kw struct DataConfig
    ntrain::Int = 1000
    nvalid::Int = 100
    perm_key::String = "perm"
    perm_file::String = datadir(model_name, "perm_gridspacing15.0.jld2")
    conc_key::String = "conc"
    conc_file::String = datadir(model_name, "conc_gridspacing15.0.jld2")
    modelConfig::ModelConfig
end

function loadDistData(config::DataConfig;
                        dist_read_x_tensor=nothing,
                        dist_read_y_tensor=nothing,
                        # comm=MPI.COMM_WORLD
                        )
    # TODO: maybe move seperating train and valid to trainconfig ? 
    # TODO: Abstract this for 2D and 3D (dimension agnostic ?) and support uneven partition
    @assert config.modelConfig.partition[1] == 1 # Creating channel dimension here
    @assert config.modelConfig.nx % config.modelConfig.partition[2] == 0
    @assert config.modelConfig.ny % config.modelConfig.partition[3] == 0
    @assert config.modelConfig.nz % config.modelConfig.partition[4] == 0
    @assert config.modelConfig.nt % config.modelConfig.partition[5] == 0

    x_train = nothing
    y_train = nothing
    x_valid = nothing
    y_valid = nothing

    # comm_cart = MPI.Cart_create(comm, config.modelConfig.partition)
    # coords = MPI.Cart_coords(comm_cart)
    coords = [0, 0, 0, 0, 0]

    function get_dist_indices(total_size, total_workers, coord)
        # Calculate the base size each worker will handle
        base_size = div(total_size, total_workers)
        
        # Calculate the number of workers that will handle an extra element
        extras = total_size % total_workers
        
        # Determine the start and end indices for the worker
        if coord < extras
            start_index = coord * (base_size + 1) + 1
            end_index = start_index + base_size
        else
            start_index = coord * base_size + extras + 1
            end_index = start_index + base_size - 1
        end
    
        return start_index, end_index
    end

    nx_start, nx_end = get_dist_indices(config.modelConfig.nx, config.modelConfig.partition[2], coords[2])
    ny_start, ny_end = get_dist_indices(config.modelConfig.ny, config.modelConfig.partition[3], coords[3])
    nz_start, nz_end = get_dist_indices(config.modelConfig.nz, config.modelConfig.partition[4], coords[4])
    nt_start, nt_end = get_dist_indices(config.modelConfig.nt, config.modelConfig.partition[5], coords[5])
    
    x_indices = (nx_start:nx_end, ny_start:ny_end, nz_start:nz_end, 1:config.ntrain+config.nvalid)
    y_indices = (nx_start:nx_end, ny_start:ny_end, nz_start:nz_end, nt_start:nt_end, 1:config.ntrain+config.nvalid)

    x_data = dist_read_x_tensor(config.perm_file, config.perm_key, x_indices)
    y_data = dist_read_y_tensor(config.conc_file, config.conc_key, y_indices)

    # x is (c, nx, ny, nz, n) make this (c, nx, ny, nz, nt, n)
    x_data = reshape(x_data, size(x_data, 1), size(x_data, 2), size(x_data, 3), size(x_data, 4), 1, size(x_data, 5))
    target_zeros = zeros(config.modelConfig.dtype, 1, nx_end-nx_start+1, ny_end-ny_start+1, nz_end-nz_start+1, nt_end-nt_start+1, config.ntrain+config.nvalid)

    x_data = target_zeros .+ x_data
    x_indices = target_zeros .+ reshape(nx_start:nx_end, (1, :, 1, 1, 1, 1))
    y_indices = target_zeros .+ reshape(ny_start:ny_end, (1, 1, :, 1, 1, 1))
    z_indices = target_zeros .+ reshape(nz_start:nz_end, (1, 1, 1, :, 1, 1))
    t_indices = target_zeros .+ reshape(nt_start:nt_end, (1, 1, 1, 1, :, 1))

    x_data = cat(x_data, x_indices, y_indices, z_indices, t_indices, dims=1)

    train_indices = (:, :, :, :, :, 1:config.ntrain)
    valid_indices = (:, :, :, :, :, config.ntrain+1:config.ntrain+config.nvalid)

    return x_data[train_indices...], y_data[train_indices...], x_data[valid_indices...], y_data[valid_indices...]
end

partition = [1,1,1,1,1]

modelConfig = ModelConfig(nx=20, ny=20, nz=20, nt=55, nblocks=4, partition=partition)

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

        dataConfig = DataConfig(modelConfig=modelConfig, 
                                        ntrain=1, 
                                        nvalid=0, 
                                        perm_file=perm_file,
                                        conc_file=conc_file,
                                        perm_key="K",
                                        conc_key="saturations")

        x_train, y_train, x_valid, y_valid = loadDistData(dataConfig, 
        dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)

        println(size(x_train), size(y_train), size(x_valid), size(y_valid))
        break
    end
end

dataset_path = "/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5/20³"
read_perlmutter_data(dataset_path)
