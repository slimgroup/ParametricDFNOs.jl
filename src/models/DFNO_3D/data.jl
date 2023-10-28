@with_kw struct DataConfig
    ntrain::Int = 1000
    nvalid::Int = 100
    perm_key::String = "perm"
    perm_file::String = datadir(model_name, "perm_gridspacing15.0.jld2")
    conc_key::String = "conc"
    conc_file::String = datadir(model_name, "conc_gridspacing15.0.jld2")
    modelConfig::ModelConfig
end

function loadDistData(config::DataConfig; comm=MPI.COMM_WORLD)
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

    comm_cart = MPI.Cart_create(comm, config.modelConfig.partition)
    coords = MPI.Cart_coords(comm_cart)

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

    function get_dist_tensor(file_name, key, indices)
        data = nothing
        h5open(file_name, "r") do file
            dataset = file[key]
            data = dataset[indices...]
        end
        return reshape(data, 1, (size(data)...))
    end

    nx_start, nx_end = get_dist_indices(config.modelConfig.nx, config.modelConfig.partition[2], coords[2])
    ny_start, ny_end = get_dist_indices(config.modelConfig.ny, config.modelConfig.partition[3], coords[3])
    nz_start, nz_end = get_dist_indices(config.modelConfig.nz, config.modelConfig.partition[4], coords[4])
    nt_start, nt_end = get_dist_indices(config.modelConfig.nt, config.modelConfig.partition[5], coords[5])
    
    x_indices = (nx_start:nx_end, ny_start:ny_end, nz_start:nz_end, 1:config.ntrain+config.nvalid)
    y_indices = (nx_start:nx_end, ny_start:ny_end, nz_start:nz_end, nt_start:nt_end, 1:config.ntrain+config.nvalid)

    x_data = get_dist_tensor(config.perm_file, config.perm_key, x_indices)
    y_data = get_dist_tensor(config.conc_file, config.conc_key, y_indices)

    # x is (1, nx, ny, nz, n) make this (c, nx, ny, nz, nt, n)
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
