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
    dist_read_x_tensor=UTILS.dist_read_tensor,
    dist_read_y_tensor=UTILS.dist_read_tensor,
    comm=MPI.COMM_WORLD)
    # TODO: maybe move seperating train and valid to trainconfig ? 
    # TODO: Abstract this for 2D and 3D (dimension agnostic ?) and support uneven partition
    @assert config.modelConfig.partition[1] == 1 # Creating channel dimension here
    @assert config.modelConfig.nx * config.modelConfig.ny % config.modelConfig.partition[2] == 0

    comm_cart = MPI.Cart_create(comm, config.modelConfig.partition)
    coords = MPI.Cart_coords(comm_cart)
    rank = MPI.Comm_rank(comm)

    xy_start, xy_end = UTILS.get_dist_indices(config.modelConfig.nx * config.modelConfig.ny, config.modelConfig.partition[2], coords[2])
    nt_start, nt_end = 1, config.modelConfig.nt # Contingous along ct

    x_data = zeros(config.modelConfig.dtype, config.modelConfig.nc_in, nt_end-nt_start+1, xy_end-xy_start+1, config.ntrain+config.nvalid)
    y_data = zeros(config.modelConfig.dtype, config.modelConfig.nc_out, nt_end-nt_start+1, xy_end-xy_start+1, config.ntrain+config.nvalid)

    for xy_coord in xy_start:xy_end

        rank == 0 && println("Loaded coordinate: ", xy_coord - xy_start, " / ", xy_end - xy_start)

        # 1D index to 2D index. column major julia
        x_coord = ((xy_coord - 1) % config.modelConfig.ny) + 1
        y_coord = ((xy_coord - 1) ÷ config.modelConfig.ny) + 1

        x_indices = (x_coord:x_coord, y_coord:y_coord, 1:config.ntrain+config.nvalid)
        y_indices = (x_coord:x_coord, y_coord:y_coord, nt_start:nt_end, 1:config.ntrain+config.nvalid)

        # x_sample is just ntrain + nvalid sized array
        data_channels = config.modelConfig.nc_in - 3
        x_sample = dist_read_x_tensor(config.perm_file, config.perm_key, x_indices)
        x_sample = reshape(x_sample, (data_channels, 1, size(x_sample, 2), size(x_sample, 3) * size(x_sample, 4), size(x_sample, 5)))

        # y_sample is nt x ntrain + nvalid 
        y_sample = dist_read_y_tensor(config.conc_file, config.conc_key, y_indices)

        # target is c,t,xy,n and x_sample is c,x,y,n since x=y=xy=1, you can broadcast for t below
        target_zeros = zeros(config.modelConfig.dtype, data_channels, nt_end-nt_start+1, 1, config.ntrain+config.nvalid)
        x_sample = target_zeros .+ x_sample

        # target is 1,t,xy,n and x_sample is 1,x,y,n since x=y=xy=1, you can broadcast for everything else here
        target_zeros = zeros(config.modelConfig.dtype, 1, nt_end-nt_start+1, 1, config.ntrain+config.nvalid)
        t_indices = target_zeros .+ reshape(nt_start:nt_end, (1, :, 1, 1))
        x_indices = target_zeros .+ reshape(x_coord:x_coord, (1, 1, :, 1))
        y_indices = target_zeros .+ reshape(y_coord:y_coord, (1, 1, :, 1))

        x_data[:, :, xy_coord-xy_start+1, :] = vec(cat(x_sample, x_indices, y_indices, t_indices, dims=1))
        y_data[:, :, xy_coord-xy_start+1, :] = vec(y_sample)
    end

    # combine c and t dim
    x_data = reshape(x_data, (size(x_data, 1) * size(x_data, 2), size(x_data, 3), size(x_data, 4)))
    y_data = reshape(y_data, (size(y_data, 1) * size(y_data, 2), size(y_data, 3), size(y_data, 4)))

    train_indices = (:, :, 1:config.ntrain)
    valid_indices = (:, :, config.ntrain+1:config.ntrain+config.nvalid)

    return x_data[train_indices...], y_data[train_indices...], x_data[valid_indices...], y_data[valid_indices...]
end
