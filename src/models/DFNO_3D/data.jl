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
    @assert config.modelConfig.ny * config.modelConfig.nz % config.modelConfig.partition[2] == 0

    comm_cart = MPI.Cart_create(comm, config.modelConfig.partition)
    coords = MPI.Cart_coords(comm_cart)
    rank = MPI.Comm_rank(comm)

    yz_start, yz_end = UTILS.get_dist_indices(config.modelConfig.ny * config.modelConfig.nz, config.modelConfig.partition[2], coords[2])

    # Contingous along ctx
    nt_start, nt_end = 1, config.modelConfig.nt
    nx_start, nx_end = 1, config.modelConfig.nx

    # c,tx,yz,n
    x_data = zeros(config.modelConfig.dtype, config.modelConfig.nc_in, config.modelConfig.nt*config.modelConfig.nx, yz_end-yz_start+1, config.ntrain+config.nvalid)
    y_data = zeros(config.modelConfig.dtype, config.modelConfig.nc_out, config.modelConfig.nt*config.modelConfig.nx, yz_end-yz_start+1, config.ntrain+config.nvalid)

    for yz_coord in yz_start:yz_end
        # 1D index to 2D index. column major julia
        y_coord = ((yz_coord - 1) % config.modelConfig.nz) + 1
        z_coord = ((yz_coord - 1) ÷ config.modelConfig.nz) + 1

        x_indices = (nx_start:nx_end, y_coord:y_coord, z_coord:z_coord, 1:config.ntrain+config.nvalid)
        y_indices = (nx_start:nx_end, y_coord:y_coord, z_coord:z_coord, nt_start:nt_end, 1:config.ntrain+config.nvalid)

        # 1,x,y,z,n -> 1,1,x,yz,n
        x_sample = dist_read_x_tensor(config.perm_file, config.perm_key, x_indices)
        x_sample = reshape(x_sample, (1, 1, size(x_sample, 2), size(x_sample, 3) * size(x_sample, 4), size(x_sample, 5)))

        # 1,x,y,z,t,n -> 1,t,x,y,z,n -> 1,tx,yz,n
        y_sample = dist_read_y_tensor(config.conc_file, config.conc_key, y_indices)
        y_sample = permutedims(y_sample, [1,5,2,3,4,6])
        y_sample = reshape(y_sample, (1, size(y_sample, 2) * size(y_sample, 3), size(y_sample, 4) * size(y_sample, 5), size(y_sample, 6)))

        # 1,t,x,yz,n
        target_zeros = zeros(config.modelConfig.dtype, 1, config.modelConfig.nt, config.modelConfig.nx, 1, config.ntrain+config.nvalid)
        
        if rank == 0
            println(size(target_zeros))

            println(size(x_sample))
            println(size(y_sample))

            println(size(x_data))
            println(size(y_data))

            println(size(x_data[:, :, yz_coord-yz_start+1, :]))
            println(size(y_data[:, :, yz_coord-yz_start+1, :]))
        end

        x_sample = target_zeros .+ x_sample
        t_indices = target_zeros .+ reshape(nt_start:nt_end, (1, :, 1, 1, 1))
        x_indices = target_zeros .+ reshape(nx_start:nx_end, (1, 1, :, 1, 1))
        y_indices = target_zeros .+ reshape(y_coord:y_coord, (1, 1, 1, :, 1))
        z_indices = target_zeros .+ reshape(z_coord:z_coord, (1, 1, 1, :, 1))

        x_data[:, :, yz_coord-yz_start+1, :] = vec(cat(x_sample, t_indices, x_indices, y_indices, z_indices, dims=1))
        y_data[:, :, yz_coord-yz_start+1, :] = vec(y_sample)
        exit()
    end

    # combine ctx dim
    x_data = reshape(x_data, (size(x_data, 1) * size(x_data, 2) * size(x_data, 3), size(x_data, 4), size(x_data, 5)))
    y_data = reshape(y_data, (size(y_data, 1) * size(y_data, 2), size(y_data, 3), size(y_data, 4)))

    train_indices = (:, :, 1:config.ntrain)
    valid_indices = (:, :, config.ntrain+1:config.ntrain+config.nvalid)

    return x_data[train_indices...], y_data[train_indices...], x_data[valid_indices...], y_data[valid_indices...]
end

function loadDistData_old(config::DataConfig;
                        dist_read_x_tensor=UTILS.dist_read_tensor,
                        dist_read_y_tensor=UTILS.dist_read_tensor,
                        comm=MPI.COMM_WORLD)
    # TODO: maybe move seperating train and valid to trainconfig ? 
    # TODO: Abstract this for 2D and 3D (dimension agnostic ?) and support uneven partition
    @assert config.modelConfig.partition[1] == 1 # Creating channel dimension here
    @assert config.modelConfig.nx % config.modelConfig.partition[2] == 0
    @assert config.modelConfig.ny % config.modelConfig.partition[3] == 0
    @assert config.modelConfig.nz % config.modelConfig.partition[4] == 0
    @assert config.modelConfig.nt % config.modelConfig.partition[5] == 0

    comm_cart = MPI.Cart_create(comm, config.modelConfig.partition)
    coords = MPI.Cart_coords(comm_cart)

    nx_start, nx_end = UTILS.get_dist_indices(config.modelConfig.nx, config.modelConfig.partition[2], coords[2])
    ny_start, ny_end = UTILS.get_dist_indices(config.modelConfig.ny, config.modelConfig.partition[3], coords[3])
    nz_start, nz_end = UTILS.get_dist_indices(config.modelConfig.nz, config.modelConfig.partition[4], coords[4])
    nt_start, nt_end = UTILS.get_dist_indices(config.modelConfig.nt, config.modelConfig.partition[5], coords[5])
    
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
