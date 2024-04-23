"""
    DataConfig

A struct for configuring the data loading process for model training and validation.

# Fields
- `ntrain`: Number of training samples.
- `nvalid`: Number of validation samples.
- `x_key`: Key under which input (X) data is stored in the JLD2 file.
- `x_file`: Path to the file containing input (X) data.
- `y_key`: Key under which output (Y) data is stored in the JLD2 file.
- `y_file`: Path to the file containing output (Y) data.
- `modelConfig`: An instance of [ModelConfig](@ref) that contains model-specific configurations.

# Description
This struct stores paths and keys for data files, along with the counts of training and validation samples,
to facilitate data preparation and loading in a distributed computing environment. It is tightly coupled with the model's configuration,
especially for partitioning the data across different processing nodes.
"""
@with_kw struct DataConfig
    ntrain::Int = 1000
    nvalid::Int = 100
    x_key::Any = "perm"
    x_file::String = datadir(model_name, "perm_gridspacing15.0.jld2")
    y_key::Any = "conc"
    y_file::String = datadir(model_name, "conc_gridspacing15.0.jld2")
    modelConfig::ModelConfig
end

"""
    loadDistData(config::DataConfig; dist_read_x_tensor=UTILS.dist_read_tensor, dist_read_y_tensor=UTILS.dist_read_tensor, comm=MPI.COMM_WORLD)

Loads and distributes training and validation data across processes for distributed training.

# Arguments
- `config`: An instance of [DataConfig](@ref) which holds configuration for data loading.
- `dist_read_x_tensor`: Function to read the distributed x tensors (defaults to [`dist_read_tensor`](@ref)).
- `dist_read_y_tensor`: Function to read the distributed y tensors (defaults to [`dist_read_tensor`](@ref)).
- `comm`: MPI communicator used for distributed data loading (defaults to `MPI.COMM_WORLD`).

# Functionality
- Initializes MPI communication to distribute data according to the model's partitioning scheme.
- Loads input and output data from specified files and keys, and distributes them according to the data partitioning logic defined in the model configuration.
- Prepares and separates the data into training and validation sets.

# Returns
- Four arrays: Training inputs, training outputs, validation inputs, and validation outputs, each formatted for the distributed training process.
- `x_train, y_train, x_valid, y_valid`

This function manages the distribution and partitioning of large datasets across multiple nodes in a parallel computing environment, using MPI for communication. It is essential for ensuring that data is appropriately sliced and distributed to match the computational architecture and memory constraints.
"""
function loadDistData(config::DataConfig;
    dist_read_x_tensor=UTILS.dist_read_tensor,
    dist_read_y_tensor=UTILS.dist_read_tensor,
    comm=MPI.COMM_WORLD,
    print_progress=false)
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

        print_progress && (rank == 0) && println("Loaded coordinate: ", yz_coord - yz_start, " / ", yz_end - yz_start)

        # 1D index to 2D index. column major julia
        y_coord = ((yz_coord - 1) % config.modelConfig.ny) + 1
        z_coord = ((yz_coord - 1) ÷ config.modelConfig.ny) + 1

        x_indices = (nx_start:nx_end, y_coord:y_coord, z_coord:z_coord, 1:config.ntrain+config.nvalid)
        y_indices = (nx_start:nx_end, y_coord:y_coord, z_coord:z_coord, nt_start:nt_end, 1:config.ntrain+config.nvalid)

        data_channels = config.modelConfig.nc_in - 4

        # 1,x,y,z,n -> 1,1,x,yz,n
        x_sample = dist_read_x_tensor(config.x_file, config.x_key, x_indices)
        x_sample = reshape(x_sample, (data_channels, 1, size(x_sample, 2), size(x_sample, 3) * size(x_sample, 4), size(x_sample, 5)))

        # 1,x,y,z,t,n -> 1,t,x,y,z,n -> 1,tx,yz,n
        y_sample = dist_read_y_tensor(config.y_file, config.y_key, y_indices)
        y_sample = permutedims(y_sample, [1,5,2,3,4,6])
        y_sample = reshape(y_sample, (config.modelConfig.nc_out, size(y_sample, 2) * size(y_sample, 3), size(y_sample, 4) * size(y_sample, 5), size(y_sample, 6)))

        # 1,t,x,yz,n # Broadcast for t here
        target_zeros = zeros(config.modelConfig.dtype, data_channels, config.modelConfig.nt, config.modelConfig.nx, 1, config.ntrain+config.nvalid)
        x_sample = target_zeros .+ x_sample

        # 1,t,x,yz,n # Broadcast everything else
        target_zeros = zeros(config.modelConfig.dtype, 1, config.modelConfig.nt, config.modelConfig.nx, 1, config.ntrain+config.nvalid)
        t_indices = target_zeros .+ reshape(nt_start:nt_end, (1, :, 1, 1, 1))
        x_indices = target_zeros .+ reshape(nx_start:nx_end, (1, 1, :, 1, 1))
        y_indices = target_zeros .+ reshape(y_coord:y_coord, (1, 1, 1, :, 1))
        z_indices = target_zeros .+ reshape(z_coord:z_coord, (1, 1, 1, :, 1))

        x_data[:, :, yz_coord-yz_start+1, :] = vec(cat(x_sample, x_indices, y_indices, z_indices, t_indices, dims=1))
        y_data[:, :, yz_coord-yz_start+1, :] = vec(y_sample)
    end

    # combine ctx dim
    x_data = reshape(x_data, (size(x_data, 1) * size(x_data, 2), size(x_data, 3), size(x_data, 4)))
    y_data = reshape(y_data, (size(y_data, 1) * size(y_data, 2), size(y_data, 3), size(y_data, 4)))

    train_indices = (:, :, 1:config.ntrain)
    valid_indices = (:, :, config.ntrain+1:config.ntrain+config.nvalid)

    return x_data[train_indices...], y_data[train_indices...], x_data[valid_indices...], y_data[valid_indices...]
end
