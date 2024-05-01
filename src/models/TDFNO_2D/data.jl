# @with_kw struct DataConfig
#     ntrain::Int = 1000
#     nvalid::Int = 100
#     perm_key::String = "perm"
#     perm_file::String = datadir(model_name, "perm_gridspacing15.0.jld2")
#     conc_key::String = "conc"
#     conc_file::String = datadir(model_name, "conc_gridspacing15.0.jld2")
#     modelConfig::ModelConfig
# end

# function loadDistData(config::DataConfig;
#     dist_read_x_tensor=UTILS.dist_read_tensor,
#     dist_read_y_tensor=UTILS.dist_read_tensor,
#     comm=MPI.COMM_WORLD)
#     # TODO: maybe move seperating train and valid to trainconfig ? 
#     # TODO: Abstract this for 2D and 3D (dimension agnostic ?) and support uneven partition
#     @assert config.modelConfig.partition[1] == 1 # Creating channel dimension here
#     @assert config.modelConfig.nx * config.modelConfig.ny % config.modelConfig.partition[2] == 0

#     comm_cart = MPI.Cart_create(comm, config.modelConfig.partition)
#     coords = MPI.Cart_coords(comm_cart)
#     rank = MPI.Comm_rank(comm)

#     xy_start, xy_end = UTILS.get_dist_indices(config.modelConfig.nx * config.modelConfig.ny, config.modelConfig.partition[2], coords[2])
#     nt_start, nt_end = 1, config.modelConfig.nt # Contingous along ct

#     x_data = zeros(config.modelConfig.dtype, config.modelConfig.nc_in, nt_end-nt_start+1, xy_end-xy_start+1, config.ntrain+config.nvalid)
#     y_data = zeros(config.modelConfig.dtype, config.modelConfig.nc_out, nt_end-nt_start+1, xy_end-xy_start+1, config.ntrain+config.nvalid)

#     for xy_coord in xy_start:xy_end

#         rank == 0 && println("Loaded coordinate: ", xy_coord - xy_start, " / ", xy_end - xy_start)

#         # 1D index to 2D index. column major julia
#         x_coord = ((xy_coord - 1) % config.modelConfig.ny) + 1
#         y_coord = ((xy_coord - 1) ÷ config.modelConfig.ny) + 1

#         x_indices = (x_coord:x_coord, y_coord:y_coord, 1:config.ntrain+config.nvalid)
#         y_indices = (x_coord:x_coord, y_coord:y_coord, nt_start:nt_end, 1:config.ntrain+config.nvalid)

#         # x_sample is just ntrain + nvalid sized array
#         x_sample = dist_read_x_tensor(config.perm_file, config.perm_key, x_indices)

#         # y_sample is nt x ntrain + nvalid 
#         y_sample = dist_read_y_tensor(config.conc_file, config.conc_key, y_indices)

#         target_zeros = zeros(config.modelConfig.dtype, 1, nt_end-nt_start+1, 1, config.ntrain+config.nvalid)

#         # target is 1,t,xy,n and x_sample is 1,x,y,n since x=y=xy=1, you can do the below
#         x_sample = target_zeros .+ x_sample
#         t_indices = target_zeros .+ reshape(nt_start:nt_end, (1, :, 1, 1))
#         x_indices = target_zeros .+ reshape(x_coord:x_coord, (1, 1, :, 1))
#         y_indices = target_zeros .+ reshape(y_coord:y_coord, (1, 1, :, 1))

#         x_data[:, :, xy_coord-xy_start+1, :] = cat(x_sample, x_indices, y_indices, t_indices, dims=1)
#         y_data[:, :, xy_coord-xy_start+1, :] = y_sample
#     end

#     # combine c and t dim
#     x_data = reshape(x_data, (size(x_data, 1) * size(x_data, 2), size(x_data, 3), size(x_data, 4)))
#     y_data = reshape(y_data, (size(y_data, 1) * size(y_data, 2), size(y_data, 3), size(y_data, 4)))

#     train_indices = (:, :, 1:config.ntrain)
#     valid_indices = (:, :, config.ntrain+1:config.ntrain+config.nvalid)

#     return x_data[train_indices...], y_data[train_indices...], x_data[valid_indices...], y_data[valid_indices...]
# end


# Francis Code x,y,t TODO: Make this dist
# function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32},nt::Int,dt::Float32)
#     tsample = [(i-1)*dt for i = 1:nt]
#     return gen_grid(n, d, tsample)
# end

# function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32},tsample::Vector{Float32})
#     nt = length(tsample)
#     grid = zeros(Float32,n[1],n[2],nt,3)
#     for i = 1:nt     
#         grid[:,:,i,1] = repeat(reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)',n[2])' # x
#         grid[:,:,i,2] = repeat(reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :),n[1])   # z
#         grid[:,:,i,3] .= tsample[i]   # t
#     end
#     return grid
# end

# function perm_to_tensor(x_perm::AbstractMatrix{Float32},grid::Array{Float32,4},AN::ActNorm)
#     # input nx*ny, output nx*ny*nt*4*1
#     nx, ny, nt, _ = size(grid)
#     return cat(reshape(cat([AN(reshape(x_perm, nx, ny, 1, 1))[:,:,1,1] for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
#     reshape(grid, nx, ny, nt, 3, 1), dims=4)
# end

# perm_to_tensor(x_perm::AbstractArray{Float32,3},grid::Array{Float32,4},AN::ActNorm) = cat([perm_to_tensor(x_perm[:,:,i],grid,AN) for i = 1:size(x_perm,3)]..., dims=5)
# ## End Francis Code

# # Returns training and validation data in format cxytn distributed according to partition. TODO: Make this distributed
# function loadData(partition; comm=MPI.COMM_WORLD)
    
#     # TODO: Make this global similar to plot_path
#     mkpath(datadir(model_name))

#     perm_path = datadir(model_name, "perm_gridspacing15.0.mat")
#     conc_path = datadir(model_name, "conc_gridspacing15.0.mat")

#     rank = MPI.Comm_rank(comm)

#     if rank == 0
#         if ~isfile(perm_path)
#             run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
#                 'perm_gridspacing15.0.mat -q -O $perm_path`)
#         end
#         if ~isfile(conc_path)
#             run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
#                 'conc_gridspacing15.0.mat -q -O $conc_path`)
#         end
#     end

#     MPI.Barrier(comm)

#     perm = matread(perm_path)["perm"];

#     ntrain = 1000
#     nvalid = 100

#     n = (64,64)   
#     d = (1f0/64, 1f0/64)

#     s = 1

#     nt = 51
#     dt = 1f0/(nt-1)

#     AN = ActNorm(ntrain)
#     AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

#     grid = gen_grid(n, d, nt, dt)

#     # Following Errors on Machine @ CODA Out of memory SIGKILL 9

#     x_train = permutedims(perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN), [4,1,2,3,5]);
#     x_valid = permutedims(perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN), [4,1,2,3,5]);

#     perm = nothing #  Free the variable for now, TODO: Dist read
#     conc = matread(conc_path)["conc"];

#     y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
#     y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

#     y_train = reshape(y_train, 1, (size(y_train)...))
#     y_valid = reshape(y_valid, 1, (size(y_valid)...))

#     # TODO: Introduce a new operator for future use
#     x_train = UTILS.dist_tensor(x_train, size(x_train), [partition..., 1])
#     y_train = UTILS.dist_tensor(y_train, size(y_train), [partition..., 1])
#     x_valid = UTILS.dist_tensor(x_valid, size(x_valid), [partition..., 1])
#     y_valid = UTILS.dist_tensor(y_valid, size(y_valid), [partition..., 1])

#     return x_train, y_train, x_valid, y_valid
# end

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



###########converter not needed

# function _mat_to_jld2_h5(;comm=MPI.COMM_WORLD)
#     # TODO: Make this global similar to plot_path
#     mkpath(datadir(model_name))

#     perm_path = datadir(model_name, "perm_gridspacing15.0.mat")
#     conc_path = datadir(model_name, "conc_gridspacing15.0.mat")

#     rank = MPI.Comm_rank(comm)

#     if rank == 0
#         if ~isfile(perm_path)
#             run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
#                 'perm_gridspacing15.0.mat -q -O $perm_path`)
#         end
#         if ~isfile(conc_path)
#             run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
#                 'conc_gridspacing15.0.mat -q -O $conc_path`)
#         end
#     end

#     MPI.Barrier(comm)

#     perm = matread(perm_path)["perm"];
#     conc = matread(conc_path)["conc"];

#     println(size(perm))
#     println(size(conc))
    
#     conc = permutedims(conc, [2,3,1,4]) # To get xytc

#     perm_store_path_h5 = datadir(model_name, "perm_gridspacing15.0.h5")
#     conc_store_path_h5 = datadir(model_name, "conc_gridspacing15.0.h5")

#     perm_store_path_jld2 = datadir(model_name, "perm_gridspacing15.0.jld2")
#     conc_store_path_jld2 = datadir(model_name, "conc_gridspacing15.0.jld2")

#     h5write(perm_store_path_h5, "perm", perm)
#     h5write(conc_store_path_h5, "conc", conc)

#     @save perm_store_path_jld2 perm
#     @save conc_store_path_jld2 conc
# end
