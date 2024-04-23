function _dist_key(key, coords)
    return "$(key.id):($(join(coords, ',')))"
end

function _dist_value(value, partition)
    # This is only for PaarTensor where the first two are the channel dim
    new_partition = [1, 1, partition...]
    return UTILS.dist_tensor(value, size(value), new_partition)
end

"""
    loadWeights!(θ, filename, key, partition; comm=MPI.COMM_WORLD, isLocal=true)

Loads and distributes weights across processes for a parallelized model.

# Arguments
- `θ`: Dictionary of model parameters to be updated with loaded weights.
- `filename`: Name or path of the file containing the saved weights.
- `key`: Key under which the weights are saved in the file.
- `partition`: The partitioning scheme used for distributed tensor weights.
- `comm`: MPI communicator for the distributed system (defaults to `MPI.COMM_WORLD`).
- `isLocal`: Flag indicating whether the file path should be under the generated 'weights' folder. (relative filepath or not)

# Functionality
- Loads weights from a JLD2 file and distributes them according to the partitioning across MPI ranks.
- If `gpu_flag` is set, ensures weights are moved to GPU memory.
"""
function loadWeights!(θ, filename, key, partition; comm=MPI.COMM_WORLD, isLocal=true)
    comm_cart = MPI.Cart_create(comm, partition)
    coords = MPI.Cart_coords(comm_cart)
    rank = MPI.Comm_rank(comm)
    
    file = isLocal ? projectdir("weights", model_name, filename) : filename
    saved = load(file)[key]
    for (k, v) in saved
        haskey(θ, k) && (rank == 0) && (k in keys(θ)) && println("LOADING: ", k)
        haskey(θ, k) && gpu_flag && (θ[k] = v |> gpu)
        haskey(θ, k) && !gpu_flag && (θ[k] = v)
        if !haskey(θ, k)
            id = _dist_key(k, [0, coords...])
            for (k1, v1) in θ
                if k1.id == id
                    rank == 0 && (k1 in keys(θ)) && println("LOADING: ", k1)
                    gpu_flag && (θ[k1] = _dist_value(v, partition) |> gpu)
                    !gpu_flag && (θ[k1] = _dist_value(v, partition))
                end
            end
        end
    end
end

function _collectWeights(θ, model; comm=MPI.COMM_WORLD)
    w_partition = [1, model.config.partition...]

    comm_cart = MPI.Cart_create(comm, w_partition)
    coords = MPI.Cart_coords(comm_cart)
    rank = MPI.Comm_rank(comm)

    gpu_flag && !isnothing(θ) && (θ = Dict(k => cpu(v) for (k, v) in pairs(θ)))

    θ_save = Dict()
    keys_to_remove = []

    for weight_mix in model.weight_mixes
        id = _dist_key(weight_mix, coords)
        for (k, v) in θ
            if k.id == id
                push!(keys_to_remove, k)
                (rank == 0) && println("SAVING: ", weight_mix.id)
                θ_save[weight_mix] = UTILS.collect_dist_tensor(v, weight_mix.weight_shape, [1, w_partition...], comm)
            end
        end
    end

    merge!(θ_save, θ)
    for key in keys_to_remove
        delete!(θ_save, key)
    end

    return θ_save
end

"""
    saveWeights(θ, model::Model; additional=Dict{String,Any}(), comm=MPI.COMM_WORLD)

Saves the current state of the model's weights to a file, only executed by the rank 0 process.

# Arguments
- `θ`: The current state of the model's parameters.
- `model`: The [Model](@ref) instance containing the model configurations.
- `additional`: Include a Dict of strings that you would like your filename to contain and objects your file should contain
- `comm`: The MPI communicator to be used for determining the process rank, can usually be ignored.

# Functionality
- Collects distributed weights from all processes.
- Saves the weights to a JLD2 file with additional metadata.

# Notes
- The file is saved with a unique name generated from model parameters and additional metadata.
"""
function saveWeights(θ, model::Model; additional=Dict{String,Any}(), comm=MPI.COMM_WORLD)

    # TODO: Make this simpler and add more info and remove dependence from model and move to utils
    rank = MPI.Comm_rank(comm)
    θ_save = _collectWeights(θ, model, comm=comm)
    
    rank > 0 && return

    lifts = model.lifts
    sconvs = model.sconvs
    convs = model.convs
    projects = model.projects
    nblocks = model.config.nblocks
    nx = model.config.nx
    ny = model.config.ny
    nt = model.config.nt
    nc_in = model.config.nc_in
    nc_mid = model.config.nc_mid
    nc_lift = model.config.nc_lift
    nc_out = model.config.nc_out
    mx = model.config.mx
    my = model.config.my
    mt = model.config.mt
    partition = model.config.partition
    dtype = model.config.dtype

    final_dict = @strdict lifts sconvs convs projects θ_save nblocks nx ny nt nc_in nc_mid nc_lift nc_out mx my mt partition dtype
    final_dict = merge(final_dict, additional)
    
    mkpath(projectdir("weights", model_name))
    file_path = joinpath("weights", model_name, savename(final_dict, "jld2"; digits=6))
    
    # Temporary fix
    jldopen(file_path, "w") do file
        for (key, value) in final_dict
            file[key] = value
        end
    end

    # TODO: Fix Below for perlmutter
    # @tagsave(
    #     projectdir("weights", model_name, savename(final_dict, "jld2"; digits=6)),
    #     final_dict;
    #     safe=false #OVERWRITES WEIGHTS
    # )
end
