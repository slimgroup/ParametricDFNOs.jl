@with_kw struct TrainConfig
    ntrain::Int = 1000
    nvalid::Int = 100
    nbatch::Int = 2
    epochs::Int = 1
    seed::Int = 1234
    plot_every::Int = 1
    learning_rate::Float32 = 1f-4
    x_train::Any
    y_train::Any
    x_valid::Any
    y_valid::Any
end

function _collect_dist_tensor(local_tensor, global_shape, partition, parent_comm)
    comm_cart = MPI.Cart_create(parent_comm, partition)
    coords = MPI.Cart_coords(comm_cart)

    sparse = zeros(eltype(local_tensor), global_shape...)
    indexes = _get_local_indices(global_shape, partition, coords)

    sparse[indexes...] = local_tensor
    return MPI.Reduce(vec(sparse), MPI.SUM, 0, parent_comm)
end

function _loss(local_pred_y, local_true_y)
    s = sum((vec(local_pred_y) - vec(local_true_y)) .^ 2)

    reduce_norm = ParReduce(eltype(local_pred_y))
    reduce_y = ParReduce(eltype(local_true_y))

    norm_diff = √(reduce_norm([s])[1])
    norm_y = √(reduce_y([sum(local_true_y .^ 2)])[1])

    return norm_diff / norm_y
end

function train(config::TrainConfig, model::Model, θ::Dict)

    # TODO: Figure out how to handle comm for the module while making it safe for perlmutter since it has reorder capabilites
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    opt = Flux.Optimise.ADAMW(config.learning_rate, (0.9f0, 0.999f0), 1f-4)
    nbatches = Int(config.ntrain/config.nbatch)

    rng1 = Random.seed!(config.seed)
    valid_idx = randperm(rng1, config.nvalid)[1:config.nbatch]

    x_sample = config.x_valid[:, :, :, :, valid_idx]
    y_sample = config.y_valid[:, :, :, :, valid_idx]

    Loss = rank == 0 ? zeros(Float32,config.epochs*nbatches) : nothing
    Loss_valid = rank == 0 ? zeros(Float32, config.epochs) : nothing
    prog = rank == 0 ? Progress(round(Int, config.ntrain * config.epochs / config.nbatch)) : nothing
    
    rank == 0 && (Loss_valid[1] = 0)

    for ep = 1:config.epochs
        rng2 = Random.seed!(config.seed)
        Base.flush(Base.stdout)
        idx_e = reshape(randperm(rng2, config.ntrain), config.nbatch, nbatches)

        for b = 1:nbatches
            x = config.x_train[:, :, :, :, idx_e[:,b]]
            y = config.y_train[:, :, :, :, idx_e[:,b]]
            
            ## TODO: Move x, y to GPU ? 

            grads = gradient(params -> _loss(forward(model, params, x), y), θ)[1]
            global loss = _loss(forward(model, θ, x), y)
            
            for (k, v) in θ
                Flux.Optimise.update!(opt, v, grads[k])
            end

            ## TODO: move grads to GPU ?

            rank == 0 && (Loss[(ep-1)*nbatches+b] = loss)
            rank == 0 && ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
        end

        y = forward(model, θ, x_sample)
        loss_valid = _loss(y, y_sample)

        # TODO: Re-evaluate validation
        rank == 0 && (Loss_valid[ep] = loss_valid)
        ep % config.plot_every > 0 && continue

        x_global_shape = (model.config.nc_in, model.config.nx, model.config.ny, model.config.nt_out)
        y_global_shape = (model.config.nc_out, model.config.nx, model.config.ny, model.config.nt_out)

        y = y[:, :, :, :, 1:1]
        x_sample = x_sample[:, :, :, :, 1:1]
        y_sample = y_sample[:, :, :, :, 1:1]

        y_global = _collect_dist_tensor(y, y_global_shape, model.config.partition, comm)
        x_sample_global = _collect_dist_tensor(x_sample, x_global_shape, model.config.partition, comm)
        y_sample_global = _collect_dist_tensor(y_sample, y_global_shape, model.config.partition, comm)

        rank > 0 && continue

        ## Plot every some epochs and save weights, images. TODO: Better way for name dict? and move weights to cpu before saving
        labels = @strdict ep
        plotEvaluation(model.config, config, x_sample_global, y_sample_global, y_global, additional=labels)
        plotLoss(ep, Loss, Loss_valid, config, additional=labels)
        # _saveWeights(θ, model, additional=labels)
    end
    # _saveWeights(θ, model)
end
