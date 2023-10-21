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

function train!(config::TrainConfig, model::Model, θ::Dict; comm=MPI.COMM_WORLD)

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

    for ep = 1:config.epochs
        rng2 = Random.seed!(config.seed)
        Base.flush(Base.stdout)
        idx_e = reshape(randperm(rng2, config.ntrain), config.nbatch, nbatches)

        for b = 1:nbatches
            x = config.x_train[:, :, :, :, idx_e[:,b]]
            y = config.y_train[:, :, :, :, idx_e[:,b]]
            
            ## TODO: Move x, y to GPU ? 

            grads = gradient(params -> UTILS.dist_loss(forward(model, params, x), y), θ)[1]
            global loss = UTILS.dist_loss(forward(model, θ, x), y)
            
            for (k, v) in θ
                Flux.Optimise.update!(opt, v, grads[k])
            end

            ## TODO: move grads to GPU ?

            rank == 0 && (Loss[(ep-1)*nbatches+b] = loss)
            rank == 0 && ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
        end

        y = forward(model, θ, x_sample)
        loss_valid = UTILS.dist_loss(y, y_sample)

        # TODO: Re-evaluate validation
        rank == 0 && (Loss_valid[ep] = loss_valid)
        ep % config.plot_every > 0 && continue

        x_global_shape = (model.config.nc_in, model.config.nx, model.config.ny, model.config.nt_out)
        y_global_shape = (model.config.nc_out, model.config.nx, model.config.ny, model.config.nt_out)

        y = y[:, :, :, :, 1:1]
        x_sample = x_sample[:, :, :, :, 1:1]
        y_sample = y_sample[:, :, :, :, 1:1]

        y_global = UTILS.collect_dist_tensor(y, y_global_shape, model.config.partition, comm)
        x_sample_global = UTILS.collect_dist_tensor(x_sample, x_global_shape, model.config.partition, comm)
        y_sample_global = UTILS.collect_dist_tensor(y_sample, y_global_shape, model.config.partition, comm)

        # TODO: Better way for name dict? and move weights to cpu before saving and handle rank conditionals better
        labels = @strdict ep Loss_valid Loss
        saveWeights(θ, model, additional=labels, comm=comm)

        rank > 0 && continue
        
        plotEvaluation(model.config, config, x_sample_global, y_sample_global, y_global, additional=labels)
        plotLoss(ep, Loss, Loss_valid, config, additional=labels)
    end
    saveWeights(θ, model, comm=comm)
end
