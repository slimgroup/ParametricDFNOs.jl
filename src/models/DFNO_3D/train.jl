"""
    TrainConfig

A configuration struct for setting up the training environment.

# Fields
- `nbatch`: The number of samples per batch.
- `epochs`: The number of full training cycles through the dataset.
- `seed`: The random seed for reproducibility of shuffling and initialization.
- `plot_every`: Frequency of plotting the evaluation metrics (every `n` epochs).
- `learning_rate`: The step size for the optimizer.
- `x_train`: Training input data.
- `y_train`: Training target data.
- `x_valid`: Validation input data.
- `y_valid`: Validation target data.

# Usage
This struct is used to encapsulate all necessary training parameters, including data and hyperparameters, to configure the training loop.
"""
@with_kw struct TrainConfig
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

"""
    train!(config::TrainConfig, model::Model, θ::Dict; comm=MPI.COMM_WORLD, plotEval::Function=plotEvaluation)

Conducts the training process for a given model using distributed computing and tracks the training and validation loss.

# Arguments
- `config`: The training configuration settings as specified by the [TrainConfig](@ref) struct.
- `model`: The neural network model to be trained as specified by the [Model](@ref) struct.
- `θ`: A dictionary of model parameters to be optimized during training.
- `comm`: The MPI communicator to be used for distributed training (defaults to `MPI.COMM_WORLD`).
- `plotEval`: The plotting function called to evaluate training progress (defaults to [plotEvaluation](@ref)).

# Training Procedure
- The function sets up the optimizer and initializes training and validation data.
- It goes through the specified number of training epochs, updating model parameters via gradient descent.
- If the `gpu_flag` is set, computations are performed on a GPU.
- At specified intervals, the training process is evaluated by plotting the training and validation losses and predicted vs. actual outputs.

# Outputs
- Updates the model's parameters in-place.
- Produces plots every epoch to visually evaluate model performance.
- Saves weights every `2` epochs

# Notes
- A progress meter will be displayed.
- Make sure to set up [TrainConfig](@ref) properly
"""
function train!(config::TrainConfig, model::Model, θ::Dict; comm=MPI.COMM_WORLD, plotEval=plotEvaluation)

    rank = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)

    ### For Labelling Plots ###

    nblocks = model.config.nblocks
    nc_lift = model.config.nc_lift
    mx = model.config.mx
    my = model.config.my
    mz = model.config.mz
    mt = model.config.mt
    nd = model.config.nx

    ###########################

    ntrain = size(config.x_train, 3)
    nvalid = size(config.x_valid, 3)

    opt = Flux.Optimise.ADAMW(config.learning_rate, (0.9f0, 0.999f0), 1f-4)
    nbatches = Int(ntrain/config.nbatch)

    rng1 = Random.seed!(config.seed)
    valid_idx = randperm(rng1, nvalid)[1:config.nbatch]

    x_sample = config.x_valid[:, :, valid_idx]
    y_sample = config.y_valid[:, :, valid_idx]

    x_sample_cpu = x_sample[:, :, 1:1]
    y_sample_cpu = y_sample[:, :, 1:1]

    x_global_shape = (model.config.nc_in * model.config.nt * model.config.nx, model.config.ny * model.config.nz)
    y_global_shape = (model.config.nc_out * model.config.nt * model.config.nx, model.config.ny * model.config.nz)

    x_sample_global = UTILS.collect_dist_tensor(x_sample_cpu, x_global_shape, model.config.partition, comm)
    y_sample_global = UTILS.collect_dist_tensor(y_sample_cpu, y_global_shape, model.config.partition, comm)

    gpu_flag && (x_sample = x_sample |> gpu)
    gpu_flag && (y_sample = y_sample |> gpu)

    Loss = rank == 0 ? zeros(Float32, config.epochs*nbatches) : nothing
    Loss_valid = rank == 0 ? zeros(Float32, config.epochs) : nothing

    Time_train = rank == 0 ? zeros(Float32, config.epochs*nbatches) : nothing
    Time_overhead = rank == 0 ? zeros(Float32, config.epochs) : nothing

    prog = rank == 0 ? Progress(round(Int, ntrain * config.epochs / config.nbatch)) : nothing

    for ep = 1:config.epochs
        rng2 = Random.seed!(config.seed + ep)
        Base.flush(Base.stdout)
        idx_e = reshape(randperm(rng2, ntrain), config.nbatch, nbatches)

        GC.gc(true)

        for b = 1:nbatches
            time_train = @elapsed begin
                x = config.x_train[:, :, idx_e[:,b]]
                y = config.y_train[:, :, idx_e[:,b]]
                
                gpu_flag && (y = y |> gpu)

                function loss_helper(params)
                    global loss = UTILS.dist_loss(forward(model, params, x), y)
                    return loss
                end

                grads = gradient(params -> loss_helper(params), θ)[1]
                
                for (k, v) in θ
                    Flux.Optimise.update!(opt, v, grads[k])
                end

                rank == 0 && (Loss[(ep-1)*nbatches+b] = loss)
                rank == 0 && ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
            end
            rank == 0 && (Time_train[(ep-1)*nbatches+b] = time_train)
        end

        time_overhead = @elapsed begin
            y = forward(model, θ, x_sample)
            loss_valid = UTILS.dist_loss(y, y_sample)

            # TODO: Re-evaluate validation
            rank == 0 && (Loss_valid[ep] = loss_valid)
            ep % config.plot_every > 0 && continue
            y_cpu = y[:, :, 1:1]
            gpu_flag && (y_cpu = y_cpu |> cpu)
            y_global = UTILS.collect_dist_tensor(y_cpu, y_global_shape, model.config.partition, comm)
        end

        rank == 0 && (Time_overhead[ep] = time_overhead)

        # TODO: Better way for name dict? and move weights to cpu before saving and handle rank conditionals better
        labels = @strdict p ep Loss_valid Loss Time_train Time_overhead nblocks mx my mz mt nd ntrain nvalid nc_lift

        # TODO: control frequency of storage
        (ep % 2 == 0) && saveWeights(θ, model, additional=labels, comm=comm)
        rank > 0 && continue
        
        plotLoss(ep, Loss, Loss_valid, config, additional=labels)
        plotEval(model.config, x_sample_global, y_sample_global, y_global, trainConfig=config, additional=labels)
    end
    labels = @strdict p Loss_valid Loss Time_train Time_overhead nblocks mx my mz mt nd ntrain nvalid nc_lift
    saveWeights(θ, model, additional=labels, comm=comm)
end
