function _getFigname(config::TrainConfig, additional::Dict)
    nbatch = config.nbatch
    epochs = config.epochs
    ntrain = size(config.x_train, 5)
    nvalid = size(config.x_valid, 5)
    
    figname = @strdict nbatch epochs ntrain nvalid
    return merge(additional,figname)
end

function plotLoss(ep, Loss, Loss_valid, trainConfig::TrainConfig ;additional=Dict())

    ntrain = size(trainConfig.x_train, 5)
    nbatches = Int(ntrain/trainConfig.nbatch)

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))
    subplot(1,3,1)
    plot(loss_train)
    xlabel("batch iterations")
    ylabel("loss")
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(1:nbatches:nbatches*ep, loss_valid);
    xlabel("batch iterations")
    ylabel("loss")
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(1:nbatches:nbatches*ep, loss_valid); 
    xlabel("batch iterations")
    ylabel("loss")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();

    figname = _getFigname(trainConfig, additional)

    safesave(joinpath(plot_path, savename(figname; digits=6)*"_$(model_name)_loss.png"), fig);
    close(fig);
end

function plotEvaluation(modelConfig::ModelConfig, trainConfig::TrainConfig, x_plot, y_plot, y_predict; additional=Dict{String,Any}())

    x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nx, modelConfig.ny, modelConfig.nt))
    y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nx, modelConfig.ny, modelConfig.nt))
    y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nx, modelConfig.ny, modelConfig.nt))

    fig = figure(figsize=(20, 12))

    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[1,:,:,10*i+1]')
        title("x")

        subplot(4,5,i+5)
        imshow(y_plot[1,:,:,10*i+1]', vmin=0, vmax=1)
        title("true y")

        subplot(4,5,i+10)
        imshow(y_predict[1,:,:,10*i+1]', vmin=0, vmax=1)
        title("predict y")

        subplot(4,5,i+15)
        imshow(5f0 .* abs.(y_plot[1,:,:,10*i+1]'-y_predict[1,:,:,10*i+1]'), vmin=0, vmax=1)
        title("5X abs difference")

    end

    tight_layout()
    figname = _getFigname(trainConfig, additional)

    safesave(joinpath(plot_path, savename(figname; digits=6)*"_$(model_name)_fitting.png"), fig);
    close(fig)
end
