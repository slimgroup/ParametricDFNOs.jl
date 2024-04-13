function _getFigname(config::TrainConfig, additional::Dict)
    isnothing(config) && return additional

    nbatch = config.nbatch
    epochs = config.epochs
    ntrain = size(config.x_train, 3)
    nvalid = size(config.x_valid, 3)
    
    figname = @strdict nbatch epochs ntrain nvalid
    return merge(additional,figname)
end

function plotLoss(ep, Loss, Loss_valid, trainConfig::TrainConfig ;additional=Dict())

    ntrain = size(trainConfig.x_train, 3)
    nbatches = Int(ntrain/trainConfig.nbatch)

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))

    PyPlot.rc("figure", titlesize=8)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
    PyPlot.rc("axes", labelsize=8)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=8)     # Default fontsize for titles

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

"""
    plotEvaluation(modelConfig::ModelConfig, x_plot, y_plot, y_predict; trainConfig::TrainConfig, additional::Dict{String,Any} = Dict{String,Any}())

Generates a plots comparing the true and predicted values along with the input data and the absolute difference magnified by a factor of 5 along the time dimension.

# Arguments
- `modelConfig`: A [ModelConfig](@ref) struct specifying the dimensions and parameters of the model.
- `x_plot`: Input data to the model.
- `y_plot`: True output data from the model.
- `y_predict`: Predicted output data from the model.
- `trainConfig`: An optional [TrainConfig](@ref) struct containing training configurations, used for constructing the filename for the plot.
- `additional`: An optional dictionary of additional objects that are added to the save file name.

This is a specific plotting function used for a 2 phase fluid flow problem, you can override this by passing your own plotting code that follows this method signature to [`train!`](@ref)
"""
function plotEvaluation(modelConfig::ModelConfig, x_plot, y_plot, y_predict; trainConfig::TrainConfig, additional=Dict{String,Any}())
    spacing = 10

    x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz))
    y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz))
    y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz))

    fig = figure(figsize=(20, 12))
    
    PyPlot.rc("figure", titlesize=8)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
    PyPlot.rc("axes", labelsize=8)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=8)     # Default fontsize for titles


    fixed_z = modelConfig.nz ÷ 2
    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[1,spacing*i+1,:,:,fixed_z]')
        title("input permeability")

        subplot(4,5,i+5)
        imshow(y_plot[1,spacing*i+1,:,:,fixed_z]', vmin=0, vmax=1)
        title("true saturation")

        subplot(4,5,i+10)
        imshow(y_predict[1,spacing*i+1,:,:,fixed_z]', vmin=0, vmax=1)
        title("predicted saturation")

        subplot(4,5,i+15)
        imshow(5f0 .* abs.(y_plot[1,spacing*i+1,:,:,fixed_z]'-y_predict[1,spacing*i+1,:,:,fixed_z]'), vmin=0, vmax=1)
        title("5X abs difference")

    end

    tight_layout()
    figname = _getFigname(trainConfig, additional)

    safesave(joinpath(plot_path, savename(figname; digits=6)*"_$(model_name)_horizontal_fitting.png"), fig);
    close(fig)

    fig = figure(figsize=(20, 12))
    fixed_y = modelConfig.ny ÷ 2

    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[1,spacing*i+1,:,fixed_y,:]')
        title("input permeability")

        subplot(4,5,i+5)
        imshow(y_plot[1,spacing*i+1,:,fixed_y,:]', vmin=0, vmax=1)
        title("true saturation")

        subplot(4,5,i+10)
        imshow(y_predict[1,spacing*i+1,:,fixed_y,:]', vmin=0, vmax=1)
        title("predicted saturation")

        subplot(4,5,i+15)
        imshow(5f0 .* abs.(y_plot[1,spacing*i+1,:,fixed_y,:]'-y_predict[1,spacing*i+1,:,fixed_y,:]'), vmin=0, vmax=1)
        title("5X abs difference")

    end

    tight_layout()
    figname = _getFigname(trainConfig, additional)

    safesave(joinpath(plot_path, savename(figname; digits=6)*"_$(model_name)_vertical_fitting.png"), fig);
    close(fig)
end
