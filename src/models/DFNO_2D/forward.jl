"""
    forward(model::Model, θ, x::Any)

Performs the forward pass using the model defined by [`Model`](@ref).

The function applies a series of transformations to the input data `x` using the model parameters `θ` and the configurations within the `model`.

# Arguments
- `model`: The [Model](@ref) object that contains configurations and parameters for the forward pass.
- `θ`: The parameters of the model, likely initialized by `initModel`.
- `x`: Input data that will be passed through the model.

# Returns
The output of the model after the forward pass, reshaped to the dimensions appropriate for the number of output channels, time steps, and spatial dimensions.

# Details
The process includes:
- Reshaping the input and applying lifting operations.
- Processing through a series of blocks that includes spectral and standard convolutions, followed by batch normalization and non-linear activation functions (ReLU).
- Final projection to the output channels and application of a double ReLU operation to finalize the forward pass.

# Notes
This function will move `x` to GPU and perform GPU-enabled computations if `gpu_flag` is set.

# Caution
The input `x` should have the correct number of elements but does not need to have any particular shape.
However, incorrect permutation of the input dimensions will lead to incorrect solution operators.
"""
function forward(model::Model, θ, x::Any)
     
     input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nt) ÷ prod(model.config.partition)
     gpu_flag && (x = x |> gpu)

     batch = length(x) ÷ input_size 
     x = reshape(x, (model.config.nc_in, :))
     x = (model.lifts(θ) * x) + model.biases[1](θ)

    for i in 1:model.config.nblocks
        x = reshape(x, (:, batch))
        x1 = (model.sconvs[i](θ) * x)

        x = reshape(x, (model.config.nc_lift, :))
        x2 = (model.convs[i](θ) * x) + model.sconv_biases[i](θ)

        x = vec(x1) + vec(x2)
        x = reshape(x, (model.config.nc_lift, model.config.nt ÷ model.config.partition[1], model.config.nx * model.config.ny ÷ model.config.partition[2], :))

        N = ndims(x)
        ϵ = 1f-5

        reduce_dims = collect(2:N)
        scale = batch * model.config.nx * model.config.ny * model.config.nt

        s = sum(x; dims=reduce_dims)
        reduce_mean = ParReduce(eltype(s))
        μ = reduce_mean(s) ./ scale

        s = (x .- μ) .^ 2

        s = sum(s; dims=reduce_dims)
        reduce_var = ParReduce(eltype(s))
        σ² = reduce_var(s) ./ scale

        x = (x .- μ) ./ sqrt.(σ² .+ ϵ)

        if i < model.config.nblocks
            x = relu.(x)
        end
    end
    x = reshape(x, (model.config.nc_lift, :))

    x = (model.projects[1](θ) * x) + model.biases[2](θ)
    x = relu.(x)

    x = (model.projects[2](θ) * x) + model.biases[3](θ)
    x = 1f0.-relu.(1f0.-relu.(x))

    return reshape(x, (model.config.nc_out * model.config.nt ÷ model.config.partition[1], model.config.nx * model.config.ny ÷ model.config.partition[2], :))
end
