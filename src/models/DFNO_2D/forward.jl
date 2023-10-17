function forward(model::Model, θ, x::Any)

    batch = size(x, 2)

    temp = ones(DDT(model.biases[1]), Domain(model.biases[1]), batch)
    gpu_flag && (global temp = gpu(temp))
    x = model.lifts(θ) * x + model.biases[1](θ) * temp

    temp = ones(DDT(model.sconv_biases[1]), Domain(model.sconv_biases[1]), batch)
    gpu_flag && (global temp = gpu(temp))

    for i in 1:model.config.n_blocks
        
        x = (model.sconvs[i](θ) * x) + (model.convs[i](θ) * x) + (model.sconv_biases[i](θ) * temp)
        x = reshape(x, (model.config.nc_lift ÷ model.config.partition[1], model.config.nx ÷ model.config.partition[2], model.config.ny ÷ model.config.partition[3], model.config.nt_in ÷ model.config.partition[4], :))

        N = ndims(x)
        ϵ = 1f-5

        reduce_dims = collect(2:N)
        scale = batch * model.config.nx * model.config.ny * model.config.nt_in

        s = sum(x; dims=reduce_dims)
        reduce_mean = ParReduce(eltype(s))
        μ = reduce_mean(s) ./ scale

        s = (x .- μ) .^ 2

        s = sum(s; dims=reduce_dims)
        reduce_var = ParReduce(eltype(s))
        σ² = reduce_var(s) ./ scale

        input_size = (model.config.nc_lift * model.config.nx * model.config.ny * model.config.nt_in) ÷ prod(model.config.partition)

        x = (x .- μ) ./ sqrt.(σ² .+ ϵ)
        x = reshape(x, (input_size, :))
        
        if i < model.config.n_blocks
            x = relu.(x)
        end
    end

    temp = ones(DDT(model.biases[2]), Domain(model.biases[2]), batch)
    gpu_flag && (global temp = gpu(temp))
    x = model.projects[1](θ) * x + model.biases[2](θ) * temp
    x = relu.(x)

    temp = ones(DDT(model.biases[3]), Domain(model.biases[3]), batch)
    gpu_flag && (global temp = gpu(temp))
    x = model.projects[2](θ) * x + model.biases[3](θ) * temp
    x = relu.(x)

    return x
end
