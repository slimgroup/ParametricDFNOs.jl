function forward(model::Model, θ, x::Any)
     
    input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nz * model.config.nt) ÷ prod(model.config.partition)
    gpu_flag && (x = x |> gpu)
    batch = length(x) ÷ input_size

    x = reshape(x, (model.config.nc_in, :))
    x = (model.lifts(θ) * x) + model.biases[1](θ)

   for i in 1:model.config.nblocks
       x1 = (model.convs[i](θ) * x) + model.sconv_biases[i](θ)

       x = reshape(x, (:, batch))
       x2 = reshape(model.sconvs[i](θ) * x, model.config.nc_lift, :)

       x = x1 + x2
       N = ndims(x)
       ϵ = 1f-5

       reduce_dims = collect(2:N)
       scale = batch * model.config.nx * model.config.ny * model.config.nz * model.config.nt

       s = sum(x; dims=reduce_dims)
       reduce_mean = ParReduce(eltype(s))
       μ = reduce_mean(s) ./ scale

       s = (x .- μ) .^ 2

       s = sum(s; dims=reduce_dims)
       reduce_var = ParReduce(eltype(s))
       σ² = reduce_var(s) ./ scale

       # https://arxiv.org/pdf/1502.03167.pdf
       scale = model.γs[i](θ) / sqrt.(σ² .+ ϵ)
       bias = -scale .* μ + model.βs[i](θ)
       println(size(scale), size(bias), size(x))
    #    x = scale .* x .+ bias

       if i < model.config.nblocks
           x = relu.(x)
       end
   end

   x = (model.projects[1](θ) * x) + model.biases[2](θ)
   x = relu.(x)

   x = (model.projects[2](θ) * x) + model.biases[3](θ)

   if model.config.relu01
        x = 1f0.-relu.(1f0.-relu.(x))
   end
   return reshape(x, (model.config.nc_out * model.config.nt * model.config.nx ÷ model.config.partition[1], model.config.ny * model.config.nz ÷ model.config.partition[2], :))
end
