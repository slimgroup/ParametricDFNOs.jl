function forward(model::Model, θ, x::Any)
     
    input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nz * model.config.nt) ÷ prod(model.config.partition)
    gpu_flag && (x = x |> gpu)
    # ignore() do
        # GC.gc(true)
    # end
    batch = length(x) ÷ input_size 
    x = reshape(x, (model.config.nc_in, :))
    x = (model.lifts(θ) * x) + model.biases[1](θ)
    # ignore() do
        # GC.gc(true)
    # end

   for i in 1:model.config.nblocks
       x = reshape(x, (:, batch))
       x1 = model.sconvs[i](θ) * x

       x = reshape(x, (model.config.nc_lift, :))
       x2 = (model.convs[i](θ) * x) + model.sconv_biases[i](θ)
    #    ignore() do
        #    GC.gc(true)
    #    end

       x = vec(x1) + vec(x2)
       x = reshape(x, (model.config.nc_lift, model.config.nt * model.config.nx ÷ model.config.partition[1], model.config.ny * model.config.nz ÷ model.config.partition[2], :))

       N = ndims(x)
       ϵ = 1f-5

       reduce_dims = collect(2:N)
       scale = batch * model.config.nx * model.config.ny * model.config.nz * model.config.nt

       s = sum(x; dims=reduce_dims)
       reduce_mean = ParReduce(eltype(s))
       μ = reduce_mean(s) ./ scale

       s = (x .- μ) .^ 2
    #    ignore() do
        #    GC.gc(true)
    #    end

       s = sum(s; dims=reduce_dims)
       reduce_var = ParReduce(eltype(s))
       σ² = reduce_var(s) ./ scale

       x = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    #    ignore() do
        #    GC.gc(true)
    #    end

       if i < model.config.nblocks
           x = relu.(x)
       end
   end
   x = reshape(x, (model.config.nc_lift, :))
#    ignore() do
    #    GC.gc(true)
#    end

   x = (model.projects[1](θ) * x) + model.biases[2](θ)
#    ignore() do
    #    GC.gc(true)
#    end
   x = relu.(x)
#    ignore() do
    #    GC.gc(true)
#    end

   x = (model.projects[2](θ) * x) + model.biases[3](θ)
#    ignore() do
    #    GC.gc(true)
#    end
   x = 1f0.-relu.(1f0.-relu.(x))
#    ignore() do
    #    GC.gc(true)
#    end

   return reshape(x, (model.config.nc_out * model.config.nt * model.config.nx ÷ model.config.partition[1], model.config.ny * model.config.nz ÷ model.config.partition[2], :))
end
