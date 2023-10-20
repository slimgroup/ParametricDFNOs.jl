# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia main.jl

using Pkg
Pkg.activate("./")

include("../src/models/DFNO_2D/DFNO_2D.jl")

using .DFNO_2D
using MPI
using Random
using Zygote
using DrWatson
using LinearAlgebra
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

partition = [1,2,2,1]

@assert MPI.Comm_size(comm) == prod(partition)

modelConfig = DFNO_2D.ModelConfig(partition=partition)
model = DFNO_2D.Model(modelConfig)

θ = DFNO_2D.initModel(model)
grads_true = DFNO_2D.initModel(model)

comm_cart = MPI.Cart_create(comm, partition)
coords = MPI.Cart_coords(comm_cart)

function dist_key(key)
    return "$(key.id):($(join(coords, ',')))"
end

function dist_value(value)
    new_partition = [1, partition...]
    return dist_tensor(value, size(value), new_partition)
end

function load_saved!(saved, loaded)
    for (k, v) in saved
        haskey(loaded, k) && (loaded[k] = v)
        if !haskey(loaded, k)
            id = dist_key(k)
            for (k1, v1) in loaded
                if k1.id == id
                    loaded[k1] = dist_value(v)
                end
            end
        end
    end    
end

# Load existing stuff from serially trained FNO
θ_save = load("gradient_test/model=serial.jld2")["θ"]
grads_save = load("gradient_test/model=serial.jld2")["grads"]
y_save = load("gradient_test/model=serial.jld2")["y_out"]

load_saved!(grads_save, grads_true)
load_saved!(θ_save, θ)

rng = Random.seed!(1234)

x = rand(rng, DDT(model.lifts), 4*64*64*51)
y = rand(rng, DDT(model.lifts), 1*64*64*51)

shape_in = (4,64,64,51)
shape_out = (1,64,64,51)

x = vec(dist_tensor(x, shape_in, partition))
y = vec(dist_tensor(y, shape_out, partition))
y_true = vec(dist_tensor(y_save, shape_out, partition))

y_out = DFNO_2D.forward(model, θ, x)
y_norm = loss(y_out, y_true)

grads = gradient(params -> loss(DFNO_2D.forward(model, params, x), y), θ)[1]

for (k, v) in grads
    @assert norm(v - grads_true[k]) <= 1e-10
end

@assert y_norm <= 1e-10

MPI.Finalize()
