# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia fno4co2.jl
using Pkg
Pkg.activate("./")

using DrWatson
using MPI
using ParametricOperators
using Parameters
using Profile
using Shuffle
using Zygote
using PyPlot
using NNlib
using NNlibCUDA
using FNO4CO2
using JLD2
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
using Random
matplotlib.use("Agg")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

cpu = ParametricOperators.cpu
# gpu = ParametricOperators.gpu
update = ParametricOperators.update!

@with_kw struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nz::Int = 64
    nt_in::Int = 51
    nt_out::Int = 51
    nc_in::Int = 4
    nc_mid::Int = 128
    nc_out::Int = 1
    nc_lift::Int = 20
    mx::Int = 4
    my::Int = 4
    mz::Int = 4
    mt::Int = 4
    n_blocks::Int = 1
    n_batch::Int = 1
    dtype::DataType = Float32
    partition::Vector{Int} = [1, 2, 2, 1]
end

function PO_FNO4CO2(config::ModelConfig)

    T = config.dtype

    function spectral_convolution(layer::Int)

        # Build 4D Fourier transform with real-valued FFT along time
        fourier_x = ParDFT(Complex{T}, config.nx)
        fourier_y = ParDFT(Complex{T}, config.ny)
        # fourier_z = ParDFT(Complex{T}, config.nz)
        fourier_t = ParDFT(T, config.nt_out)

        # Build restrictions to low-frequency modes
        restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:config.mx, config.nx-config.mx+1:config.nx])
        restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:config.my, config.ny-config.my+1:config.ny])
        # restrict_z = ParRestriction(Complex{T}, Range(fourier_z), [1:config.mz, config.nz-config.mz+1:config.nz])
        restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:config.mt])

        input_shape = (config.nc_lift, 2*config.mx, 2*config.my, config.mt)
        weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt)

        input_order = (1, 2, 3, 4)
        weight_order = (5, 1, 2, 3, 4)
        target_order = (5, 2, 3, 4)

        # Setup FFT-restrict pattern and weightage with Kroneckers
        weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape, "ParMatrixN_SCONV:($(layer))")
        restrict_dft = (restrict_t * fourier_t) ⊗ (restrict_y * fourier_y) ⊗ (restrict_x * fourier_x) ⊗ ParIdentity(T, config.nc_lift)

        weight_mix = distribute(weight_mix, config.partition)
        restrict_dft = distribute(restrict_dft, config.partition)

        sconv = restrict_dft' * weight_mix * restrict_dft

        return sconv
    end

    sconvs = []
    convs = []
    projects = []
    sconv_biases = []
    biases = []

    # Lift Channel dimension
    # lifts = ParKron([ParIdentity(T,config.nt_in), ParIdentity(T,config.ny), ParIdentity(T,config.nx), ParMatrix(T, config.nc_lift, config.nc_in)], [4, 3, 2, 1])
    lifts = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParMatrix(T, config.nc_lift, config.nc_in, "ParMatrix_LIFTS:(1)")
    bias = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParDiagonal(T, config.nc_lift, "ParDiagonal_BIAS:(1)") # TODO: Rearrange code for all bias so it makes more sense mathematically

    lifts = distribute(lifts, config.partition)
    bias = distribute(bias, config.partition)

    push!(biases, bias)

    for i in 1:config.n_blocks

        sconv_layer = spectral_convolution(i)
        conv_layer = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParMatrix(T, config.nc_lift, config.nc_lift, "ParMatrix_SCONV:($(i))")
        bias = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParDiagonal(T, config.nc_lift, "ParDiagonal_SCONV:($(i))")

        conv_layer = distribute(conv_layer, config.partition)
        bias = distribute(bias, config.partition)

        push!(sconv_biases, bias)
        push!(sconvs, sconv_layer)
        push!(convs, conv_layer)
    end

    # Uplift channel dimension once more
    uc = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParMatrix(T, config.nc_mid, config.nc_lift, "ParMatrix_LIFTS:(2)")
    bias = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParDiagonal(T, config.nc_mid, "ParDiagonal_BIAS:(2)")

    uc = distribute(uc, config.partition)
    bias = distribute(bias, config.partition)

    push!(biases, bias)
    push!(projects, uc)

    # Project channel dimension
    pc = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParMatrix(T, config.nc_out, config.nc_mid, "ParMatrix_LIFTS:(3)")
    bias = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParDiagonal(T, config.nc_out, "ParDiagonal_BIAS:(3)")

    pc = distribute(pc, config.partition)
    bias = distribute(bias, config.partition)

    push!(biases, bias)
    push!(projects, pc)

    return lifts, sconvs, convs, projects, biases, sconv_biases
end

function xytcb_to_cxytb(x)
    return permutedims(x, [4,1,2,3,5])
end

function forward(θ, x::Any)

    batch = size(x, 2)

    temp = ones(DDT(biases[1]), Domain(biases[1]), batch)
    gpu_flag && (global temp = gpu(temp))
    x = lifts(θ) * x + biases[1](θ) * temp

    temp = ones(DDT(sconv_biases[1]), Domain(sconv_biases[1]), batch)
    gpu_flag && (global temp = gpu(temp))

    for i in 1:config.n_blocks
        
        x = (sconvs[i](θ) * x) + (convs[i](θ) * x) + (sconv_biases[i](θ) * temp)
        x = reshape(x, (config.nc_lift ÷ config.partition[1], config.nx ÷ config.partition[2], config.ny ÷ config.partition[3], config.nt_in ÷ config.partition[4], :))

        N = ndims(x)
        ϵ = 1f-5

        reduce_dims = collect(2:N)
        scale = batch * config.nx * config.ny * config.nt_in

        s = sum(x; dims=reduce_dims)
        reduce_mean = ParReduce(eltype(s))
        μ = reduce_mean(s) ./ scale

        s = (x .- μ) .^ 2

        s = sum(s; dims=reduce_dims)
        reduce_var = ParReduce(eltype(s))
        σ² = reduce_var(s) ./ scale

        input_size = (config.nc_lift * config.nx * config.ny * config.nt_in) ÷ prod(config.partition)

        x = (x .- μ) ./ sqrt.(σ² .+ ϵ)
        x = reshape(x, (input_size, :))
        
        if i < config.n_blocks
            x = relu.(x)
        end
    end

    temp = ones(DDT(biases[2]), Domain(biases[2]), batch)
    gpu_flag && (global temp = gpu(temp))
    x = projects[1](θ) * x + biases[2](θ) * temp
    x = relu.(x)

    temp = ones(DDT(biases[3]), Domain(biases[3]), batch)
    gpu_flag && (global temp = gpu(temp))
    x = projects[2](θ) * x + biases[3](θ) * temp
    x = relu.(x)

    return x
end

function get_local_indices(global_shape, partition, coords)
    indexes = []
    for (dim, value) in enumerate(global_shape)
        local_size = value ÷ partition[dim]
        start = 1 + coords[dim] * local_size

        r = value % partition[dim]

        if coords[dim] < r
            local_size += 1
            start += coords[dim]
        else
            start += r
        end

        push!(indexes, start:start+local_size-1)
    end
    return indexes
end

# Test Code block to test weights on one sample and then save output
function dist_tensor(tensor, global_shape, partition, parent_comm)
    comm_cart = MPI.Cart_create(parent_comm, partition)
    coords = MPI.Cart_coords(comm_cart)

    indexes = get_local_indices(global_shape, partition, coords)
    tensor = reshape(tensor, global_shape)
    return tensor[indexes...]
end

function collect_dist_tensor(local_tensor, global_shape, partition, parent_comm)
    comm_cart = MPI.Cart_create(parent_comm, partition)
    coords = MPI.Cart_coords(comm_cart)

    sparse = zeros(eltype(local_tensor), global_shape...)
    indexes = get_local_indices(global_shape, partition, coords)

    sparse[indexes...] = local_tensor
    return MPI.Reduce(vec(sparse), MPI.SUM, 0, parent_comm)
end

function dist_key(key, partition, comm)

    # # TODO: Comm is not being created correctly for some reason, fix this so root comm is passed
    # comm_cart = MPI.Cart_create(parent_comm, partition)
    # coords = MPI.Cart_coords(comm_cart)

    coords = MPI.Cart_coords(comm)
    return "$(key.id):($(join(coords, ',')))"
end

function dist_value(value, partition, parent_comm)
    partition = [1, partition...]
    return dist_tensor(value, size(value), partition, parent_comm)
end

function loss(local_pred_y, local_true_y)
    s = sum((vec(local_pred_y) - vec(local_true_y)) .^ 2)

    reduce_norm = ParReduce(eltype(local_pred_y))
    reduce_y = ParReduce(eltype(local_true_y))

    norm_diff = √(reduce_norm([s])[1])
    norm_y = √(reduce_y([sum(local_true_y .^ 2)])[1])

    return norm_diff / norm_y
end

modes = 4
width = 20

config = ModelConfig(mx=modes, my=modes, mt=modes, nc_lift=width, n_blocks=4, n_batch=2)
lifts, sconvs, convs, projects, biases, sconv_biases = PO_FNO4CO2(config)

comm_cart = MPI.Cart_create(comm, config.partition)
coords = MPI.Cart_coords(comm_cart)

# To Load Saved Dict: 
# key = load("./data/3D_FNO/.jld2")["key"]

θ = init(lifts)
for operator in Iterators.flatten((sconvs, convs, biases, sconv_biases, projects))
    init!(operator, θ)
end

# # Test Code block to Load existing weights from serially trained FNO
# θ_save = load("./data/3D_FNO/batch_size=2_dt=0.02_ep=85_epochs=250_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2")["θ_save"]
# for (k, v) in θ_save
#     haskey(θ, k) && (θ[k] = v)
#     if !haskey(θ, k)
#         id = dist_key(k, config.partition, comm_cart) # TODO: do not send comm_cart, send parent comm instead

#         for (k1, v1) in θ
#             # Update if distributed key is in the weight dict
#             if k1.id == id
#                 θ[k1] = dist_value(v, config.partition, comm)
#             end
#         end
#     end
# end

# MPI.Finalize()
# exit()

# # Test Code block to load and print missing keys
# θ_save = load("./data/3D_FNO/batch_size=2_dt=0.02_ep=10_epochs=250_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2")["θ_save"]
# for (k, v) in θ
#     (rank == 0) && !haskey(θ_save, k) && println(k)
# end
# exit()

gpu_flag && (global θ = gpu(θ))

# # Test Code block to do a foward pass and loss:
# x = rand(DDT(lifts), Domain(lifts))
# y = rand(RDT(projects[2]), Range(projects[2]))
# forward(θ, x)
# println("Loss: ", loss(forward(θ, x), y))
# exit()

# # Test Code block to check gradient with random input
# rng = Random.seed!(rank)

# x = rand(rng, DDT(lifts), Domain(lifts))
# y = rand(rng, RDT(projects[2]), Range(projects[2]))

# grads_dfno = gradient(params -> loss(forward(params, x), y), θ)[1]

# MPI.Finalize()
# exit()

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
        'perm_gridspacing15.0.mat -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
        'conc_gridspacing15.0.mat -q -O $conc_path`)
end

perm = matread(perm_path)["perm"];
conc = matread(conc_path)["conc"];

nsamples = size(perm, 3)

ntrain = 1000
nvalid = 100

batch_size = config.n_batch
learning_rate = 1f-4

epochs = 250

modes = 4
width = 20

n = (config.nx,config.ny)
#d = (15f0,15f0) # dx, dy in m
d = (1f0/config.nx, 1f0/config.ny)

s = 1

nt = 51
#dt = 20f0    # dt in day
dt = 1f0/(nt-1)

AN = ActNorm(ntrain)
AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);
conc = nothing # Free the variable for now
grid = gen_grid(n, d, nt, dt)

# Following Errors on Machine @ CODA Out of memory SIGKILL 9

x_train = perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN);
x_valid = perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);
perm = nothing # Free the variable for now
x_valid_dfno = xytcb_to_cxytb(x_valid)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(ntrain/batch_size)

Loss = rank == 0 ? zeros(Float32,epochs*nbatches) : nothing
Loss_valid = rank == 0 ? zeros(Float32, epochs + 1) : nothing
prog = rank == 0 ? Progress(round(Int, ntrain * epochs / batch_size)) : nothing

# plot figure
x_plot = x_valid[:, :, :, :, 1:1]
y_plot = y_valid[:, :, :, 1:1]
x_plot_dfno = vec(xytcb_to_cxytb(x_plot))
y_plot_dfno = y_plot

if gpu_flag
    global x_plot_dfno = x_plot_dfno |> gpu
end

# Define result directory

sim_name = "3D_FNO"
exp_name = "2phaseflow"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

valid_idx = randperm(nvalid)[1:batch_size]

x_valid_sample = x_valid_dfno[:, :, :, :, valid_idx]
y_valid_sample = y_valid[:, :, :, valid_idx]

if gpu_flag
    global x_valid_sample = x_valid_sample |> gpu
    global y_valid_sample = y_valid_sample |> gpu
end

# Test block to plot a forward pass

shape_in = (config.nc_in, config.nx, config.ny, config.nt_in)
shape_out = (config.nc_out, config.nx, config.ny, config.nt_out)

local_x_plot_dfno = dist_tensor(x_plot_dfno, shape_in, config.partition, comm)
local_y_plot_dfno = dist_tensor(y_plot_dfno, shape_out, config.partition, comm)

y_local = reshape(forward(θ, vec(local_x_plot_dfno)), (config.nc_out ÷ config.partition[1], config.nx ÷ config.partition[2], config.ny ÷ config.partition[3], config.nt_out ÷ config.partition[4])) |> cpu
y_global = collect_dist_tensor(y_local, shape_out, config.partition, comm)

# # Test code block to compute loss
# l = loss(y_local, local_y_plot_dfno)

if rank > 0
    MPI.Finalize()
    exit()
end

y_predict = reshape(y_global, (64,64,51,1))

fig = figure(figsize=(20, 12))

for i = 1:5
    subplot(4,5,i)
    imshow(x_plot[:,:,10*i+1,1,1]')
    title("x")

    subplot(4,5,i+5)
    imshow(y_plot[:,:,10*i+1,1]', vmin=0, vmax=1)
    title("true y")

    subplot(4,5,i+10)
    imshow(y_predict[:,:,10*i+1,1]', vmin=0, vmax=1)
    title("predict y")

    subplot(4,5,i+15)
    imshow(5f0 .* abs.(y_plot[:,:,10*i+1,1]'-y_predict[:,:,10*i+1,1]'), vmin=0, vmax=1)
    title("5X abs difference")

end

tight_layout()
fig_name = @strdict exp_name
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
close(fig)

MPI.Finalize()
exit()

Loss_valid[1] = norm(forward(θ, reshape(x_valid_sample, (:, config.n_batch))) - reshape(y_valid_sample, (:, config.n_batch)))/norm(y_valid_sample)

## training

for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    for b = 1:nbatches
        x = x_train[:, :, :, :, idx_e[:,b]]
        y = y_train[:, :, :, idx_e[:,b]]

        x_dfno = reshape(xytcb_to_cxytb(x), (:, config.n_batch))
        y_dfno = reshape(y, (:, config.n_batch))

        if gpu_flag
            x_dfno = x_dfno |> gpu
            y_dfno = y_dfno |> gpu
        end

        grads_dfno = gradient(params -> norm(forward(params, x_dfno)-y_dfno)/norm(y_dfno), θ)[1]
        global loss = norm(forward(θ, x_dfno)-y_dfno)/norm(y_dfno)

        if gpu_flag
            global grads_dfno = Dict(k => gpu(v) for (k, v) in pairs(grads_dfno))
        end

        for (k, v) in θ
            Flux.Optimise.update!(opt, v, grads_dfno[k])
        end

        Loss[(ep-1)*nbatches+b] = loss
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    Loss_valid[ep + 1] = norm(forward(θ, reshape(x_valid_sample, (:, config.n_batch))) - reshape(y_valid_sample, (:, config.n_batch)))/norm(y_valid_sample)
    (ep % 5 > 0) && continue

    y_predict = reshape(forward(θ, vec(x_plot_dfno)), (64,64,51,1)) |> cpu

    fig = figure(figsize=(20, 12))

    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[:,:,10*i+1,1,1]')
        title("x")

        subplot(4,5,i+5)
        imshow(y_plot[:,:,10*i+1,1]', vmin=0, vmax=1)
        title("true y")

        subplot(4,5,i+10)
        imshow(y_predict[:,:,10*i+1,1]', vmin=0, vmax=1)
        title("predict y")

        subplot(4,5,i+15)
        imshow(5f0 .* abs.(y_plot[:,:,10*i+1,1]'-y_predict[:,:,10*i+1,1]'), vmin=0, vmax=1)
        title("5X abs difference")

    end
    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
    close(fig)

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep+1]
    fig = figure(figsize=(20, 12))
    subplot(1,3,1)
    plot(loss_train)
    xlabel("batch iterations")
    ylabel("loss")
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(0:nbatches:nbatches*ep, loss_valid);
    xlabel("batch iterations")
    ylabel("loss")
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(0:nbatches:nbatches*ep, loss_valid); 
    xlabel("batch iterations")
    ylabel("loss")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    close(fig);

    # θ_save = θ |> cpu
    θ_save = Dict(k => cpu(v) for (k, v) in pairs(θ))

    param_dict = @strdict ep lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid loss_train loss_valid
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
end

# θ_save = θ |> cpu
θ_save = Dict(k => cpu(v) for (k, v) in pairs(θ))

final_dict = @strdict Loss Loss_valid epochs lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)

MPI.Finalize()
