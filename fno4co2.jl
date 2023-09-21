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

    function lifting(in_shape, lift_dim, out_features, T=Float32)

        net = ParIdentity(T, 1) 
    
        for dim in eachindex(in_shape)
            if dim == lift_dim
                layer = ParMatrix(T, out_features, in_shape[dim])
            else 
                layer = ParIdentity(T, in_shape[dim])
            end
            
            if dim == 1
                net = layer
            else
                net = layer ⊗ net
            end
        end
    
        return net
    end

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

    # weight_mix = ParIdentity(Complex{T}, Range(restrict_dft) ÷ config.nc_lift) ⊗
    #             ParMatrix(Complex{T}, config.nc_lift, config.nc_lift)

    # Setup FFT-restrict pattern with Kroneckers
    restrict_dft = (restrict_t * fourier_t) ⊗ (restrict_y * fourier_y) ⊗ (restrict_x * fourier_x) ⊗ ParIdentity(T, config.nc_lift)
    rank == 0 && println("Distributing Restrict DFT")
    restrict_dft = distribute(restrict_dft, config.partition)

    function spectral_convolution()

        input_shape = (config.nc_lift, 2*config.mx, 2*config.my, config.mt)
        weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt)

        # Specify Einsum multiplication
        input_order = (1, 2, 3, 4)
        weight_order = (5, 1, 2, 3, 4)
        target_order = (5, 2, 3, 4)

        weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape) 
        rank == 0 && println("Distributing weight_mix")
        weight_mix = distribute(weight_mix, config.partition)

        sconv = restrict_dft' * weight_mix * restrict_dft

        return sconv
    end

    shape = [config.nc_in, config.nx, config.ny, config.nt_in]

    # Lift Channel dimension
    # lifts = ParKron([ParIdentity(Float32, config.nt_in), ParIdentity(Float32, config.ny), ParIdentity(Float32, config.nx), ParMatrix(Float32, config.nc_lift, config.nc_in)], [4, 3, 2, 1])
    lifts = ParIdentity(Float32, config.nt_in) ⊗ ParIdentity(Float32, config.ny) ⊗ ParIdentity(Float32, config.nx) ⊗  ParIdentity(Float32, config.nc_lift) # ⊗ ParMatrix(Float32, config.nc_lift, config.nc_in) # lifting(shape, 1, config.nc_lift)
    rank == 0 && println("Distributing lifts")
    lifts = distribute(lifts, config.partition)
    shape[1] = config.nc_lift

    sconvs = []
    convs = []
    projects = []

    for i in 1:config.n_blocks

        sconv_layer = spectral_convolution()
        conv_layer = ParIdentity(Float32, config.nt_in) ⊗ ParIdentity(Float32, config.ny) ⊗ ParIdentity(Float32, config.nx) ⊗ ParMatrix(Float32, config.nc_lift, config.nc_lift) # lifting(shape, 1, config.nc_lift)
        rank == 0 && println("Distributing conv_layer")
        conv_layer = distribute(conv_layer, config.partition)
        push!(sconvs, sconv_layer)
        push!(convs, conv_layer)
    end

    # Uplift channel dimension once more
    uc = ParIdentity(Float32, config.nt_in) ⊗ ParIdentity(Float32, config.ny) ⊗ ParIdentity(Float32, config.nx) ⊗ ParMatrix(Float32, config.nc_mid, config.nc_lift) # lifting(shape, 1, config.nc_mid)
    rank == 0 && println("Distributing uc")
    uc = distribute(uc, config.partition)
    shape[1] = config.nc_mid
    push!(projects, uc)

    # Project channel dimension
    pc = ParIdentity(Float32, config.nt_in) ⊗ ParIdentity(Float32, config.ny) ⊗ ParIdentity(Float32, config.nx) ⊗ ParMatrix(Float32, config.nc_out, config.nc_mid) # lifting(shape, 1, config.nc_out)
    rank == 0 && println("Distributing pc")
    pc = distribute(pc, config.partition)
    shape[1] = config.nc_out
    push!(projects, pc)

    return lifts, sconvs, convs, projects
end

modes = 4
width = 20

config = ModelConfig(mx=modes, my=modes, mt=modes, nc_lift=width, n_blocks=1, n_batch=2)
println(rank, " was here @ 0")
lifts, sconvs, convs, projects = PO_FNO4CO2(config)

# To Load Saved Dict: 
# key = load("./data/3D_FNO/.jld2")["key"]

θ = init(lifts)
for sconv in sconvs
    init!(sconv, θ)
end
for conv in convs
    init!(conv, θ)
end
init!(projects[1], θ)
init!(projects[2], θ)

gpu_flag && (global θ = gpu(θ))

function xytcb_to_cxytb(x)
    return permutedims(x, [4,1,2,3,5])
end

function cxytb_to_xytcb(x)
    return permutedims(x, [2,3,4,1,5])
end

println(rank, " was here @ 1")

function forward(θ, x::Any)
    println("RxD: ", Range(lifts), " x ", Domain(lifts))
    x = lifts * x
    println("Lifted")
    return x
    for i in 1:config.n_blocks

        x = (sconvs[i](θ) * x) + (convs[i](θ) * x)
        println("Sconv chilling")
        # x = cxytb_to_xytcb(reshape(x, (config.nc_lift, config.nx, config.ny, config.nt_in, :)))

        # N = ndims(x)
        # ϵ = 1f-5

        # reduce_dims = [1:N-2; N]

        # μ = mean(x; dims=reduce_dims)
        # σ² = var(x; mean=μ, dims=reduce_dims, corrected=false)

        # prod = config.nc_lift * config.nx * config.ny * config.nt_in

        # x = (x .- μ) ./ sqrt.(σ² .+ ϵ)
        # x = reshape(xytcb_to_cxytb(x), (prod, :))
        
        if i < config.n_blocks
            x = relu.(x)
        end
        println("Relu Done")
    end

    x = projects[1](θ) * x
    println("Projected 1")
    x = relu.(x)
    x = projects[2](θ) * x
    println("Projected 2")
    return x
end

# # Define raw data directory
# mkpath(datadir("training-data"))
# perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
# conc_path = datadir("training-data", "conc_gridspacing15.0.mat")

# # Download the dataset into the data directory if it does not exist
# if ~isfile(perm_path)
#     run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
#         'perm_gridspacing15.0.mat -q -O $perm_path`)
# end
# if ~isfile(conc_path)
#     run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
#         'conc_gridspacing15.0.mat -q -O $conc_path`)
# end

# perm = matread(perm_path)["perm"];
# conc = matread(conc_path)["conc"];

# nsamples = size(perm, 3)

# ntrain = 1000
# nvalid = 100

# batch_size = config.n_batch
# learning_rate = 1f-4

# epochs = 3

# modes = 4
# width = 20

# n = (config.nx,config.ny)
# #d = (15f0,15f0) # dx, dy in m
# d = (1f0/config.nx, 1f0/config.ny)

# s = 1

# nt = 51
# #dt = 20f0    # dt in day
# dt = 1f0/(nt-1)

# AN = ActNorm(ntrain)
# AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

# y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
# y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

# grid = gen_grid(n, d, nt, dt)

# x_train = perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN);
# x_train_dfno = xytcb_to_cxytb(x_train)
# x_valid = perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);
# x_valid_dfno = xytcb_to_cxytb(x_valid)

# # value, x, y, t

# NN = Net3d(modes, width)
# gpu_flag && (global NN = NN |> gpu)

# Flux.trainmode!(NN, true)
# w = Flux.params(NN)

# opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
# nbatches = Int(ntrain/batch_size)

# Loss = zeros(Float32,epochs*nbatches)
# Loss_valid = zeros(Float32, epochs)
# prog = Progress(round(Int, ntrain * epochs / batch_size))

# # plot figure
# x_plot = x_valid[:, :, :, :, 1:1]
# y_plot = y_valid[:, :, :, 1:1]
# x_plot_dfno = vec(xytcb_to_cxytb(x_plot))

# # Define result directory

# sim_name = "3D_FNO"
# exp_name = "2phaseflow"

# save_dict = @strdict exp_name
# plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

# comm_in  = MPI.Cart_create(comm, config.partition)
# coords = MPI.Cart_coords(comm_in)

# function get_start_end(global_size, rank, size)
#     start_value = 1
#     for r in 1:(rank-1)
#         start_value += local_size(global_size, r, size)
#     end
#     end_value = start_value + local_size(global_size, rank, size) - 1

#     return start_value, end_value
# end
        
# @assert length(config.partition) == 4
# @assert config.partition[1] == 1
# @assert config.nt_in == config.nt_out  # Would need to switch start, ends if t_in != t_out

# # Assuming the partition is only across x, y, t
# x_start, x_end = get_start_end(config.nx, coords[2], config.partition[2])
# y_start, y_end = get_start_end(config.ny, coords[3], config.partition[3])
# t_start, t_end = get_start_end(config.nt_in, coords[4], config.partition[4])

# x_train_dfno = x_train_dfno[:, x_start:x_end, y_start:y_end, t_start:t_end, :]
# y_train = y_train[x_start:x_end, y_start:y_end, t_start:t_end, :]

## training
for ep = 1:1

    println(rank, " was here @ 2")

    # Base.flush(Base.stdout)
    # rng = Random.seed!(1234)
    # idx_e = reshape(randperm(rng, ntrain), batch_size, nbatches)

    for b = 1:1
        # x = x_train_dfno[:, :, :, :, idx_e[:,b]]
        # y = y_train[:, :, :, idx_e[:,b]]

        # x_dfno = reshape(x, (:, config.n_batch))
        # y_dfno = reshape(y, (:, config.n_batch))

        # if gpu_flag
        #     x_dfno = x_dfno |> gpu
        #     y_dfno = y_dfno |> gpu
        # end

        rng = Random.seed!(1234)
        x_dfno = rand(rng, config.dtype, Domain(lifts))
        y_dfno = rand(rng, config.dtype, Range(lifts))
        
        global loss = norm(relu01(forward(θ, x_dfno))-x_dfno)/norm(x_dfno)
        println("Loss: ", loss)
        break
        grads_dfno = gradient(params -> norm(relu01(forward(params, x_dfno))-y_dfno)/norm(y_dfno), θ)[1] |> gpu

        # scale!(1e-4, grads_dfno)
        # update(θ, grads_dfno)

        for (k, v) in θ
            Flux.Optimise.update!(opt, v, grads_dfno[k])
        end

        Loss[(ep-1)*nbatches+b] = loss
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end
    break

    y_predict = relu01(reshape(forward(θ, vec(x_plot_dfno) |> gpu), (64,64,51,1))) |> cpu

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

    θ_save = θ |> cpu

    valid_idx = randperm(nvalid)[1:batch_size]
    Loss_valid[ep] = norm(relu01(forward(θ_save, reshape(x_valid_dfno[:, :, :, :, valid_idx], (:, config.n_batch)))) - reshape(y_valid[:, :, :, valid_idx], (:, config.n_batch)))/norm(y_valid[:, :, :, valid_idx])

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))
    subplot(1,3,1)
    plot(loss_train)
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(1:nbatches:nbatches*ep, loss_valid); 
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(1:nbatches:nbatches*ep, loss_valid); 
    xlabel("iterations")
    ylabel("value")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    close(fig);

    param_dict = @strdict ep lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid loss_train loss_valid
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
end

# θ_save = θ |> cpu

# final_dict = @strdict Loss Loss_valid epochs lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
# @tagsave(
#     datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
#     final_dict;
#     safe=true
# )

MPI.Finalize()

# Probably wrong loss: no expansion over identities
# Loss: 0.8721553
# Loss: 0.8721636
# Loss: 0.8721521
# Loss: 0.87218535

# Wrong Loss: 
# Loss: 1.1671512
# Loss: 1.0335398
# Loss: 1.0310665
# Loss: 1.0532541

# Probably correct Loss: no repartition optimization but expansion over identities to get math right
# Loss: 0.965132
# Loss: 0.9703671
# Loss: 0.9708155
# Loss: 0.97457004

# Optimizing on repartition over identities
# Loss: 0.96874785
# Loss: 0.9687614
# Loss: 0.9687524
# Loss: 0.9687363

# Fake Data with skipping Identity Dims
# Loss: 0.80037093
# Loss: 0.8003314
# Loss: 0.8003537
# Loss: 0.80035543

# Fake data without skipping identity dims
# Loss: Loss: 0.7940493
# 0.7946504
# Loss: Loss: 0.79481375
# 0.7950967

# Fake data without skipping identity but just lift layer
# Loss: 0.84373724
# Loss: 0.84348816
# Loss: 0.84348804
# Loss: 0.843095

# Fake data with skpping identity jsut lift layer
# Loss: 0.84353924
# Loss: 0.84353924
# Loss: 0.84353924
# Loss: 0.84353924
