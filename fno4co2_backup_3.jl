# Serial FNO that uses the framework built by thomas 
# and has a different filter for all the points in the restriction space

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
    partition::Vector{Int} = [1]
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

    function spectral_convolution()

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

        input_shape = (config.nc_lift, 2*config.mx, 2*config.my, config.mt)
        weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt)

        input_order = (1, 2, 3, 4)
        weight_order = (5, 1, 2, 3, 4)
        target_order = (5, 2, 3, 4)

        weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape) 

        # Setup FFT-restrict pattern with Kroneckers
        restrict_dft = (restrict_t * fourier_t) ⊗ (restrict_y * fourier_y) ⊗ (restrict_x * fourier_x) ⊗ ParIdentity(T, config.nc_lift)

        sconv = restrict_dft' * weight_mix * restrict_dft

        return sconv
    end

    shape = [config.nc_in, config.nx, config.ny, config.nt_in]

    sconvs = []
    convs = []
    projects = []
    sconv_biases = []
    biases = []

    # Lift Channel dimension
    lifts = ParIdentity(Float32,round(Int, prod(shape)/config.nc_in)) ⊗ ParMatrix(Float32, config.nc_lift, config.nc_in) # lifting(shape, 1, config.nc_lift)
    bias = ParIdentity(Float32,round(Int, prod(shape)/config.nc_in)) ⊗ ParDiagonal(Float32, config.nc_lift) # TODO: Rearrange code for all bias so it makes more sense mathematically
    push!(biases, bias)

    shape[1] = config.nc_lift

    for i in 1:config.n_blocks

        sconv_layer = spectral_convolution()
        conv_layer = ParIdentity(Float32,round(Int, prod(shape)/config.nc_lift)) ⊗ ParMatrix(Float32, config.nc_lift, config.nc_lift) # lifting(shape, 1, config.nc_lift)
        bias = ParIdentity(Float32,round(Int, prod(shape)/config.nc_lift)) ⊗ ParDiagonal(Float32, config.nc_lift)

        push!(sconv_biases, bias)
        push!(sconvs, sconv_layer)
        push!(convs, conv_layer)
    end

    # Uplift channel dimension once more
    uc = ParIdentity(Float32,round(Int, prod(shape)/config.nc_lift)) ⊗ ParMatrix(Float32, config.nc_mid, config.nc_lift) # lifting(shape, 1, config.nc_mid)
    bias = ParIdentity(Float32,round(Int, prod(shape)/config.nc_lift)) ⊗ ParDiagonal(Float32, config.nc_mid)
    push!(biases, bias)
    push!(projects, uc)

    shape[1] = config.nc_mid

    # Project channel dimension
    pc = ParIdentity(Float32,round(Int, prod(shape)/config.nc_mid)) ⊗ ParMatrix(Float32, config.nc_out, config.nc_mid) # lifting(shape, 1, config.nc_out)
    bias = ParIdentity(Float32,round(Int, prod(shape)/config.nc_mid)) ⊗ ParDiagonal(Float32, config.nc_out)
    push!(biases, bias)
    push!(projects, pc)

    shape[1] = config.nc_out

    return lifts, sconvs, convs, projects, biases, sconv_biases
end

modes = 4
width = 20

config = ModelConfig(mx=modes, my=modes, mt=modes, nc_lift=width, n_blocks=4, n_batch=2)
lifts, sconvs, convs, projects, biases, sconv_biases = PO_FNO4CO2(config)

# To Load Saved Dict: 
# key = load("./data/3D_FNO/.jld2")["key"]

θ = init(lifts)
for operator in Iterators.flatten((sconvs, convs, biases, sconv_biases, projects))
    init!(operator, θ)
end

gpu_flag && (global θ = gpu(θ))

function xytcb_to_cxytb(x)
    return permutedims(x, [4,1,2,3,5])
end

function cxytb_to_xytcb(x)
    return permutedims(x, [2,3,4,1,5])
end

function forward(θ, x::Any)
    temp = ones(DDT(biases[1]), Domain(biases[1]), config.n_batch)
    x = lifts(θ) * x + biases[1](θ) * temp

    temp = ones(DDT(sconv_biases[1]), Domain(sconv_biases[1]), config.n_batch)

    for i in 1:config.n_blocks

        x = (sconvs[i](θ) * x) + (convs[i](θ) * x) + (sconv_biases[i](θ) * temp)
        x = cxytb_to_xytcb(reshape(x, (config.nc_lift, config.nx, config.ny, config.nt_in, :)))

        N = ndims(x)
        ϵ = 1f-5

        reduce_dims = [1:N-2; N]

        μ = mean(x; dims=reduce_dims)
        σ² = var(x; mean=μ, dims=reduce_dims, corrected=false)

        prod = config.nc_lift * config.nx * config.ny * config.nt_in

        x = (x .- μ) ./ sqrt.(σ² .+ ϵ)
        x = reshape(xytcb_to_cxytb(x), (prod, :))
        
        if i < config.n_blocks
            x = relu.(x)
        end
    end

    temp = ones(DDT(biases[2]), Domain(biases[2]), config.n_batch)
    x = projects[1](θ) * x + biases[2](θ) * temp
    x = relu.(x)

    temp = ones(DDT(biases[3]), Domain(biases[3]), config.n_batch)
    x = projects[2](θ) * x + biases[3](θ) * temp
    return x
end

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

epochs = 3

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

grid = gen_grid(n, d, nt, dt)

x_train = perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN);
x_valid = perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);
x_valid_dfno = xytcb_to_cxytb(x_valid)

# value, x, y, t

NN = Net3d(modes, width)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(ntrain/batch_size)

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(round(Int, ntrain * epochs / batch_size))

# plot figure
x_plot = x_valid[:, :, :, :, 1:1]
y_plot = y_valid[:, :, :, 1:1]
x_plot_dfno = vec(xytcb_to_cxytb(x_plot))

# Define result directory

sim_name = "3D_FNO"
exp_name = "2phaseflow"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

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
        
        grads_dfno = gradient(params -> norm(relu01(forward(params, x_dfno))-y_dfno)/norm(y_dfno), θ)[1] |> gpu
        global loss = norm(relu01(forward(θ, x_dfno))-y_dfno)/norm(y_dfno)

        # scale!(1e-4, grads_dfno)
        # update(θ, grads_dfno)

        for (k, v) in θ
            Flux.Optimise.update!(opt, v, grads_dfno[k])
        end

        Loss[(ep-1)*nbatches+b] = loss
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

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

θ_save = θ |> cpu

final_dict = @strdict Loss Loss_valid epochs lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)
