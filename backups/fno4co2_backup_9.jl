# Serial FNO that uses the framework built by thomas 
# and has a different filter for all the points in the restriction space
# GPU support for everything but Einsum
# Modified backup 4 to train for 1 epoch to make comparison to dfno
# Modified backup 7 to test forward pass
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
gpu = ParametricOperators.gpu
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
    nblocks::Int = 1
    nbatch::Int = 1
    dtype::DataType = Float64
    partition::Vector{Int} = [1]
end

function gpu_stat(input)
    GC.gc(true)
    println("$(input): \n")
    CUDA.memory_status()
    println("\n")
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

        weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape, "ParMatrixN_SCONV:($(layer))") 

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
    lifts = ParIdentity(T,round(Int, prod(shape)/config.nc_in)) ⊗ ParMatrix(T, config.nc_lift, config.nc_in, "ParMatrix_LIFTS:(1)") # lifting(shape, 1, config.nc_lift)
    # bias = ParIdentity(T,round(Int, prod(shape)/config.nc_in)) ⊗ ParDiagonal(T, config.nc_lift, "ParDiagonal_BIAS:(1)") # TODO: Rearrange code for all bias so it makes more sense mathematically
    bias = ParMatrix(T, config.nc_lift, 1, "ParDiagonal_BIAS:(1)")

    push!(biases, bias)

    shape[1] = config.nc_lift

    for i in 1:config.nblocks

        sconv_layer = spectral_convolution(i)
        conv_layer = ParIdentity(T,round(Int, prod(shape)/config.nc_lift)) ⊗ ParMatrix(T, config.nc_lift, config.nc_lift, "ParMatrix_SCONV:($(i))") # lifting(shape, 1, config.nc_lift)
        bias = ParMatrix(T, config.nc_lift, 1, "ParDiagonal_SCONV:($(i))")

        push!(sconv_biases, bias)
        push!(sconvs, sconv_layer)
        push!(convs, conv_layer)
    end

    # Uplift channel dimension once more
    uc = ParIdentity(T,round(Int, prod(shape)/config.nc_lift)) ⊗ ParMatrix(T, config.nc_mid, config.nc_lift, "ParMatrix_LIFTS:(2)") # lifting(shape, 1, config.nc_mid)
    bias = ParMatrix(T, config.nc_mid, 1, "ParDiagonal_BIAS:(2)")
    push!(biases, bias)
    push!(projects, uc)

    shape[1] = config.nc_mid

    # Project channel dimension
    pc = ParIdentity(T,round(Int, prod(shape)/config.nc_mid)) ⊗ ParMatrix(T, config.nc_out, config.nc_mid, "ParMatrix_LIFTS:(3)") # lifting(shape, 1, config.nc_out)
    bias = ParMatrix(T, config.nc_out, 1, "ParDiagonal_BIAS:(3)")
    push!(biases, bias)
    push!(projects, pc)

    shape[1] = config.nc_out

    return lifts, sconvs, convs, projects, biases, sconv_biases
end

function xytcb_to_cxytb(x)
    return permutedims(x, [4,1,2,3,5])
end

function cxytb_to_xytcb(x)
    return permutedims(x, [2,3,4,1,5])
end

function forward(θ, x::Any)

    batch = size(x)[2]
    
    # gpu_stat("Before lifts layer")
    x = reshape(lifts(θ) * x, (config.nc_lift, :))
    x = reshape(x + biases[1](θ), (:, batch))
    # gpu_stat("After lifts layer")
    ignore() do
        GC.gc(true)
    end

    for i in 1:config.nblocks

        # gpu_stat("Before sconv layer")
        x = reshape((sconvs[i](θ) * x) + (convs[i](θ) * x), (config.nc_lift, :))
        x = reshape(x + sconv_biases[i](θ), (:, batch))
        # gpu_stat("After sconv layer")

        x = cxytb_to_xytcb(reshape(x, (config.nc_lift, config.nx, config.ny, config.nt_in, :)))

        N = ndims(x)
        ϵ = 1f-5

        reduce_dims = [1:N-2; N]

        μ = mean(x; dims=reduce_dims)
        σ² = var(x; mean=μ, dims=reduce_dims, corrected=false)

        prod = config.nc_lift * config.nx * config.ny * config.nt_in

        x = (x .- μ) ./ sqrt.(σ² .+ ϵ)
        ignore() do
            GC.gc(true)
            # CUDA.unsafe_free!(μ)
            # CUDA.unsafe_free!(σ²)
        end
        x = reshape(xytcb_to_cxytb(x), (prod, :))
        
        if i < config.nblocks
            x = relu.(x)
        end
    end

    # gpu_stat("Before projects 1 layer")
    ignore() do
        GC.gc(true)
        # gpu_stat("Before projects 1 layer")
    end
    x = reshape(projects[1](θ) * x, (config.nc_mid, :))
    x = reshape(x + biases[2](θ), (:, batch))
    # gpu_stat("After projects 1 layer")
    x = relu.(x)

    # gpu_stat("Before projects 2 layer")
    x = reshape(projects[2](θ) * x, (config.nc_out, :))
    x = reshape(x + biases[3](θ), (:, batch))

    # gpu_stat("After projects 2 layer")
    return x
end

modes = 4
width = 20

config = ModelConfig(mx=modes, my=modes, mt=modes, nc_lift=width, nblocks=1, nbatch=3)
lifts, sconvs, convs, projects, biases, sconv_biases = PO_FNO4CO2(config)

# To Load Saved Dict: 
# key = load("./data/3D_FNO/.jld2")["key"]

θ = init(lifts)
for operator in Iterators.flatten((sconvs, convs, biases, sconv_biases, projects))
    init!(operator, θ)
end

# function reshape_conv_channel(input)
#     length = size(vec(input))[1]
#     return reshape(input, (length, 1, 1, 1, 1)) #cxytb
# end

# for operator in Iterators.flatten((biases, sconv_biases))
#     haskey(θ, operator) && (θ[operator] = reshape_conv_channel(θ[operator]))
# end

gpu_stat("Before loading weights")
gpu_flag && (global θ = gpu(θ))
gpu_stat("After loading weights")
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

ntrain = 990
nvalid = 100

batch_size = config.nbatch
learning_rate = 1f-4

epochs = 200

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

x_train = Float64.(x_train)
y_train = Float64.(y_train)
x_valid = Float64.(x_valid)
y_valid = Float64.(y_valid)

x_valid_dfno = xytcb_to_cxytb(x_valid)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(ntrain/batch_size)

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(round(Int, ntrain * epochs / batch_size))

rng1 = Random.seed!(1234)
valid_idx = randperm(rng1, nvalid)[1]

# plot figure
x_plot = x_valid[:, :, :, :, valid_idx:valid_idx]
y_plot = y_valid[:, :, :, valid_idx:valid_idx]
x_plot_dfno = vec(xytcb_to_cxytb(x_plot))

# Define result directory

sim_name = "3D_FNO"
exp_name = "2phaseflow"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

valid_idx = randperm(nvalid)[1:batch_size]

x_valid_sample = x_valid_dfno[:, :, :, :, valid_idx]
y_valid_sample = y_valid[:, :, :, valid_idx]

# gpu_stat("Before loading sample")
# if gpu_flag
#     global x_valid_sample = x_valid_sample |> gpu
#     global y_valid_sample = y_valid_sample |> gpu
# end
# gpu_stat("After loading sample")
# Loss_valid[1] = norm(relu01(forward(θ, reshape(x_valid_sample, (:, config.nbatch)))) - reshape(y_valid_sample, (:, config.nbatch)))/norm(y_valid_sample)

## training
x_true = nothing
y_true = nothing
y_pred = nothing

for ep = 1:epochs

    Base.flush(Base.stdout)
    rng2 = Random.seed!(1234)
    idx_e = reshape(randperm(rng2, ntrain), batch_size, nbatches)

    for b = 1:nbatches
        x = x_train[:, :, :, :, idx_e[:,b]]
        y = y_train[:, :, :, idx_e[:,b]]

        x_dfno = reshape(xytcb_to_cxytb(x), (:, config.nbatch))
        y_dfno = reshape(y, (:, config.nbatch))
        
        # global x_true = vec(x_dfno)
        # global y_true = vec(y_dfno)
        # global y_pred = vec(relu01(forward(θ, x_dfno)))

        # gpu_stat("Before loading xy batch $(prod(size(x_dfno)) + prod(size(y_dfno)))")
        if gpu_flag
            x_dfno = x_dfno |> gpu
            y_dfno = y_dfno |> gpu
        end
        # gpu_stat("After loading xy batch")

        function compute_grad(params)
            global loss = norm(relu01(forward(params, x_dfno))-y_dfno)/norm(y_dfno)
            return loss
        end
        
        grads_dfno = gradient(params -> compute_grad(params), θ)[1]
        # gpu_stat("After Forward Pass")
        # exit()

        if gpu_flag
            global grads_dfno = Dict(k => gpu(v) for (k, v) in pairs(grads_dfno))
        end

        for (k, v) in θ
            Flux.Optimise.update!(opt, v, grads_dfno[k])
            CUDA.unsafe_free!(grads_dfno[k])
        end

        Loss[(ep-1)*nbatches+b] = loss
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
        # break
    end

    # if gpu_flag
    #     global x_plot_dfno = x_plot_dfno |> gpu
    # end

    # Loss_valid[ep] = norm(relu01(forward(θ, reshape(x_valid_sample, (:, config.nbatch)))) - reshape(y_valid_sample, (:, config.nbatch)))/norm(y_valid_sample)
    # # (ep % 5 > 0) && continue
    # break
    # y_predict = relu01(reshape(forward(θ, vec(x_plot_dfno)), (64,64,51,1))) |> cpu

    # fig = figure(figsize=(20, 12))

    # for i = 1:5
    #     subplot(4,5,i)
    #     imshow(x_plot[:,:,10*i+1,1,1]')
    #     title("x")

    #     subplot(4,5,i+5)
    #     imshow(y_plot[:,:,10*i+1,1]', vmin=0, vmax=1)
    #     title("true y")

    #     subplot(4,5,i+10)
    #     imshow(y_predict[:,:,10*i+1,1]', vmin=0, vmax=1)
    #     title("predict y")

    #     subplot(4,5,i+15)
    #     imshow(5f0 .* abs.(y_plot[:,:,10*i+1,1]'-y_predict[:,:,10*i+1,1]'), vmin=0, vmax=1)
    #     title("5X abs difference")

    # end
    # tight_layout()
    # fig_name = @strdict ep batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
    # safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
    # close(fig)

    # loss_train = Loss[1:ep*nbatches]
    # loss_valid = Loss_valid[1:ep]
    # fig = figure(figsize=(20, 12))
    # subplot(1,3,1)
    # plot(loss_train)
    # xlabel("batch iterations")
    # ylabel("loss")
    # title("training loss at epoch $ep")
    # subplot(1,3,2)
    # plot(1:nbatches:nbatches*ep, loss_valid);
    # xlabel("batch iterations")
    # ylabel("loss")
    # title("validation loss at epoch $ep")
    # subplot(1,3,3)
    # plot(loss_train);
    # plot(1:nbatches:nbatches*ep, loss_valid); 
    # xlabel("batch iterations")
    # ylabel("loss")
    # title("Objective function at epoch $ep")
    # legend(["training", "validation"])
    # tight_layout();
    # safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    # close(fig);

    # # θ_save = θ |> cpu
    # θ_save = Dict(k => cpu(v) for (k, v) in pairs(θ))

    # param_dict = @strdict ep lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid loss_train loss_valid
    # @tagsave(
    #     datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
    #     param_dict;
    #     safe=true
    # )
end

# θ_save = θ |> cpu
θ_save = Dict(k => cpu(v) for (k, v) in pairs(θ))

final_dict = @strdict x_true y_true y_pred Loss Loss_valid epochs lifts sconvs convs projects θ_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)
