using Pkg
Pkg.activate("./")

using DrWatson
using ParametricOperators
using Parameters
using Profile
using Zygote
using PyPlot
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
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
    n_blocks::Int = 1
    n_batch::Int = 1
    dtype::DataType = Float32
    partition::Vector{Int} = [1]
end

modes = 4
width = 20

config = ModelConfig(mx=modes, my=modes, mt=modes, nc_lift=width, n_blocks=4, n_batch=1)

T = config.dtype

function xytcb_to_cxytb(x)
    return permutedims(x, [4,1,2,3,5])
end

function cxytb_to_xytcb(x)
    return permutedims(x, [2,3,4,1,5])
end

lifting = ParIdentity(T, config.nx*config.ny*config.nt_in) ⊗ ParMatrix(T, config.nc_lift, config.nc_in)
θ_new = init(lifting)

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
weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt) # 3 is the no of dimensions including time
target_shape = input_shape

input_order = (1, 2, 3, 4)
weight_order = (5, 1, 2, 3, 4)
target_order = (5, 2, 3, 4)

weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, target_shape) 
init!(weight_mix, θ_new)

# dft = (restrict_t * fourier_t) ⊗
#     (restrict_y * fourier_y) ⊗
#     (restrict_x * fourier_x) ⊗
#     ParIdentity(T, config.nc_lift)

dft = (fourier_t) ⊗
    (fourier_y) ⊗
    (fourier_x) ⊗
    ParIdentity(T, config.nc_lift)

rng = Random.seed!(1234)

x_dfno = rand(rng, T, Domain(lifting)) |> gpu
y_dfno = rand(rng, T, Range(dft)) |> gpu
θ_new = θ_new |> gpu

grads_dfno = gradient(params -> norm(dft * lifting(params) * x_dfno - y_dfno) / norm(y_dfno), θ_new)[1]
# output = cxytb_to_xytcb(reshape(dft' * weight_mix(θ_new) * dft * lifting(θ_new) * x_dfno, (config.nc_lift, config.nx, config.ny, config.nt_out, config.n_batch)));
