using Pkg
Pkg.activate("./")

using ParametricOperators

struct Config
    nx
    ny
    nz
    nt
    nc_lift
    mx
    my
    mz
    mt
end

config = Config(64, 32, 1, 1, 1, 4, 4, 1, 1)
T = Float32

# Build 3D Fourier transform with real-valued FFT along time
fourier_x = ParDFT(Complex{T}, config.nx)
fourier_y = ParDFT(Complex{T}, config.ny)
fourier_z = ParDFT(Complex{T}, config.nz)
fourier_t = ParDFT(T, config.nt)

mx = config.mx÷2
my = config.my÷2
mz = config.mz÷2
mt = config.mt

function unique_range(ranges)
    unique_ranges = unique(vcat(ranges...))
    return isempty(unique_ranges) ? [1:1] : ranges
end

# Usage with your ParRestriction instances remains the same
unique_x = unique_range([1:mx, config.nx-mx+1:config.nx])
unique_y = unique_range([1:my, config.ny-my+1:config.ny])
unique_z = unique_range([1:mz, config.nz-mz+1:config.nz])
unique_t = unique_range([1:mt])

println(unique_x)
println(unique_y)
println(unique_z)
println(unique_t)

restrict_x = ParRestriction(Complex{T}, Range(fourier_x), unique_x)
restrict_y = ParRestriction(Complex{T}, Range(fourier_y), unique_y)
restrict_z = ParRestriction(Complex{T}, Range(fourier_z), unique_z)
restrict_t = ParRestriction(Complex{T}, Range(fourier_t), unique_t)

input_shape = (config.nc_lift, config.mt*(config.mx), (config.my)*(config.mz))
weight_shape = (config.nc_lift, config.nc_lift, config.mt*(config.mx), (config.my)*(config.mz))

input_order = (1, 2, 3)
weight_order = (1, 4, 2, 3)
target_order = (4, 2, 3)

# Setup FFT-restrict pattern and weightage with Kroneckers
weight_mix = ParTensor(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape)

restrict_dft_1 = (restrict_z * fourier_z) ⊗ (restrict_y * fourier_y) ⊗ (restrict_x * fourier_x) ⊗ (restrict_t * fourier_t) ⊗ ParIdentity(Complex{T}, config.nc_lift)
restrict_dft_2 = (restrict_y * fourier_y) ⊗ (restrict_x * fourier_x) ⊗ (restrict_t * fourier_t) ⊗ ParIdentity(Complex{T}, config.nc_lift)

println(Range(restrict_dft_1), ":", Range(restrict_dft_2))

sconv_1 = restrict_dft_1' * weight_mix * restrict_dft_1
sconv_2 = restrict_dft_2' * weight_mix * restrict_dft_2

θ_1 = init(sconv_1)
θ_2 = init(sconv_2)

x = vec(rand(T, config.nc_lift, config.nx, config.ny))

y_1 = sconv_1(θ_1)(x)
y_2 = sconv_2(θ_2)(x)

println(norm(y_1 - y_2))

