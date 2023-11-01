using MPI
using ParametricOperators
using Parameters

MPI.Init()

@with_kw struct ModelConfig
    nx::Int
    ny::Int
    nz::Int
    nt_in::Int = 1
    nt_out::Int
    nc_in::Int = 1
    nc_out::Int = 1
    nc_lift::Int = 20
    mx::Int
    my::Int
    mz::Int
    mt::Int
    nblocks::Int = 4
    dtype::DataType = Float32
    partition::Vector{Int}
end

function spectral_conv(config::ModelConfig)
    T = config.dtype

    # Build 4D Fourier transform with real-valued FFT along time
    fourier_x = ParDFT(Complex{T}, config.nx)
    fourier_y = ParDFT(Complex{T}, config.ny)
    fourier_z = ParDFT(Complex{T}, config.nz)
    fourier_t = ParDFT(T, config.nt_out)

    # Build restrictions to low-frequency modes
    restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:config.mx, config.nx-config.mx+1:config.nx])
    restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:config.my, config.ny-config.my+1:config.ny])
    restrict_z = ParRestriction(Complex{T}, Range(fourier_z), [1:config.mz, config.nz-config.mz+1:config.nz])
    restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:config.mt])

    # Setup FFT-restrict pattern with Kroneckers
    restrict_dft = (restrict_t * fourier_t) ⊗
                   (restrict_z * fourier_z) ⊗
                   (restrict_y * fourier_y) ⊗
                   (restrict_x * fourier_x) ⊗
                   ParIdentity(T, config.nc_lift)

    # Diagonal/mixing of modes on each channel
    weight_diag = ParDiagonal(Complex{T}, Range(restrict_dft))

    weight_mix = ParIdentity(Complex{T}, Range(weight_diag) ÷ config.nc_lift) ⊗
                 ParMatrix(Complex{T}, config.nc_lift, config.nc_lift)

    # Distribute operators
    weight_mix = distribute(weight_mix, [1, prod(config.partition)])
    weight_diag = distribute(weight_diag)
    restrict_dft = distribute(restrict_dft, config.partition)

    sconv = restrict_dft' * weight_mix * weight_diag * restrict_dft
    return sconv
end

config = ModelConfig(nx=16, ny=16, nz=16, nt_out=32, mx=4, my=4, mz=4, mt=8, partition=[1, 1, 1, 4, 1])
S = spectral_conv(config)

rank = MPI.Comm_rank(MPI.COMM_WORLD)
θ = init(S)
x = rand(DDT(S), Domain(S))
St = S(θ)
y = St*x

MPI.Finalize()

# source $HOME/.bash_profile
# mpiexecjl --project=./ -n 4 julia old_dfno.jl