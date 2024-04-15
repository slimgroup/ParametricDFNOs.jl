"""
    ModelConfig(;nx::Int=64, ny::Int=64, nz::Int=64, nt::Int=51, nc_in::Int=5, nc_mid::Int=128, nc_lift::Int=20, nc_out::Int=1, mx::Int=8, my::Int=8, mz::Int=8, mt::Int=4, nblocks::Int=4, dtype::DataType=Float32, partition::Vector{Int}=[1, 8], relu01::Bool=true)

A configuration struct that holds parameters for the [Model](@ref) setup. 

# Fields
- `nx`: The discretization along the x-dimension.
- `ny`: The discretization along the y-dimension.
- `nz`: The discretization along the z-dimension.
- `nt`: The discretization along time.
- `nc_in`: The number of input channels.
- `nc_mid`: The number of intermediate channels.
- `nc_lift`: The number of lifted channels.
- `nc_out`: The number of output channels.
- `mx`: The size of the model in the x-dimension for Fourier transform.
- `my`: The size of the model in the y-dimension for Fourier transform.
- `mz`: The size of the model in the z-dimension for Fourier transform.
- `mt`: The size of the model in time for Fourier transform.
- `nblocks`: The number of blocks in the model.
- `dtype`: The data type used for computations, default is `Float32`.
- `partition`: The partitioning configuration for distributed computing, default is `[1, 8]`.
- `relu01`: Boolean that determines whether the final relu layer is applied
"""
@with_kw struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nz::Int = 64
    nt::Int = 51
    nc_in::Int = 5
    nc_mid::Int = 128
    nc_lift::Int = 20
    nc_out::Int = 1
    mx::Int = 8
    my::Int = 8
    mz::Int = 8
    mt::Int = 4
    nblocks::Int = 4
    dtype::DataType = Float32
    partition::Vector{Int} = [1, 8]
    relu01::Bool = true
end

"""
    Model(config::ModelConfig)

A mutable structure representing the FNO, including configurations, weights, biases, and other parameters.

# Constructor
This constructor initializes the model's layers and distributes the weights and biases according to the provided [ModelConfig](@ref).
"""
mutable struct Model
    config::ModelConfig
    lifts::Any
    convs::Vector
    sconvs::Vector
    biases::Vector
    sconv_biases::Vector
    projects::Vector
    weight_mixes::Vector

    function Model(config::ModelConfig)

        T = config.dtype
        
        sconvs = []
        convs = []
        projects = []
        sconv_biases = []
        biases = []
        weight_mixes = []

        mt = config.mt
        mx = config.mx÷2
        my = config.my÷2
        mz = config.mz÷2
    
        function spectral_convolution(layer::Int)
    
            # Build 3D Fourier transform with real-valued FFT along time
            fourier_x = ParDFT(Complex{T}, config.nx)
            fourier_y = ParDFT(Complex{T}, config.ny)
            fourier_z = ParDFT(Complex{T}, config.nz)
            fourier_t = ParDFT(T, config.nt)

            # Build restrictions to low-frequency modes
            restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:mx, config.nx-mx+1:config.nx])
            restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:my, config.ny-my+1:config.ny])
            restrict_z = ParRestriction(Complex{T}, Range(fourier_z), [1:mz, config.nz-mz+1:config.nz])
            restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:mt])
    
            input_shape = (config.nc_lift, config.mt*config.mx, config.my*config.mz)
            weight_shape = (config.nc_lift, config.nc_lift, config.mt*config.mx, config.my*config.mz)
    
            input_order = (1, 2, 3)
            weight_order = (1, 4, 2, 3)
            target_order = (4, 2, 3)
    
            # Setup FFT-restrict pattern and weightage with Kroneckers
            weight_mix = ParTensor(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape, "ParTensor_SCONV:($(layer))")
            restrict_dft = ParKron((restrict_z * fourier_z) ⊗ (restrict_y * fourier_y), (restrict_x * fourier_x) ⊗ (restrict_t * fourier_t) ⊗ ParIdentity(Complex{T}, config.nc_lift))
            
            push!(weight_mixes, weight_mix)
            
            weight_mix = distribute(weight_mix, [1, config.partition...])
            restrict_dft = distribute(restrict_dft, config.partition)
    
            sconv = restrict_dft' * weight_mix * restrict_dft
    
            return sconv
        end
    
        # Lift Channel dimension
        lifts = ParMatrix(T, config.nc_lift, config.nc_in, "ParMatrix_LIFTS:(1)")
        bias = ParMatrix(T, config.nc_lift, 1, "ParMatrix_BIAS:(1)")

        lifts = distribute(lifts)
        bias = distribute(bias)

        push!(biases, bias)
    
        for i in 1:config.nblocks
    
            sconv_layer = spectral_convolution(i)

            conv_layer = ParMatrix(T, config.nc_lift, config.nc_lift, "ParMatrix_SCONV:($(i))")
            bias = ParMatrix(T, config.nc_lift, 1, "ParMatrix_SCONV:($(i))")
    
            conv_layer = distribute(conv_layer)
            bias = distribute(bias)
    
            push!(sconv_biases, bias)
            push!(sconvs, sconv_layer)
            push!(convs, conv_layer)
        end
    
        # Uplift channel dimension once more
        uc = ParMatrix(T, config.nc_mid, config.nc_lift, "ParMatrix_LIFTS:(2)")
        bias = ParMatrix(T, config.nc_mid, 1, "ParMatrix_BIAS:(2)")
    
        uc = distribute(uc)
        bias = distribute(bias)
    
        push!(biases, bias)
        push!(projects, uc)
    
        # Project channel dimension
        pc = ParMatrix(T, config.nc_out, config.nc_mid, "ParMatrix_LIFTS:(3)")
        bias = ParMatrix(T, config.nc_out, 1, "ParMatrix_BIAS:(3)")
    
        pc = distribute(pc)
        bias = distribute(bias)
    
        push!(biases, bias)
        push!(projects, pc)
    
        new(config, lifts, convs, sconvs, biases, sconv_biases, projects, weight_mixes)
    end
end

"""
    initModel(model::Model)

Initializes the [Model](@ref) by allocating and setting the initial values for the parameters. 

# Arguments
- `model`: An instance of [Model](@ref) to be initialized.

# Returns
Returns the initialized parameters `θ` which may be placed on GPU if the global `gpu_flag` is set to `true`.
"""
function initModel(model::Model)
    θ = init(model.lifts)
    for operator in Iterators.flatten((model.convs, model.sconvs, model.biases, model.sconv_biases, model.projects))
        init!(operator, θ)
    end
    gpu_flag && (θ = gpu(θ))
    return θ
end
