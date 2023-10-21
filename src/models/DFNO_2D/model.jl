@with_kw struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nt_in::Int = 51
    nt_out::Int = 51
    nc_in::Int = 4
    nc_mid::Int = 128
    nc_lift::Int = 20
    nc_out::Int = 1
    mx::Int = 4
    my::Int = 4
    mt::Int = 4
    nblocks::Int = 1
    dtype::DataType = Float64
    partition::Vector{Int} = [1, 2, 2, 1]
end

mutable struct Model
    config::ModelConfig
    lifts::ParKron
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
    
        function spectral_convolution(layer::Int)
    
            # Build 3D Fourier transform with real-valued FFT along time
            fourier_x = ParDFT(Complex{T}, config.nx)
            fourier_y = ParDFT(Complex{T}, config.ny)
            fourier_t = ParDFT(T, config.nt_out)
    
            # Build restrictions to low-frequency modes
            restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:config.mx, config.nx-config.mx+1:config.nx])
            restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:config.my, config.ny-config.my+1:config.ny])
            restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:config.mt])
    
            input_shape = (config.nc_lift, 2*config.mx, 2*config.my, config.mt)
            weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt)
    
            input_order = (1, 2, 3, 4)
            weight_order = (5, 1, 2, 3, 4)
            target_order = (5, 2, 3, 4)
    
            # Setup FFT-restrict pattern and weightage with Kroneckers
            weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape, "ParMatrixN_SCONV:($(layer))")
            restrict_dft = (restrict_t * fourier_t) ⊗ (restrict_y * fourier_y) ⊗ (restrict_x * fourier_x) ⊗ ParIdentity(T, config.nc_lift)
            
            push!(weight_mixes, weight_mix)

            weight_mix = distribute(weight_mix, config.partition)
            restrict_dft = distribute(restrict_dft, config.partition)
    
            sconv = restrict_dft' * weight_mix * restrict_dft
    
            return sconv
        end
    
        # Lift Channel dimension
        lifts = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParMatrix(T, config.nc_lift, config.nc_in, "ParMatrix_LIFTS:(1)")
        bias = ParIdentity(T,config.nt_in) ⊗ ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx) ⊗ ParDiagonal(T, config.nc_lift, "ParDiagonal_BIAS:(1)") # TODO: Rearrange code for all bias so it makes more sense mathematically
    
        lifts = distribute(lifts, config.partition)
        bias = distribute(bias, config.partition)
    
        push!(biases, bias)
    
        for i in 1:config.nblocks
    
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
    
        new(config, lifts, convs, sconvs, biases, sconv_biases, projects, weight_mixes)
    end
end

function initModel(model::Model)
    θ = init(model.lifts)
    for operator in Iterators.flatten((model.sconvs, model.convs, model.biases, model.sconv_biases, model.projects))
        init!(operator, θ)
    end
    return θ
end
