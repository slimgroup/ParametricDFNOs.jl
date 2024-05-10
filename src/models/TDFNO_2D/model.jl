


@with_kw struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nt::Int = 51
    nc_in::Int = 4
    nc_mid::Int = 128
    nc_lift::Int = 20
    nc_out::Int = 1
    mx::Int = 20
    my::Int = 20
    mt::Int = 20
    nblocks::Int = 4
    dtype::DataType = Float32
    partition::Vector{Int} = [1, 4] 
    TuckerRank::Vector{Int} = [5,5,3,3,3,1] ##need to adjust this
end


mutable struct Model
    config::ModelConfig
    lifts::ParKron
    convs::Vector
    sconv::Function
    biases::Vector
    sconv_biases::Vector
    projects::Vector
    weight_mix::ParTucker

    function Model(config::ModelConfig)

        function TuckerCompress(size::Vector,rank::Vector,spatiotemp::Vector,restriction::Vector)
            factors = []
            n = length(size)
            T = config.dtype
        
            core = ParMatrix(Complex{T},rank[1],prod(rank[2:n]))
            push!(factors,ParMatrix(Complex{T},size[1],rank[1]))
            for i  = 2:length(size)
                push!(factors,ParMatrix(Complex{T},rank[i],size[i]))
            end
            
            block_diagonals = []
            for i = 3:5
                push!(block_diagonals, ParBlockDiagonal([factors[i][:, j] for j in 1:size[i]]...))
            end
            return ParTucker(core,factors,spatiotemp,restriction,block_diagonals)
        end

        T = config.dtype
        
        convs = []
        projects = []
        sconv_biases = []
        biases = []
    
        weight_shape = [config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my,
        config.mt, config.nblocks]
        weight_mix = TuckerCompress(weight_shape,config.TuckerRank,[config.nx, config.ny, config.nt],
        [config.mx, config.my, config.mt]);

        function sconv(θ, x::AbstractArray, layer::Int; w::ParTucker = weight_mix)
            b = size(x,2)

            o = Range(w.factors[1]) # U1 is o \times k_1
            i = Domain(w.factors[2]) # U2 is k_2 \times i

            nx = w.input_dimension[1]; ny = w.input_dimension[2]; nt = w.input_dimension[3]
            mx = w.restriction[1]; my = w.restriction[2]; mt = w.restriction[3]

            fourier_x = ParDFT(Complex{T}, nx)
            fourier_y = ParDFT(Complex{T}, ny)
            fourier_t = ParDFT(T, nt)

            # Build restrictions to low-frequency modes
            restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:mx,nx-mx+1:nx])
            restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:my,ny-my+1:ny])
            restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:mt])

            restrict_dft = ParKron((restrict_y * fourier_y) ⊗ (restrict_x * fourier_x), (restrict_t * fourier_t) ⊗ ParIdentity(T, i))

            Id = ParIdentity(Complex{T}, b)
            x = restrict_dft(x)

            ### WRONG RESHAPE: SHOULD RESHAPE like : x = reshape(x,(i,mt,2*mx,2*my,b)). 
            ### In fact you should only reshape like (:, b) and only fix factors accordingly
            x = reshape(x,(i,b,2*mx,2*my,mt))

            P = w.factors[6](θ)[:,layer] ⊗ w.block_diagonals[3](θ) ⊗ w.block_diagonals[2](θ) ⊗ w.block_diagonals[1](θ) ⊗ w.factors[2](θ)
            y = P * vec(x)

            y = w.factors[1](θ) * w.core(θ) * reshape(y, Domain(w.core(θ)), :)
            y = reshape(y,(:,b))
            y = restrict_dft'(y)
        end      
        
    
        # Lift Channel dimension
        lifts = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_lift, config.nc_in, "ParMatrix_LIFTS:(1)"))
        bias = ParBroadcasted(ParMatrix(T, config.nc_lift, 1, "ParMatrix_BIAS:(1)"))
    
        lifts = distribute(lifts, config.partition)
    
        push!(biases, bias)
    
        for i in 1:config.nblocks
    
            conv_layer = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_lift, config.nc_lift, "ParMatrix_SCONV:($(i))"))
            bias = ParBroadcasted(ParMatrix(T, config.nc_lift, 1, "ParMatrix_SCONV:($(i))"))
    
            conv_layer = distribute(conv_layer, config.partition)
    
            push!(sconv_biases, bias)
            push!(convs, conv_layer)
        end
    
        # Uplift channel dimension once more
        uc = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_mid, config.nc_lift, "ParMatrix_LIFTS:(2)"))
        bias = ParBroadcasted(ParMatrix(T, config.nc_mid, 1, "ParMatrix_BIAS:(2)"))
    
        uc = distribute(uc, config.partition)
    
        push!(biases, bias)
        push!(projects, uc)
    
        # Project channel dimension
        pc = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_out, config.nc_mid, "ParMatrix_LIFTS:(3)"))
        bias = ParBroadcasted(ParMatrix(T, config.nc_out, 1, "ParMatrix_BIAS:(3)"))
    
        pc = distribute(pc, config.partition)
    
        push!(biases, bias)
        push!(projects, pc)
    
        new(config, lifts, convs, sconv, biases, sconv_biases, projects, weight_mix)
    end
end

function initModel(model::Model)
    θ = init(model.lifts)
    init!(model.weight_mix,θ)
    for operator in Iterators.flatten((model.convs, model.biases, model.sconv_biases, model.projects))
        init!(operator, θ)
    end
    gpu_flag && (θ = gpu(θ));
    return θ;
end

function print_storage_complexity(config::ModelConfig; batch=1)
    
    # Prints the approximate memory required for forward and backward pass

    multiplier = Dict([(Float16, 16), (Float32, 32), (Float64, 64)])

    x_shape = (config.nc_in, config.nx, config.ny, config.nt)
    y_shape = (config.nc_out, config.nx, config.ny, config.nt)
    weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt)

    weights_count = 0.0
    data_count = 0.0

    # Storage costs for x and y
    data_count = (prod(x_shape) + prod(y_shape))

    # Lift costs for 2 sums + 1 target
    lift_count = 3 * config.nc_lift * prod(x_shape) / config.nc_in 

    # Sconv costs for 2 sums + 1 target
    sconv_count = 3 * config.nc_lift * prod(x_shape) / config.nc_in 
    
    # Projects costs for 2 sums + 1 target
    projects_count1 = 3 * config.nc_mid * prod(x_shape) / config.nc_in 
    projects_count2 = 3 * config.nc_out * prod(x_shape) / config.nc_in 

    # Most # of Kronecker stores (2x Range) in PO * max(input)
    data_count += max(lift_count, sconv_count, projects_count1, projects_count2)

    # println(batch * lift_count * multiplier[config.dtype] / 8e+6)
    # println(batch * sconv_count * multiplier[config.dtype] / 8e+6)
    # println(batch * projects_count1 * multiplier[config.dtype] / 8e+6)
    # println(batch * projects_count2 * multiplier[config.dtype] / 8e+6)

    # Lifts weights and bias
    weights_count += (config.nc_lift * config.nc_in) + config.nc_lift

    # Sconv layers weights
    for i in 1:config.nblocks

        # Par Matrix N in restriction space,
        weights_count += prod(weight_shape)
        
        # Convolution and bias
        weights_count += (config.nc_lift * config.nc_lift) + config.nc_lift
    end

    # Projects 1 weights and bias
    weights_count += (config.nc_lift * config.nc_mid) + config.nc_mid

    # Projects 2 weights and bias
    weights_count += (config.nc_mid * config.nc_out) + config.nc_out

    # For Francis b=8 passes b=9 fails with F32, Richard b=5 passes b=6 fails after couple batches (forward and backward)
    # For Francis & Richard b=25 passes b=26 fails with F32

    w_scale = 1.88 # Empirically chosen (Due to Dict ?)
    c_scale = batch * 0.97 # Empirically chosen
    g_scale = 4.5 # Empirically chosen, for 3.0 for Francis, 4.5 for Richard. Gradient is only ~67% as memory efficient as Francis 

    w_mb = w_scale * weights_count * multiplier[config.dtype] / 8e+6
    c_gb = c_scale * data_count * multiplier[config.dtype] / 8e+9
    g_gb = c_gb * g_scale

    t_gb = (w_mb / 8e+3) + max(c_gb, g_gb)

    output = @sprintf("DFNO_2D (batch=%d) | ~ %.2f MB for weights | ~ %.2f GB for forward pass | ~ %.2f GB for backward pass | Total ~ %.2f GB |", batch, w_mb, c_gb, g_gb, t_gb)

    @info output
end
