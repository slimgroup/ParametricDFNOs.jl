using Parameters
 using Pkg
 Pkg.activate("./")

# include("src/models/TDFNO_2D/TDFNO_2D.jl")
# include("src/utils.jl")

# using .TDFNO_2D
# using .UTILS
# using MPI
# using Zygote
# using DrWatson
using ParametricOperators
# using CUDA

# gpu = ParametricOperators.gpu

# MPI.Init()

# comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# size = MPI.Comm_size(comm)

@with_kw struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nt::Int = 51
    nc_in::Int = 4
    nc_mid::Int = 128
    nc_lift::Int = 20
    nc_out::Int = 1
    mx::Int = 4
    my::Int = 4
    mt::Int = 4
    nblocks::Int = 1
    dtype::DataType = Float32
    partition::Vector{Int} = [1, 4] 
    TuckerRank::Vector{Int} = [2,2,2,2,2,2] ##need to adjust this
end
config = ModelConfig()


mutable struct Model
    config::ModelConfig
    lifts::ParKron
    convs::Vector
    # sconvs::Vector
    sconv::Function
    biases::Vector
    sconv_biases::Vector
    projects::Vector
    weight_mixes::Vector

    #first, function to get Tucker factors (ParMatrix)
    function TuckerCompress(size::Vector,rank::Vector,spatiotemp::Vector,restriction::Vector)
        factors = []
        n = length(size)
        T = config.dtype
    
        core = ParMatrix(Complex{T},rank[1],prod(rank[2:n]))
        push!(factors,ParMatrix(Complex{T},size[1],rank[1]))
        for i  = 2:length(size)
            push!(factors,ParMatrix(Complex{T},rank[i],size[i]))
        end
        return ParTucker(core,factors,spatiotemp,restriction)
    end


    function Model(config::ModelConfig)

        T = config.dtype
        
        # sconvs = []
        convs = []
        projects = []
        sconv_biases = []
        biases = []
        weight_mixes = []
    
        weight_shape = [config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my,
        config.mt, config.nblocks]
        weight_mix = TuckerCompress(weight_shape,config.TuckerRank,[config.nx, config.ny, config.nt],
        [config.mx, config.my, config.mt])
        push!(weight_mixes, weight_mix)

        function sconv(θ, x::AbstractArray, layer::Int; w::ParTucker = weight_mix)
            T = config.dtype
    
                
            
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
            Id = ParIdentity(Complex{T},b)
            x = restrict_dft(x)
            x = reshape(x,(i,b,2*mx,2*my,mt))
            z = x
        
            y = vcat([(Id ⊗ (w.factors[1](θ)*w.core(θ)*(w.factors[5](θ)[:,k]⊗w.factors[4](θ)[:,j] ⊗
           w.factors[3](θ)[:,i]⊗w.factors[2](θ))))*
             vec(z[:,:,i,j,k]) for i = 1:Domain(w.factors[3]), j = 1:Domain(w.factors[4]), k = 1:Domain(w.factors[5])]...)
         
            y = reshape(y,(o,b,2*mx,2*my,mt))  
            y = reshape(y,(:,b))
            y = restrict_dft'(y)   
        end
    

        # function spectral_convolution(layer::Int)
    
        #     # Build 3D Fourier transform with real-valued FFT along time
        #     fourier_x = ParDFT(Complex{T}, config.nx)
        #     fourier_y = ParDFT(Complex{T}, config.ny)
        #     fourier_t = ParDFT(T, config.nt)
    
        #     # Build restrictions to low-frequency modes
        #     restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:config.mx, config.nx-config.mx+1:config.nx])
        #     restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:config.my, config.ny-config.my+1:config.ny])
        #     restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:config.mt])
            
        #     restrict_dft = ParKron((restrict_y * fourier_y) ⊗ (restrict_x * fourier_x), (restrict_t * fourier_t) ⊗ ParIdentity(T, config.nc_lift))
    
        #     sconv = restrict_dft' * weight_mix(layer) * restrict_dft
        #     # sconv =  weight_mix(layer) 

    
        #     return sconv
        # end

    
        # Lift Channel dimension
        lifts = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_lift, config.nc_in, "ParMatrix_LIFTS:(1)"))
        bias = ParBroadcasted(ParMatrix(T, config.nc_lift, 1, "ParMatrix_BIAS:(1)"))
    
        # lifts = distribute(lifts, config.partition)
    
        push!(biases, bias)
    
        for i in 1:config.nblocks
    
            # sconv_layer = spectral_convolution(i)
            conv_layer = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_lift, config.nc_lift, "ParMatrix_SCONV:($(i))"))
            bias = ParBroadcasted(ParMatrix(T, config.nc_lift, 1, "ParMatrix_SCONV:($(i))"))
    
            # conv_layer = distribute(conv_layer, config.partition)
    
            push!(sconv_biases, bias)
            # push!(sconvs, sconv_layer)
            push!(convs, conv_layer)
        end
    
        # Uplift channel dimension once more
        uc = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_mid, config.nc_lift, "ParMatrix_LIFTS:(2)"))
        bias = ParBroadcasted(ParMatrix(T, config.nc_mid, 1, "ParMatrix_BIAS:(2)"))
    
        # uc = distribute(uc, config.partition)
    
        push!(biases, bias)
        push!(projects, uc)
    
        # Project channel dimension
        pc = ParKron(ParIdentity(T,config.ny) ⊗ ParIdentity(T,config.nx), ParIdentity(T,config.nt) ⊗ ParMatrix(T, config.nc_out, config.nc_mid, "ParMatrix_LIFTS:(3)"))
        bias = ParBroadcasted(ParMatrix(T, config.nc_out, 1, "ParMatrix_BIAS:(3)"))
    
        # pc = distribute(pc, config.partition)
    
        push!(biases, bias)
        push!(projects, pc)
        new(config, lifts, convs, sconv, biases, sconv_biases, projects, weight_mixes)
    end
end

# function initModel(model::Model)
#     θ = init(model.lifts)
#     init!(weight_mix,θ)
#     # for operator in Iterators.flatten((model.sconvs))
#     #     init!(operator, θ)
#     # end
    
#     # gpu_flag && (θ = gpu(θ))
#     return θ
# end
model = Model(config);


# θ = init(model.sconvs[1])
# θ = init(weight_mix)
# lay = 1
# model.spectral_convolution(weight_mix,θ,x,layer)
