using Pkg
Pkg.activate("./")

using ParametricOperators
using Parameters

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

rank = 0
function print_rd(op::Any)
    if rank > -1
        println(Range(op), " x ", Domain(op)," @ Rank ", rank)
    end
end

config = ModelConfig()
T = config.dtype

input_shape = (config.nc_lift, 2*config.mx, 2*config.my, config.mt)
weight_shape = (config.nc_lift, config.nc_lift, 2*config.mx, 2*config.my, config.mt)

# Specify Einsum multiplication
input_order = (1, 2, 3, 4)
weight_order = (5, 1, 2, 3, 4)
target_order = (5, 2, 3, 4)

weight_mix = ParMatrixN(Complex{T}, weight_order, weight_shape, input_order, input_shape, target_order, input_shape) 
print_rd(weight_mix)
weight_mix = distribute(weight_mix, config.partition)
print_rd(weight_mix)
