using MPI
using CUDA
using Zygote
using ParametricDFNOs.DFNO_2D
using ParametricDFNOs.UTILS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

global gpu_flag = parse(Bool, get(ENV, "DFNO_2D_GPU", "0"))
DFNO_2D.set_gpu_flag(gpu_flag)

# Julia requires you to manually assign the gpus, modify to your case.
DFNO_2D.gpu_flag && (CUDA.device!(rank % 4))
partition = [1, pe_count]

nx, ny, nt = 20, 20, 30
modes, nblocks = 8, 4

@assert MPI.Comm_size(comm) == prod(partition)
modelConfig = DFNO_2D.ModelConfig(nx=nx, ny=ny, nt=nt, mx=modes, my=modes, mt=modes, nblocks=nblocks, partition=partition, dtype=Float32)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nt) ÷ prod(partition)
output_size = input_size * model.config.nc_out ÷ model.config.nc_in

x_sample = rand(modelConfig.dtype, input_size, 1)
y_sample = rand(modelConfig.dtype, output_size, 1)

DFNO_2D.gpu_flag && (y_sample = cu(y_sample))

@time y = DFNO_2D.forward(model, θ, x_sample)
@time y = DFNO_2D.forward(model, θ, x_sample)
@time y = DFNO_2D.forward(model, θ, x_sample)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_2D.forward(model, params, x_sample), y_sample)
    return loss
end

rank == 0 && println("STARTED GRADIENT SCALING")

@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]

MPI.Finalize()
