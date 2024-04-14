using MPI
using CUDA
using Zygote
using ParametricDFNOs.DFNO_3D
using ParametricDFNOs.UTILS

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

global gpu_flag = parse(Bool, get(ENV, "DFNO_3D_GPU", "0"))
UTILS.set_gpu_flag(gpu_flag)

# Julia requires you to manually assign the gpus, modify to your case.
DFNO_3D.gpu_flag && (CUDA.device!(rank % 4))
partition = [1, pe_count]

nx, ny, nz, nt = 20, 20, 20, 30
modes, nblocks = 8, 4

@assert MPI.Comm_size(comm) == prod(partition)
modelConfig = DFNO_3D.ModelConfig(nx=nx, ny=ny, nz=nz, nt=nt, mx=modes, my=modes, mz=modes, mt=modes, nblocks=nblocks, partition=partition, dtype=Float32)

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nz * model.config.nt) ÷ prod(partition)
output_size = input_size * model.config.nc_out ÷ model.config.nc_in

x_sample = rand(modelConfig.dtype, input_size, 1)
y_sample = rand(modelConfig.dtype, output_size, 1)

DFNO_3D.gpu_flag && (y_sample = cu(y_sample))

@time y = DFNO_3D.forward(model, θ, x_sample)
@time y = DFNO_3D.forward(model, θ, x_sample)
@time y = DFNO_3D.forward(model, θ, x_sample)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, params, x_sample), y_sample)
    return loss
end

rank == 0 && println("STARTED GRADIENT SCALING")

@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]

MPI.Finalize()
