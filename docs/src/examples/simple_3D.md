### Simple 3D forward and gradient pass

!!! note "Jump right in"
    To get started, you can run some [examples](https://github.com/turquoisedragon2926/ParametricDFNOs.jl-Examples)

Make sure to add necessary dependencies. You might also need to load a proper MPI implementation based on your hardware.

```julia
julia> ]
(v1.9) activate /path/to/your/environment 
(venv) add MPI Zygote CUDA ParametricDFNOs
```

!!! warning "To run on multiple GPUs"
    If you wish to run on multiple GPUs, make sure the GPUs are binded to different tasks. The approach we use is to unbind our GPUs on request and assign manually:

    ```julia
    CUDA.device!(rank % 4)
    ```

    which might be different if you have more or less than 4 GPUs per node. Also, make sure your MPI distribution is functional.

```julia
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
DFNO_3D.set_gpu_flag(gpu_flag)

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

MPI.Barrier()

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_3D.forward(model, params, x_sample), y_sample)
    return loss
end

rank == 0 && println("STARTED GRADIENT SCALING")

@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]

MPI.Finalize()
```

If you have [`mpiexecjl`](https://juliaparallel.org/MPI.jl/stable/usage/#Installation) set up, you can run the above by doing:

```shell
mpiexecjl --project=/path/to/your/environment -n NTASKS julia code_above.jl
```

OR if you have a HPC cluster with [`slurm`](https://slurm.schedmd.com/documentation.html) set up, you can do:

```shell
salloc --gpus=NTASKS --time=01:00:00 --ntasks=NTASKS --gpus-per-task=1 --gpu-bind=none
srun julia --project=/path/to/your/environment code_above.jl
```

!!! warning "Allocation"
    Your `salloc` might look different based on your HPC cluster

