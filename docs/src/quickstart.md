## Installation

Add `ParametricDFNOs.jl` as a dependency to your environment.

To add, either do:

```julia
julia> ]
(v1.9) add ParametricDFNOs
```

OR

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/environment")
julia> Pkg.add("ParametricDFNOs")
```

!!! note "Jump right in"
    To get started, you can also try running some [examples](https://github.com/turquoisedragon2926/ParametricDFNOs.jl-Examples)

## Setup

Make sure to include the right dependency you plan on using in your environment

```julia
using MPI
using CUDA

# If you plan on using the 2D Time varying FNO or 3D FNO.
using ParametricDFNOs.DFNO_2D

# If you plan on using the 3D Time varying FNO or 4D FNO.
using ParametricDFNOs.DFNO_3D
```

### MPI setup

!!! note "MPI Distribution"
    Make sure you have a functional MPI Distribution set up
    
All code must be wrapped in:

```julia
MPI.Init()

### Code here ###

MPI.Finalize()
```

!!! tip "Change to custom use case"
    We show the usage for `ParametricDFNOs.DFNO_2D` but the extension to the other FNOs should be as simple as changing the number. Please refer to the API for exact differences.

### GPU usage

!!! note "Default behavior"
    By default, the package will be set to use the GPU based the whether the `DFNO_2D_GPU` flag was set during compile time of the package

You can set the GPU flag by using:

```shell
export DFNO_2D_GPU=1
```

and 

```julia
global gpu_flag = parse(Bool, get(ENV, "DFNO_2D_GPU", "0"))
DFNO_2D.set_gpu_flag(gpu_flag)
```

!!! warning "Binding GPUs"
    If you wish to run on multiple GPUs, make sure the GPUs are binded to different tasks. The approach we choose in our examples is to unbind our GPUs on request and assign manually:

    ```julia
    using CUDA

    CUDA.device!(rank % 4)
    ```

    which might be different if you have more or less than 4 GPUs per node.

## Data Partitioning

Data is considered to be combined along certain dimensions:


## Model Setup

Define a [2D Model](@ref) configuration:

```julia
modelConfig = DFNO_2D.ModelConfig(nx=20, ny=20, nt=50, mx=4, my=4, mt=4, nblocks=4, partition=partition, dtype=Float32)
```

Define some random inputs to operate on:

```julia
input_size = (modelConfig.nc_in * modelConfig.nx * modelConfig.ny * modelConfig.nt) ÷ prod(partition)
output_size = input_size * modelConfig.nc_out ÷ modelConfig.nc_in

x = rand(modelConfig.dtype, input_size, 1)
y = rand(modelConfig.dtype, output_size, 1)
```

### Initializing model
```julia
model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)
```

### Forward and backward pass

See [Simple 2D forward and gradient pass](@ref) for a full example.

```julia
DFNO_2D.forward(model, θ, x)
```

!!! note "Distributed Loss Function"
    We provide a distributed relative L2 loss but most distributed loss functions should be straight-forward to build with [`ParametricOperators.jl`](https://github.com/slimgroup/ParametricOperators.jl)

To compute gradient:

```julia
using Zygote
using ParametricDFNOs.UTILS

gradient(params -> loss_helper(UTILS.dist_loss(DFNO_2D.forward(model, params, x), y)), θ)[1]
```

### Training wrapper

We also provide a training wrapper to train out the box. See [Training 2D Time varying FNO](@ref) for a full example.

Define a [2D Training](@ref) configuration:

```julia
trainConfig = DFNO_2D.TrainConfig(
    epochs=10,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    plot_every=1
)
```

And train using:

```julia
DFNO_2D.train!(trainConfig, model, θ)
```
