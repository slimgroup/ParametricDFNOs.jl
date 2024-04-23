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

We also use `PyPlot`, so you would need to do:

```shell
python3 -m pip install matplotlib
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

## Data Partitioning

We have a [2D Data Loading](@ref) `struct` to store information about our data, consider [Training 2D Time varying FNO](@ref)

```julia
dataConfig = DFNO_2D.DataConfig(modelConfig=modelConfig, 
                                x_key = "perm",
                                x_file = perm_store_path_jld2,
                                y_key="conc",
                                y_file = conc_store_path_jld2)
```

Consider the following dimensions:

`c, t, x, y, z` where

```
x, y, z - Spatial Dimensions
t - Time Dimension
c - Channel Dimension
```

Data is considered to be combined along certain dimensions:

`ct, xy` for `DFNO_2D`

`ctx, yz` for `DFNO_3D`

The partition array which is a two dimensional array specifies across how many workers is a given combined dimension split across.

By default we do:

```julia
comm = MPI.COMM_WORLD
pe_count = MPI.Comm_size(comm)

partition = [1, pe_count]
```

The models are implemented to modify the operators according to the specified partition. We suggest you leave this as it is.

!!! warning "Running into assertion errors"
    If you run into any assertion errors that the number of workers do not divide the data evenly, please make a github issue

We provide a distributed read wrapper which allows you to read data seamlessly.

Simply implement:

```julia

# Returns tensor of size (in_channels, size(indices)...)
function dist_read_x_tensor(file_name, key, indices)

# Returns tensor of size (out_channels=1, size(indices)...)
function dist_read_y_tensor(file_name, key, indices)
```

!!! note "in channels"
    The number of `in_channels` you specify at [Model Setup](@ref) is `data_channels + 3` for `DFNO_2D` and  `data_channels + 4` for `DFNO_3D`. This is to account for the grid data we include for each of the dimensions in the FNO.

!!! warning "out channels"
    Currently, distributed wrapper only supports reading for the case where out channel is 1. You can implement your own read function or wait for a version update

### `DFNO_2D`

Here, `indicies` for the `dist_read_x_tensor` represents 
```
(x_start:x_end, y_start:y_end, sample_start:sample_end)
```

and the `indices` for the `dist_read_y_tensor` represents:
```
(x_start:x_end, y_start:y_end, t_start:t_end, sample_start:sample_end)
```

### `DFNO_3D`

Here, `indicies` for the `dist_read_x_tensor` represents 
```
(x_start:x_end, y_start:y_end, z_start:z_end, sample_start:sample_end)
```

and the `indices` for the `dist_read_y_tensor` represents:
```
(x_start:x_end, y_start:y_end, z_start:z_end, t_start:t_end, sample_start:sample_end)
```

Now you can use `loadDistData` from [2D Data Loading](@ref) or [3D Data Loading](@ref)

This can also be extended to complex storage regime. Consider the following case:

```
samples/
├── sample1/
│   ├── inputs.jld2
│   └── outputs.jld2
└── sample2/
    ├── inputs.jld2
    └── outputs.jld2
```

We can do [Custom 3D Time varying FNO](@ref)

## Training wrapper

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
