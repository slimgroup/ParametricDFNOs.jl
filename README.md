# Distributed Fourier Neural Operators

## Description

The Fourier Neural Operator (FNO) is a neural network designed to approximate solutions to partial differential equations (PDEs), specifically for two-phase flows such as CO2 plume evolution in carbon capture and storage (CCS) processes, atmospheric fields, etc. By transforming inputs to frequency space using spectral convolution operators and leveraging the efficiency of Fourier transforms, FNOs offer a significant speed-up in simulation times compared to traditional methods. This project involves extending FNOs to operate in a distributed fashion for handling large-scale, realistic three-dimensional two-phase flow problems.

We offer support for 2D and 3D time varying problems.
## Getting Started

### Dependencies

- `Julia 1.8.5`
- MPI distribution.

### Installing

```
git clone https://github.com/turquoisedragon2926/dfno.git
cd dfno
julia
> ] instantiate .
> using MPI
> MPI.install_mpiexecjl()
```

NOTE: Add mpiexecjl to your PATH

### Executing program on custom dataset for 2D time varying
1. Open `examples/training/training_2d.jl`

2. Update data reading function.

* We have provided a wrapper for distributed reading if you do not have a data reading function set up. In order to use this, implement the following two functions:

```
function dist_read_x_tensor(file_name, key, indices)
    data = nothing
    h5open(file_name, "r") do file
        dataset = file[key]
        data = dataset[indices...]
    end
    return reshape(data, 1, (size(data)...))
end
```

```
function dist_read_y_tensor(file_name, key, indices)
    data = nothing
    h5open(file_name, "r") do file
        dataset = file[key]
        data = dataset[indices...]
    end
    return reshape(data, 1, (size(data)...))
end
```

* See `examples/perlmutter/data.jl` on example for how to use this for even cases where your samples are stored across multiple files.

3. Pass required hyperparams following the ModelConfig:

* Line to modify:

```
modelConfig = DFNO_2D.ModelConfig(nblocks=4, partition=partition)
```

* Options:
```
struct ModelConfig
    nx::Int = 64
    ny::Int = 64
    nt::Int = 51
    nc_in::Int = 4
    nc_mid::Int = 128
    nc_lift::Int = 20
    nc_out::Int = 1
    mx::Int = 8
    my::Int = 8
    mt::Int = 4
    nblocks::Int = 4
    dtype::DataType = Float32
    partition::Vector{Int} = [1, 4]
end
```

4. Modify parameters to train config based on requirement:

* Line to modify:
```
trainConfig = DFNO_2D.TrainConfig(
    epochs=200,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
)
```

* Options:
```
struct TrainConfig
    nbatch::Int = 2
    epochs::Int = 1
    seed::Int = 1234
    plot_every::Int = 1
    learning_rate::Float32 = 1f-4
    x_train::Any
    y_train::Any
    x_valid::Any
    y_valid::Any
end
```

5. Execute the program with number of required workers

```
mpiexecjl --project=./ -n 4 julia examples/training/training_2d.jl
```

## Help

Common problems or issues.

```
This section will be updated with common issues and fixes
```

## Authors

[Richard Rex](https://www.linkedin.com/in/richard-rex/) - Georgia Institute of Technology

## Version History

* v2.0.0
    * Various bug fixes and memory optimizations
    * Ability to scale to $512^3$ across 500 GPUs

* v1.0.0
    * Initial working DFNO

## License

This project is licensed under the Creative Commons License - see the LICENSE.md file for details

## Acknowledgments

This research was carried out with the support of Georgia Research Alliance, Extreme Scale Solutions and partners of the ML4Seismic Center.
