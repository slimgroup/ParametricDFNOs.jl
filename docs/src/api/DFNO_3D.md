# 3D Time varying FNO

!!! tip "3D Time varying"
    The implementation allows for discretization along time dimension to be 1 (only 1 time step). But you can also treat time as any other dimension, so this could also be as a generic 4D FNO

## Model Definition

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["model.jl"]
```

## Forward Pass

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["forward.jl"]
```

## Training

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["train.jl"]
```

## Data Loading

!!! warning "Critical component"
    See [this]() for instructions on how to set it up properly.

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["data.jl"]
```

!!! tip "Distributed read for complex storage scenarios"
    View an example of how you can extend this distributed read to a complex storage scheme [here]().

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D.UTILS]
Order  = [:type, :function]
Pages = ["utils.jl"]
```

## Plotting

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["plot.jl"]
```

## Checkpoints

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["weights.jl"]
```
