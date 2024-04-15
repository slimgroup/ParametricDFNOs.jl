# 2D Time varying FNO

!!! tip "2D Time varying"
    The implementation allows for discretization along time dimension to be 1 (only 1 time step). But you can also treat time as any other dimension, so this could also be as a generic 3D FNO

## 2D Model

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["model.jl"]
```

## 2D Forward Pass

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["forward.jl"]
```

## 2D Training

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["train.jl"]
```

## 2D Data Loading

!!! warning "Critical component"
    See [Data Partitioning](@ref) for instructions on how to set it up properly.

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["data.jl"]
```

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D.UTILS]
Order  = [:type, :function]
Pages = ["utils.jl"]
```

## 2D Plotting

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["plot.jl"]
```

## 2D Checkpoints

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["weights.jl"]
```
