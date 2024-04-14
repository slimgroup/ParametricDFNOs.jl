# Utilities

!!! note "Distributed Loss Function"
    We provide a distributed relative L2 loss but most distributed loss functions should be straight-forward to build with [`ParametricOperators.jl`](https://github.com/slimgroup/ParametricOperators.jl)

```@autodocs
Modules = [ParametricDFNOs.UTILS]
Order  = [:type, :function]
Pages = ["utils.jl"]
```

### GPU Helpers

```@autodocs
Modules = [ParametricDFNOs.DFNO_2D]
Order  = [:type, :function]
Pages = ["DFNO_2D.jl"]
```

```@autodocs
Modules = [ParametricDFNOs.DFNO_3D]
Order  = [:type, :function]
Pages = ["DFNO_3D.jl"]
```
